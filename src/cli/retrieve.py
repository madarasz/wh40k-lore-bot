"""CLI command for testing hybrid retrieval."""

import asyncio
import os
import sys

import click
import structlog
from dotenv import load_dotenv

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.orchestration import QueryOrchestrator, RetrievalResult
from src.rag.context_expander import ContextExpander
from src.rag.hybrid_retrieval import HybridRetrievalService
from src.rag.vector_store import ChromaVectorStore
from src.repositories.bm25_repository import BM25Repository
from src.utils.exceptions import ConfigurationError

load_dotenv()
logger = structlog.get_logger(__name__)

# Display constants
PREVIEW_LENGTH = 200


def _create_retrieval_orchestrator() -> QueryOrchestrator:
    """Create QueryOrchestrator configured for retrieval-only mode.

    Returns:
        QueryOrchestrator with retrieval services only (no LLM)

    Raises:
        ConfigurationError: If required services cannot be initialized
    """
    # Initialize vector store
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "data/chroma-db/")
    vector_store = ChromaVectorStore(storage_path=chroma_db_path)

    chunk_count = vector_store.count()
    if chunk_count == 0:
        raise ConfigurationError(
            "Vector store is empty. Run 'poetry run ingest' to populate the database."
        )

    logger.info("vector_store_loaded", chunk_count=chunk_count)

    # Initialize BM25 repository
    bm25_repo = BM25Repository()
    bm25_repo.load_index()

    if not bm25_repo.is_index_built():
        raise ConfigurationError(
            "BM25 index not found. Run 'poetry run build-bm25' to create the index."
        )

    logger.info("bm25_index_loaded", chunk_count=len(bm25_repo.chunk_ids))

    # Initialize remaining services
    embedding_gen = EmbeddingGenerator()
    hybrid_service = HybridRetrievalService(
        vector_store=vector_store,
        bm25_repository=bm25_repo,
    )
    context_expander = ContextExpander(vector_store=vector_store)

    logger.info("retrieval_services_initialized")

    # Create orchestrator in retrieval-only mode (no LLM services)
    return QueryOrchestrator(
        embedding_generator=embedding_gen,
        hybrid_retrieval=hybrid_service,
        context_expander=context_expander,
        llm_router=None,
        response_formatter=None,
    )


def _display_results(query_text: str, result: RetrievalResult) -> None:
    """Display retrieval results in user-friendly format.

    Args:
        query_text: The original query text
        result: RetrievalResult from QueryOrchestrator.retrieve_only()
    """
    metadata = result.metadata
    chunks = result.chunks

    click.echo("=" * 80)
    click.echo("HYBRID RETRIEVAL RESULTS")
    click.echo("=" * 80)
    click.echo(f"Query: {query_text}")
    click.echo(f"Results: {len(chunks)} chunks")

    # Show expansion statistics
    if metadata.initial_count < metadata.expanded_count:
        added_count = metadata.expanded_count - metadata.initial_count
        click.echo(f"  ├─ Initial retrieval: {metadata.initial_count} chunks")
        click.echo(f"  └─ Context expansion: +{added_count} chunks")

    click.echo(f"Latency: {metadata.latency_ms}ms")
    click.echo(f"  ├─ Embedding: {metadata.embedding_ms}ms")
    click.echo(f"  ├─ Retrieval: {metadata.retrieval_ms}ms")
    click.echo(f"  └─ Expansion: {metadata.expansion_ms}ms")
    click.echo("=" * 80)
    click.echo()

    if not chunks:
        click.echo("No results found.")
        return

    for i, (chunk, score) in enumerate(chunks, start=1):
        click.echo(f"[{i}] Score: {score:.6f}")
        click.echo(f"    Article: {chunk['article_title']}")
        click.echo(f"    Section: {chunk['section_path']}")
        click.echo(f"    Chunk ID: {chunk['id']}")

        # Display first N characters of chunk text
        chunk_text = chunk["chunk_text"]
        preview = (
            chunk_text[:PREVIEW_LENGTH] + "..." if len(chunk_text) > PREVIEW_LENGTH else chunk_text
        )
        click.echo(f"    Preview: {preview}")
        click.echo()


async def _execute_retrieval(query_text: str, top_k: int) -> None:
    """Execute hybrid retrieval using QueryOrchestrator.

    Args:
        query_text: User query text
        top_k: Number of results to retrieve

    Raises:
        ConfigurationError: If required configuration is missing
        Exception: If retrieval fails
    """
    try:
        # Create retrieval-only orchestrator
        orchestrator = _create_retrieval_orchestrator()

        click.echo(f"Executing retrieval for: {query_text}")

        # Execute retrieval
        result = await orchestrator.retrieve_only(query_text, top_k=top_k)

        # Display results
        _display_results(query_text, result)

        logger.info(
            "retrieval_completed",
            results_count=len(result.chunks),
            latency_ms=result.metadata.latency_ms,
        )

    except ConfigurationError as e:
        click.echo(f"❌ Configuration Error: {e}", err=True)
        raise
    except Exception as e:
        logger.error("retrieval_failed", error=str(e), exc_info=True)
        click.echo(f"❌ Retrieval failed: {e}", err=True)
        raise


@click.command()
@click.argument("query_text", type=str)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=lambda: int(os.getenv("RETRIEVAL_TOP_K", "20")),
    help="Number of results to retrieve (default: from RETRIEVAL_TOP_K env var)",
)
def retrieve(query_text: str, top_k: int) -> None:
    """Execute hybrid retrieval for a query.

    Combines vector similarity search and BM25 keyword matching to retrieve
    the most relevant chunks from the knowledge base.

    Args:
        query_text: The query text to search for
        top_k: Number of results to retrieve (default: 5)

    Examples:
        poetry run retrieve "Who is Roboute Guilliman?"

        poetry run retrieve "Ultramarines homeworld" --top-k 10
    """
    try:
        asyncio.run(_execute_retrieval(query_text, top_k))
    except KeyboardInterrupt:
        click.echo("\n⚠️  Retrieval cancelled by user")
    except Exception:
        # Error already logged and displayed
        sys.exit(1)
