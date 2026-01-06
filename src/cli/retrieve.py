"""CLI command for testing hybrid retrieval."""

import asyncio
import os
import sys
import time

import click
import structlog
from dotenv import load_dotenv

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.rag.hybrid_retrieval import HybridRetrievalService
from src.rag.vector_store import ChromaVectorStore, ChunkData
from src.repositories.bm25_repository import BM25Repository
from src.utils.exceptions import ConfigurationError

load_dotenv()
logger = structlog.get_logger(__name__)

# Display constants
PREVIEW_LENGTH = 200


def _display_results(
    query_text: str, results: list[tuple[ChunkData, float]], latency_ms: float
) -> None:
    """Display retrieval results in user-friendly format.

    Args:
        query_text: The original query text
        results: List of (ChunkData, score) tuples from hybrid retrieval
        latency_ms: Retrieval latency in milliseconds
    """
    click.echo("=" * 80)
    click.echo("HYBRID RETRIEVAL RESULTS")
    click.echo("=" * 80)
    click.echo(f"Query: {query_text}")
    click.echo(f"Results: {len(results)} chunks")
    click.echo(f"Latency: {latency_ms:.2f}ms")
    click.echo("=" * 80)
    click.echo()

    if not results:
        click.echo("No results found.")
        return

    for i, (chunk, score) in enumerate(results, start=1):
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
    """Execute hybrid retrieval for query text.

    Args:
        query_text: User query text
        top_k: Number of results to retrieve

    Raises:
        ConfigurationError: If required configuration is missing
        Exception: If retrieval fails
    """
    start_time = time.time()

    try:
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

        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator()
        logger.info("embedding_generator_initialized")

        # Generate embedding for query text
        click.echo(f"Generating embedding for query: {query_text}")
        embeddings = embedding_gen.generate_embeddings([query_text])

        if not embeddings or embeddings[0] is None:
            raise RuntimeError("Failed to generate embedding for query text")

        query_embedding = embeddings[0]
        logger.info("query_embedding_generated", dimensions=len(query_embedding))

        # Initialize hybrid retrieval service
        hybrid_service = HybridRetrievalService(
            vector_store=vector_store,
            bm25_repository=bm25_repo,
            top_k=top_k,
        )

        logger.info("hybrid_retrieval_service_initialized")

        # Execute retrieval
        click.echo("Executing hybrid retrieval...")
        results = await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text=query_text,
        )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Display results
        _display_results(query_text, results, latency_ms)

        logger.info(
            "retrieval_completed",
            results_count=len(results),
            latency_ms=round(latency_ms, 2),
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
    default=5,
    help="Number of results to retrieve (default: 5)",
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
