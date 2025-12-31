"""CLI command for displaying chunk details."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import structlog
import tiktoken
from dotenv import load_dotenv

from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _display_basic_info(chunk_id: str, metadata: Mapping[str, Any], token_count: int) -> None:
    """Display basic chunk information."""
    try:
        wiki_page_id, chunk_index = chunk_id.rsplit("_", 1)
    except ValueError:
        wiki_page_id, chunk_index = chunk_id, "N/A"

    click.echo("-" * 80)
    click.echo("Basic Information")
    click.echo("-" * 80)
    click.echo(f"  Chunk ID: {chunk_id}")
    click.echo(f"  Wiki Page ID: {wiki_page_id}")
    click.echo(f"  Chunk Index: {chunk_index}")
    click.echo(f"  Article Title: {metadata.get('article_title', 'N/A')}")
    click.echo(f"  Section Path: {metadata.get('section_path', 'N/A')}")
    click.echo(f"  Token Count: {token_count:,}")
    click.echo()


def _display_metadata(metadata: Mapping[str, Any]) -> None:
    """Display chunk metadata."""
    click.echo("-" * 80)
    click.echo("Metadata")
    click.echo("-" * 80)
    for key, value in sorted(metadata.items()):
        click.echo(f"  {key}: {value}")
    click.echo()


def _display_content(document: str) -> None:
    """Display chunk text content."""
    click.echo("-" * 80)
    click.echo("Chunk Text Content")
    click.echo("-" * 80)
    click.echo()
    click.echo(document)
    click.echo()


@click.command()
@click.argument("chunk_id")
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def show_chunk(chunk_id: str, chroma_path: Path) -> None:
    """Display detailed information about a specific chunk.

    CHUNK_ID: The chunk ID to look up (format: {wiki_page_id}_{chunk_index})
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Chunk Details")
    click.echo("=" * 80)
    click.echo()

    try:
        vector_store = ChromaVectorStore(storage_path=str(chroma_path))
        results = vector_store.collection.get(ids=[chunk_id], include=["documents", "metadatas"])

        if not results["ids"]:
            click.echo(f"Error: Chunk '{chunk_id}' not found", err=True)
            raise click.Abort()

        document = results["documents"][0] if results["documents"] else ""
        metadata = results["metadatas"][0] if results["metadatas"] else {}

        encoder = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoder.encode(document)) if document else 0

        _display_basic_info(chunk_id, metadata, token_count)
        _display_metadata(metadata)
        _display_content(document)

    except click.Abort:
        raise
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error("show_chunk_failed", chunk_id=chunk_id, error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    show_chunk()
