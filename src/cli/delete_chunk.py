"""CLI command for deleting a chunk from ChromaDB."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import structlog
from dotenv import load_dotenv

from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _check_chroma(vector_store: ChromaVectorStore, chunk_id: str) -> tuple[bool, Mapping[str, Any]]:
    """Check if chunk exists in Chroma and return metadata."""
    results = vector_store.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    exists = bool(results["ids"])
    metadata = results["metadatas"][0] if exists and results["metadatas"] else {}
    return exists, metadata


def _perform_deletion(vector_store: ChromaVectorStore, chunk_id: str, chroma_exists: bool) -> bool:
    """Delete chunk from ChromaDB and return deletion status."""
    chroma_deleted = False
    if chroma_exists:
        vector_store.collection.delete(ids=[chunk_id])
        chroma_deleted = True
    return chroma_deleted


@click.command()
@click.argument("chunk_id")
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
@click.option("--force", is_flag=True, default=False, help="Skip confirmation prompt")
def delete_chunk(chunk_id: str, chroma_path: Path, force: bool) -> None:
    """Delete a chunk from ChromaDB.

    CHUNK_ID: The chunk ID to delete (format: {wiki_page_id}_{chunk_index})
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Delete Chunk")
    click.echo("=" * 80)
    click.echo()

    try:
        vector_store = ChromaVectorStore(storage_path=str(chroma_path))
        chroma_exists, metadata = _check_chroma(vector_store, chunk_id)

        if chroma_exists:
            click.echo("Chunk found in ChromaDB:")
            click.echo(f"  Chunk ID: {chunk_id}")
            click.echo(f"  Article Title: {metadata.get('article_title', 'N/A')}")
        else:
            click.echo(f"  Chunk '{chunk_id}' not found in ChromaDB")
            click.echo(f"\nError: Chunk '{chunk_id}' not found", err=True)
            raise click.Abort()

        if not force and not click.confirm("\nAre you sure you want to delete this chunk?"):
            click.echo("Deletion cancelled")
            return

        click.echo("\nDeleting chunk...")
        chroma_deleted = _perform_deletion(vector_store, chunk_id, chroma_exists)

        click.echo(f"  Deleted from ChromaDB: {'Yes' if chroma_deleted else 'No (not found)'}")
        click.echo("\n" + "=" * 80)
        click.echo("Deletion Complete!")
        click.echo("=" * 80)

        logger.info(
            "chunk_deleted",
            chunk_id=chunk_id,
            chroma_deleted=chroma_deleted,
        )

    except click.Abort:
        raise
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error("delete_chunk_failed", chunk_id=chunk_id, error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    delete_chunk()
