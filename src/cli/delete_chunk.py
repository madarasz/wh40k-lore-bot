"""CLI command for deleting a chunk from both stores."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _get_database_url() -> str:
    """Get database URL from environment."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        click.echo("Error: DATABASE_URL environment variable is not set", err=True)
        raise click.Abort()
    return db_url


def _check_chroma(vector_store: ChromaVectorStore, chunk_id: str) -> tuple[bool, Mapping[str, Any]]:
    """Check if chunk exists in Chroma and return metadata."""
    results = vector_store.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    exists = bool(results["ids"])
    metadata = results["metadatas"][0] if exists and results["metadatas"] else {}
    return exists, metadata


def _check_sqlite(engine: Engine, chunk_id: str) -> bool:
    """Check if chunk exists in SQLite."""
    with engine.connect() as conn:
        stmt = select(WikiChunk).where(WikiChunk.id == chunk_id)
        result = conn.execute(stmt)
        return result.fetchone() is not None


def _perform_deletion(
    vector_store: ChromaVectorStore,
    engine: Engine,
    chunk_id: str,
    chroma_exists: bool,
    sqlite_exists: bool,
) -> tuple[bool, bool]:
    """Delete chunk from stores and return deletion status."""
    chroma_deleted = False
    if chroma_exists:
        vector_store.collection.delete(ids=[chunk_id])
        chroma_deleted = True

    sqlite_deleted = False
    if sqlite_exists:
        with engine.begin() as conn:
            stmt = delete(WikiChunk).where(WikiChunk.id == chunk_id)
            conn.execute(stmt)
        sqlite_deleted = True

    return chroma_deleted, sqlite_deleted


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
    """Delete a chunk from both Chroma and SQLite stores.

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
            click.echo("Chunk found in Chroma:")
            click.echo(f"  Chunk ID: {chunk_id}")
            click.echo(f"  Article Title: {metadata.get('article_title', 'N/A')}")
        else:
            click.echo(f"  Chunk '{chunk_id}' not found in Chroma")

        engine = create_engine(_get_database_url())
        sqlite_exists = _check_sqlite(engine, chunk_id)
        click.echo(f"  Chunk {'found' if sqlite_exists else 'not found'} in SQLite")

        if not chroma_exists and not sqlite_exists:
            click.echo(f"\nError: Chunk '{chunk_id}' not found in any store", err=True)
            raise click.Abort()

        if not force and not click.confirm("\nAre you sure you want to delete this chunk?"):
            click.echo("Deletion cancelled")
            return

        click.echo("\nDeleting chunk...")
        chroma_deleted, sqlite_deleted = _perform_deletion(
            vector_store, engine, chunk_id, chroma_exists, sqlite_exists
        )

        click.echo(f"  Deleted from Chroma: {'Yes' if chroma_deleted else 'No (not found)'}")
        click.echo(f"  Deleted from SQLite: {'Yes' if sqlite_deleted else 'No (not found)'}")
        click.echo("\n" + "=" * 80)
        click.echo("Deletion Complete!")
        click.echo("=" * 80)

        logger.info(
            "chunk_deleted",
            chunk_id=chunk_id,
            chroma_deleted=chroma_deleted,
            sqlite_deleted=sqlite_deleted,
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
