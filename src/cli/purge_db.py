"""CLI command for purging all data from both stores."""

import os
from pathlib import Path

import click
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import Engine

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _get_counts(chroma_path: Path) -> tuple[ChromaVectorStore | None, int, Engine | None, int]:
    """Get current counts from both stores."""
    click.echo("-" * 80)
    click.echo("Current Data Counts")
    click.echo("-" * 80)

    vector_store: ChromaVectorStore | None = None
    chroma_count = 0
    try:
        vector_store = ChromaVectorStore(storage_path=str(chroma_path))
        chroma_count = vector_store.count()
        click.echo(f"  Chroma Chunks: {chroma_count:,}")
    except Exception as e:
        click.echo(f"  Chroma: ERROR ({e})")

    engine: Engine | None = None
    sqlite_count = 0
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                stmt = select(func.count()).select_from(WikiChunk)
                sqlite_count = conn.execute(stmt).scalar() or 0
            click.echo(f"  SQLite Chunks: {sqlite_count:,}")
        except Exception as e:
            click.echo(f"  SQLite: ERROR ({e})")
    else:
        click.echo("  SQLite: DATABASE_URL not set")

    return vector_store, chroma_count, engine, sqlite_count


def _confirm_deletion(force: bool) -> bool:
    """Prompt for deletion confirmation."""
    click.echo("-" * 80)
    click.echo("WARNING: This will permanently delete ALL data!")
    click.echo("-" * 80)
    click.echo()

    if force:
        return True

    confirmation: str = click.prompt(
        'Type "DELETE ALL" to confirm (or Ctrl+C to cancel)', default="", show_default=False
    )
    return confirmation == "DELETE ALL"


def _perform_purge(
    vector_store: ChromaVectorStore | None,
    chroma_count: int,
    engine: Engine | None,
    sqlite_count: int,
) -> tuple[int, int]:
    """Perform the actual purge and return counts deleted."""
    click.echo("Purging data...")

    chroma_deleted = 0
    if vector_store and chroma_count > 0:
        results = vector_store.collection.get(include=[])
        if results["ids"]:
            vector_store.collection.delete(ids=results["ids"])
            chroma_deleted = len(results["ids"])
        click.echo(f"  Deleted from Chroma: {chroma_deleted:,} chunks")
    else:
        click.echo("  Deleted from Chroma: 0 (empty or unavailable)")

    sqlite_deleted = 0
    if engine and sqlite_count > 0:
        with engine.begin() as conn:
            stmt = delete(WikiChunk)
            result = conn.execute(stmt)
            sqlite_deleted = result.rowcount
        click.echo(f"  Deleted from SQLite: {sqlite_deleted:,} rows")
    else:
        click.echo("  Deleted from SQLite: 0 (empty or unavailable)")

    return chroma_deleted, sqlite_deleted


@click.command()
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt (requires typing DELETE ALL otherwise)",
)
def purge_db(chroma_path: Path, force: bool) -> None:
    """Delete ALL chunks from both Chroma and SQLite stores.

    WARNING: This is a destructive operation that cannot be undone!
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Purge All Data")
    click.echo("=" * 80)
    click.echo()

    try:
        vector_store, chroma_count, engine, sqlite_count = _get_counts(chroma_path)
        click.echo()

        if chroma_count == 0 and sqlite_count == 0:
            click.echo("No data to delete - both stores are empty.")
            return

        if not _confirm_deletion(force):
            click.echo("\nDeletion cancelled - confirmation text did not match")
            return

        click.echo()
        chroma_deleted, sqlite_deleted = _perform_purge(
            vector_store, chroma_count, engine, sqlite_count
        )

        click.echo("\n" + "=" * 80)
        click.echo("Purge Complete!")
        click.echo("=" * 80)
        click.echo(f"  Chroma Chunks Deleted: {chroma_deleted:,}")
        click.echo(f"  SQLite Rows Deleted: {sqlite_deleted:,}")
        click.echo(f"  Total Deleted: {chroma_deleted + sqlite_deleted:,}")
        click.echo()

        logger.warning(
            "database_purged", chroma_deleted=chroma_deleted, sqlite_deleted=sqlite_deleted
        )

    except KeyboardInterrupt:
        click.echo("\nPurge cancelled by user", err=True)
        raise click.Abort() from None
    except click.Abort:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error("purge_db_failed", error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    purge_db()
