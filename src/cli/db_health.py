"""CLI command for database health check."""

import os
from pathlib import Path

import click
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, func, select, text

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _check_sqlite() -> tuple[bool, int]:
    """Check SQLite health and return (ok, count)."""
    click.echo("-" * 80)
    click.echo("SQLite Database")
    click.echo("-" * 80)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        click.echo("  Status: ERROR")
        click.echo("  Error: DATABASE_URL environment variable is not set")
        return False, 0

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            stmt = select(func.count()).select_from(WikiChunk)
            count = conn.execute(stmt).scalar() or 0

        click.echo("  Status: OK")
        click.echo(f"  Database URL: {db_url}")
        click.echo(f"  WikiChunk Row Count: {count:,}")
        return True, count

    except Exception as e:
        click.echo("  Status: ERROR")
        click.echo(f"  Error: {e}")
        logger.error("sqlite_health_check_failed", error=str(e))
        return False, 0


def _check_chroma(chroma_path: Path) -> tuple[bool, int, str]:
    """Check Chroma health and return (ok, count, collection_name)."""
    click.echo("-" * 80)
    click.echo("Chroma Vector Database")
    click.echo("-" * 80)

    try:
        vector_store = ChromaVectorStore(storage_path=str(chroma_path))
        count = vector_store.count()
        collection_name = vector_store.collection_name

        click.echo("  Status: OK")
        click.echo(f"  Storage Path: {chroma_path}")
        click.echo(f"  Collection Name: {collection_name}")
        click.echo(f"  Chunk Count: {count:,}")
        return True, count, collection_name

    except Exception as e:
        click.echo("  Status: ERROR")
        click.echo(f"  Error: {e}")
        logger.error("chroma_health_check_failed", error=str(e))
        return False, 0, "N/A"


def _display_consistency(
    sqlite_ok: bool, chroma_ok: bool, sqlite_count: int, chroma_count: int
) -> None:
    """Display consistency check results."""
    click.echo("-" * 80)
    click.echo("Consistency Check")
    click.echo("-" * 80)

    if not (sqlite_ok and chroma_ok):
        click.echo("  Status: UNABLE TO CHECK")
        if not sqlite_ok:
            click.echo("  Reason: SQLite connection failed")
        if not chroma_ok:
            click.echo("  Reason: Chroma connection failed")
        return

    if sqlite_count == chroma_count:
        click.echo("  Status: CONSISTENT")
        click.echo(f"  Both stores have {chroma_count:,} chunks")
    else:
        click.echo("  Status: INCONSISTENT")
        click.echo(f"  SQLite Count: {sqlite_count:,}")
        click.echo(f"  Chroma Count: {chroma_count:,}")
        click.echo(f"  Difference: {abs(sqlite_count - chroma_count):,}")


@click.command()
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def db_health(chroma_path: Path) -> None:
    """Check health of both SQLite and Chroma databases."""
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Database Health Check")
    click.echo("=" * 80)
    click.echo()

    sqlite_ok, sqlite_count = _check_sqlite()
    click.echo()

    chroma_ok, chroma_count, _ = _check_chroma(chroma_path)
    click.echo()

    _display_consistency(sqlite_ok, chroma_ok, sqlite_count, chroma_count)
    click.echo()

    click.echo("=" * 80)
    click.echo("Health Check Summary")
    click.echo("=" * 80)
    click.echo(f"  SQLite: {'OK' if sqlite_ok else 'FAILED'}")
    click.echo(f"  Chroma: {'OK' if chroma_ok else 'FAILED'}")

    if sqlite_ok and chroma_ok:
        consistency = "CONSISTENT" if sqlite_count == chroma_count else "INCONSISTENT"
        click.echo(f"  Consistency: {consistency}")

    click.echo()

    if not sqlite_ok or not chroma_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    db_health()
