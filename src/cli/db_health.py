"""CLI command for ChromaDB health check."""

from pathlib import Path

import click
import structlog
from dotenv import load_dotenv

from src.rag.vector_store import ChromaVectorStore

load_dotenv()
logger = structlog.get_logger(__name__)


def _check_chroma(chroma_path: Path) -> tuple[bool, int, str]:
    """Check Chroma health and return (ok, count, collection_name)."""
    click.echo("-" * 80)
    click.echo("ChromaDB Vector Database")
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


@click.command()
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def db_health(chroma_path: Path) -> None:
    """Check health of ChromaDB vector database."""
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - ChromaDB Health Check")
    click.echo("=" * 80)
    click.echo()

    chroma_ok, chroma_count, _ = _check_chroma(chroma_path)
    click.echo()

    click.echo("=" * 80)
    click.echo("Health Check Summary")
    click.echo("=" * 80)
    click.echo(f"  ChromaDB: {'OK' if chroma_ok else 'FAILED'}")
    click.echo(f"  Total Chunks: {chroma_count:,}" if chroma_ok else "  Total Chunks: N/A")
    click.echo()

    if not chroma_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    db_health()
