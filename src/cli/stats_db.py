"""CLI command for vector database statistics."""

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


def _collect_chunk_stats(
    results: Mapping[str, Any], encoder: tiktoken.Encoding
) -> tuple[list[tuple[str, str, int]], str | None]:
    """Collect statistics for all chunks."""
    chunk_stats: list[tuple[str, str, int]] = []
    most_recent_updated: str | None = None

    for i, chunk_id in enumerate(results["ids"]):
        document = results["documents"][i] if results["documents"] else ""
        metadata = results["metadatas"][i] if results["metadatas"] else {}

        token_count = len(encoder.encode(document)) if document else 0
        article_title = metadata.get("article_title", "Unknown")
        chunk_stats.append((chunk_id, article_title, token_count))

        last_updated = metadata.get("article_last_updated")
        if last_updated and (most_recent_updated is None or last_updated > most_recent_updated):
            most_recent_updated = last_updated

    return chunk_stats, most_recent_updated


def _display_chunk_stats(
    sorted_stats: list[tuple[str, str, int]], total: int, most_recent: str | None
) -> None:
    """Display chunk statistics."""
    click.echo("-" * 80)
    click.echo("Summary")
    click.echo("-" * 80)
    click.echo(f"  Total Chunks: {total:,}")
    click.echo(f"  Most Recent article_last_updated: {most_recent or 'N/A'}")
    click.echo()

    click.echo("-" * 80)
    click.echo("Top 10 Smallest Chunks (by token count)")
    click.echo("-" * 80)
    for i, (chunk_id, article_title, token_count) in enumerate(sorted_stats[:10], 1):
        click.echo(f"  {i:2}. {chunk_id} ({article_title}): {token_count:,} tokens")
    click.echo()

    click.echo("-" * 80)
    click.echo("Top 10 Largest Chunks (by token count)")
    click.echo("-" * 80)
    for i, (chunk_id, article_title, token_count) in enumerate(sorted_stats[-10:][::-1], 1):
        click.echo(f"  {i:2}. {chunk_id} ({article_title}): {token_count:,} tokens")
    click.echo()


@click.command()
@click.option(
    "--chroma-path",
    type=click.Path(path_type=Path),
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def stats_db(chroma_path: Path) -> None:
    """Display vector database statistics.

    Shows total chunk count, top 10 smallest and largest chunks by token count,
    and most recent article_last_updated date.
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Vector Database Statistics")
    click.echo("=" * 80)
    click.echo()
    click.echo(f"Chroma Path: {chroma_path}")
    click.echo()

    try:
        vector_store = ChromaVectorStore(storage_path=str(chroma_path))
        total_chunks = vector_store.count()

        if total_chunks == 0:
            click.echo("  No chunks found in vector database", err=True)
            raise click.Abort()

        click.echo(f"Scanning {total_chunks:,} chunks...")
        results = vector_store.collection.get(include=["documents", "metadatas"])

        if not results["ids"]:
            click.echo("  Failed to retrieve chunks", err=True)
            raise click.Abort()

        encoder = tiktoken.get_encoding("cl100k_base")
        chunk_stats, most_recent = _collect_chunk_stats(results, encoder)
        sorted_stats = sorted(chunk_stats, key=lambda x: x[2])

        click.echo()
        _display_chunk_stats(sorted_stats, total_chunks, most_recent)

    except click.Abort:
        raise
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error("stats_db_failed", error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    stats_db()
