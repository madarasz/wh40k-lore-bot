"""CLI command for building BM25 keyword search index."""

import json
import os
from pathlib import Path

import click
import structlog
from tqdm import tqdm

from src.ingestion.models import Chunk
from src.repositories.bm25_repository import BM25Repository

logger = structlog.get_logger(__name__)


def _load_chunks_from_json(chunks_file: Path) -> list[Chunk]:
    """Load chunks from JSON file and convert to Chunk objects.

    Args:
        chunks_file: Path to chunks JSON file (output of 'chunk' command)

    Returns:
        List of Chunk objects

    Raises:
        ValueError: If JSON file is malformed or missing required fields
    """
    with chunks_file.open("r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks_list = chunks_data.get("chunks", [])
    if not chunks_list:
        raise ValueError("No chunks found in JSON file")

    chunk_objects: list[Chunk] = []
    for entry in chunks_list:
        try:
            chunk = Chunk(
                chunk_text=entry["chunk_text"],
                article_title=entry["article_title"],
                section_path=entry["section_path"],
                chunk_index=entry["chunk_index"],
                links=entry.get("metadata", {}).get("links", []),
                wiki_page_id=entry.get("wiki_page_id"),
            )
            chunk_objects.append(chunk)
        except KeyError as e:
            raise ValueError(f"Missing required field in chunk entry: {e}") from e

    return chunk_objects


def _build_and_save_index(chunks: list[Chunk], output: Path) -> dict[str, int]:
    """Build BM25 index and save to disk.

    Args:
        chunks: List of chunks to index
        output: Output path for index file

    Returns:
        Index statistics dictionary

    Raises:
        Exception: If index building or saving fails
    """
    bm25_repo = BM25Repository()

    # Build index with progress bar
    with tqdm(total=len(chunks), desc="Indexing", unit="chunk") as pbar:
        bm25_repo.build_index(chunks)
        pbar.update(len(chunks))

    # Get index stats
    stats = bm25_repo.get_index_stats()

    # Save index
    bm25_repo.save_index(output)

    return stats


@click.command()
@click.argument("chunks_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for BM25 index (default: from BM25_INDEX_PATH env var)",
)
def build_bm25(
    chunks_file: Path,
    output: Path | None,
) -> None:
    """Build BM25 keyword search index from chunks JSON.

    Reads chunks from JSON file (output of 'chunk' command) and builds
    a BM25 index for keyword-based retrieval. Useful for debugging and
    standalone index building.

    CHUNKS_FILE: Path to the chunks JSON file (output of 'chunk' command)

    Examples:

        \b
        # Build BM25 index from chunks
        poetry run build-bm25 data/chunks.json

        \b
        # Custom output path
        poetry run build-bm25 data/chunks.json --output data/my-bm25-index.pkl
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - BM25 Index Builder")
    click.echo("=" * 80)
    click.echo()

    # Determine output path
    if output is None:
        output_str = os.getenv("BM25_INDEX_PATH", "data/bm25-index/bm25_index.pkl")
        output = Path(output_str)

    click.echo(f"  Chunks File: {chunks_file}")
    click.echo(f"  Output Path: {output}")
    click.echo()

    # Load chunks from JSON
    click.echo("Loading chunks from JSON...")
    try:
        chunks = _load_chunks_from_json(chunks_file)
        click.echo(f"  Loaded {len(chunks)} chunks")
    except Exception as e:
        click.echo(f"  Failed to load chunks: {e}", err=True)
        logger.error("chunks_load_failed", chunks_file=str(chunks_file), error=str(e))
        raise click.Abort() from e

    click.echo()

    # Build and save index
    click.echo("Building BM25 index...")
    try:
        stats = _build_and_save_index(chunks, output)
        click.echo("  Index built successfully")
        click.echo(f"  Total chunks: {stats['total_chunks']}")
        click.echo(f"  Unique tokens: {stats['unique_tokens']}")
        click.echo()
        click.echo("  Index saved to disk")
    except Exception as e:
        click.echo(f"  Index building/saving failed: {e}", err=True)
        logger.error("bm25_build_save_failed", error=str(e), exc_info=True)
        raise click.Abort() from e

    click.echo()
    click.echo("=" * 80)
    click.echo("BM25 Index Building Complete!")
    click.echo("=" * 80)
    click.echo(f"  Total Chunks: {stats['total_chunks']}")
    click.echo(f"  Unique Tokens: {stats['unique_tokens']}")
    click.echo(f"  Index File: {output}")
    click.echo(f"  File Size: {output.stat().st_size / 1024:.1f} KB")
    click.echo()


if __name__ == "__main__":
    build_bm25()
