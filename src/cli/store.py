"""CLI command for storing embeddings in vector database."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import structlog
from tqdm import tqdm

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore

logger = structlog.get_logger(__name__)


def _group_by_wiki_page_id(
    embeddings_list: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group embeddings by wiki_page_id for change detection."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in embeddings_list:
        grouped[entry["wiki_page_id"]].append(entry)
    return grouped


def _get_article_last_updated(entries: list[dict[str, Any]]) -> str | None:
    """Extract article_last_updated from entries metadata."""
    if not entries:
        return None
    last_updated: str | None = entries[0].get("metadata", {}).get("article_last_updated")
    return last_updated


@click.command()
@click.argument("embeddings_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--chroma-path",
    type=str,
    default="data/chroma-db/",
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    help="Number of chunks to store per batch (default: 1000)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-ingestion even if article hasn't changed",
)
def store(  # noqa: PLR0912, PLR0915
    embeddings_file: Path,
    chroma_path: str,
    batch_size: int,
    force: bool,
) -> None:
    """Store embeddings in Chroma vector database.

    Reads embeddings JSON file (from the embed command) and stores them
    in the Chroma vector database along with chunk metadata.

    Supports incremental updates: articles already in the database with the
    same last_updated timestamp are skipped. Use --force to re-ingest all.

    EMBEDDINGS_FILE: Path to the embeddings JSON file (output of 'embed' command)

    Examples:

        \b
        # Store embeddings (skips unchanged articles)
        poetry run store data/embeddings.json

        \b
        # Force re-ingestion of all articles
        poetry run store data/embeddings.json --force

        \b
        # Custom Chroma path
        poetry run store data/embeddings.json --chroma-path /path/to/chroma-db
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Vector Store")
    click.echo("=" * 80)
    click.echo()

    click.echo(f"  Embeddings File: {embeddings_file}")
    click.echo(f"  Chroma Path: {chroma_path}")
    click.echo(f"  Batch Size: {batch_size}")
    click.echo(f"  Force Mode: {force}")
    click.echo()

    # Load embeddings
    click.echo("Loading embeddings...")
    try:
        with embeddings_file.open("r", encoding="utf-8") as f:
            embeddings_data = json.load(f)
    except Exception as e:
        click.echo(f"  Failed to load embeddings: {e}", err=True)
        raise click.Abort() from e

    embeddings_list = embeddings_data.get("embeddings", [])
    if not embeddings_list:
        click.echo("  No embeddings found in file", err=True)
        raise click.Abort()

    click.echo(f"  Loaded {len(embeddings_list)} embeddings")
    click.echo()

    # Initialize vector store
    try:
        vector_store = ChromaVectorStore(storage_path=chroma_path)
    except Exception as e:
        click.echo(f"  Failed to initialize vector store: {e}", err=True)
        raise click.Abort() from e

    # Get initial count
    initial_count = vector_store.count()
    click.echo(f"  Existing chunks in store: {initial_count}")
    click.echo()

    # Group embeddings by wiki_page_id for change detection
    grouped_embeddings = _group_by_wiki_page_id(embeddings_list)
    total_articles = len(grouped_embeddings)
    click.echo(f"  Total articles to process: {total_articles}")

    # Change detection phase
    click.echo()
    click.echo("Checking for changes...")
    articles_to_add: list[str] = []
    articles_to_update: list[str] = []
    articles_skipped: list[str] = []

    for wiki_page_id, entries in tqdm(grouped_embeddings.items(), desc="Checking", unit="article"):
        if force:
            # Force mode: check if article exists to decide add vs update
            stored_timestamp = vector_store.get_article_last_updated(wiki_page_id)
            if stored_timestamp is None:
                articles_to_add.append(wiki_page_id)
            else:
                articles_to_update.append(wiki_page_id)
        else:
            # Normal mode: compare timestamps
            stored_timestamp = vector_store.get_article_last_updated(wiki_page_id)
            new_timestamp = _get_article_last_updated(entries)

            if stored_timestamp is None:
                # New article
                articles_to_add.append(wiki_page_id)
            elif new_timestamp is None:
                # No timestamp in new data, skip to be safe
                articles_skipped.append(wiki_page_id)
            elif new_timestamp > stored_timestamp:
                # Article has been updated
                articles_to_update.append(wiki_page_id)
            else:
                # Article unchanged
                articles_skipped.append(wiki_page_id)

    click.echo()
    click.echo(f"  New articles: {len(articles_to_add)}")
    click.echo(f"  Updated articles: {len(articles_to_update)}")
    click.echo(f"  Skipped (unchanged): {len(articles_skipped)}")
    click.echo()

    # Nothing to do?
    if not articles_to_add and not articles_to_update:
        click.echo("=" * 80)
        click.echo("No changes detected - nothing to store!")
        click.echo("=" * 80)
        return

    # Delete old chunks for articles that will be updated
    if articles_to_update:
        click.echo("Removing old chunks for updated articles...")
        total_deleted = 0
        for wiki_page_id in tqdm(articles_to_update, desc="Deleting", unit="article"):
            deleted = vector_store.delete_by_wiki_page_id(wiki_page_id)
            total_deleted += deleted
        click.echo(f"  Deleted {total_deleted} old chunks")
        click.echo()

    # Collect all chunks to store
    articles_to_store = articles_to_add + articles_to_update
    chunks_to_store: list[dict[str, Any]] = []
    for wiki_page_id in articles_to_store:
        chunks_to_store.extend(grouped_embeddings[wiki_page_id])

    # Process in batches
    total_stored = 0

    click.echo(f"Storing {len(chunks_to_store)} chunks...")

    try:
        for i in tqdm(range(0, len(chunks_to_store), batch_size), desc="Batches", unit="batch"):
            batch = chunks_to_store[i : i + batch_size]

            # Convert to WikiChunk objects and numpy arrays
            wiki_chunks = []
            embeddings = []

            for entry in batch:
                wiki_chunk = WikiChunk(
                    id=entry["chunk_id"],
                    wiki_page_id=entry["wiki_page_id"],
                    article_title=entry["article_title"],
                    section_path=entry["section_path"],
                    chunk_text=entry["chunk_text"],
                    chunk_index=entry["chunk_index"],
                    metadata_json=entry.get("metadata", {}),
                )
                wiki_chunks.append(wiki_chunk)
                embeddings.append(np.array(entry["embedding"]))

            # Store batch
            vector_store.add_chunks(wiki_chunks, embeddings)
            total_stored += len(wiki_chunks)

    except KeyboardInterrupt:
        click.echo()
        click.echo("  Interrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo()
        click.echo(f"  Storage failed: {e}", err=True)
        logger.error("storage_failed", error=str(e), exc_info=True)
        raise click.Abort() from e

    # Get final count
    final_count = vector_store.count()

    click.echo()
    click.echo("=" * 80)
    click.echo("Storage Complete!")
    click.echo("=" * 80)
    click.echo(f"  Articles Added: {len(articles_to_add)}")
    click.echo(f"  Articles Updated: {len(articles_to_update)}")
    click.echo(f"  Articles Skipped: {len(articles_skipped)}")
    click.echo(f"  Chunks Stored: {total_stored}")
    click.echo(f"  Total Chunks in Store: {final_count}")
    click.echo(f"  Net Change: {final_count - initial_count:+d}")
    click.echo(f"  Chroma Path: {chroma_path}")
    click.echo()


if __name__ == "__main__":
    store()
