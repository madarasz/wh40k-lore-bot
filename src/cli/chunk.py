"""CLI command for chunking markdown articles."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import structlog
from tqdm import tqdm

from src.cli.utils import load_wiki_ids
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.text_chunker import MarkdownChunker

logger = structlog.get_logger(__name__)


@click.command()
@click.option(
    "--archive-path",
    type=click.Path(exists=True, path_type=Path),
    default="data/markdown-archive",
    help="Path to markdown archive directory (default: data/markdown-archive)",
)
@click.option(
    "--wiki-ids-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to file containing wiki IDs to process (one per line)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/chunks.json",
    help="Output file path for chunks JSON (default: data/chunks.json)",
)
def chunk(  # noqa: PLR0915
    archive_path: Path,
    wiki_ids_file: Path | None,
    output: Path,
) -> None:
    """Chunk markdown articles from archive.

    Reads markdown files and chunks them into smaller pieces suitable for
    embedding. Outputs chunks as JSON for the embed step.

    Examples:

        \b
        # Chunk entire archive
        poetry run chunk

        \b
        # Chunk specific articles
        poetry run chunk --wiki-ids-file data/test-bed-pages.txt

        \b
        # Custom output path
        poetry run chunk --output data/my-chunks.json
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Markdown Chunking")
    click.echo("=" * 80)
    click.echo()

    # Load wiki IDs from file if provided
    wiki_ids = None
    if wiki_ids_file:
        try:
            wiki_ids = load_wiki_ids(wiki_ids_file)
            click.echo(f"  Loaded {len(wiki_ids)} wiki IDs from {wiki_ids_file}")
        except Exception as e:
            click.echo(f"  Failed to load wiki IDs: {e}", err=True)
            raise click.Abort() from e

    click.echo(f"  Archive Path: {archive_path}")
    click.echo(f"  Output File: {output}")
    click.echo()

    # Initialize components
    loader = MarkdownLoader(archive_path)
    chunker = MarkdownChunker()
    metadata_extractor = MetadataExtractor()

    # Process articles
    chunks_data: list[dict[str, Any]] = []
    total_articles = 0
    total_chunks = 0

    click.echo("Processing articles...")

    try:
        articles = loader.load_all(wiki_ids=wiki_ids)

        for article in tqdm(articles, desc="Chunking", unit="article"):
            try:
                chunks = chunker.chunk_markdown(article.content, article.title)

                for chunk_obj in chunks:
                    # Extract metadata
                    try:
                        metadata = metadata_extractor.extract_metadata(chunk_obj)
                    except Exception as e:
                        logger.warning(
                            "metadata_extraction_failed",
                            article_title=article.title,
                            chunk_index=chunk_obj.chunk_index,
                            error=str(e),
                        )
                        metadata = {}

                    chunk_entry = {
                        "wiki_page_id": article.wiki_id,
                        "article_title": article.title,
                        "last_updated": article.last_updated,
                        "section_path": chunk_obj.section_path,
                        "chunk_index": chunk_obj.chunk_index,
                        "chunk_text": chunk_obj.chunk_text,
                        "metadata": metadata,
                    }
                    chunks_data.append(chunk_entry)
                    total_chunks += 1

                total_articles += 1

            except Exception as e:
                logger.error(
                    "chunking_failed",
                    article_title=article.title,
                    error=str(e),
                )
                continue

    except KeyboardInterrupt:
        click.echo()
        click.echo("  Interrupted by user", err=True)
        raise click.Abort() from None

    # Create output structure
    output_data = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_archive": str(archive_path),
        "total_articles": total_articles,
        "total_chunks": total_chunks,
        "chunks": chunks_data,
    }

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    click.echo()
    click.echo("=" * 80)
    click.echo("Chunking Complete!")
    click.echo("=" * 80)
    click.echo(f"  Articles Processed: {total_articles}")
    click.echo(f"  Chunks Created: {total_chunks}")
    click.echo(f"  Output File: {output}")
    click.echo()


if __name__ == "__main__":
    chunk()
