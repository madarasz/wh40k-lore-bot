"""CLI command for running the markdown-based ingestion pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING

import click
import structlog
from dotenv import load_dotenv

from src.cli.utils import load_wiki_ids
from src.ingestion.pipeline import IngestionPipeline

# Load environment variables from .env file
load_dotenv()

if TYPE_CHECKING:
    from src.ingestion.pipeline import IngestionStatistics

logger = structlog.get_logger(__name__)


def _display_configuration(
    archive_path: Path,
    batch_size: int,
    dry_run: bool,
    force: bool,
    wiki_ids: list[str] | None,
) -> None:
    """Display pipeline configuration to user."""
    click.echo(f"Archive Path: {archive_path}")
    click.echo(f"Batch Size: {batch_size}")
    click.echo(f"Dry Run: {dry_run}")
    click.echo(f"Force Re-ingest: {force}")
    click.echo(f"Wiki ID Filter: {'Yes' if wiki_ids else 'No (processing all files)'}")
    if wiki_ids:
        click.echo(f"   {len(wiki_ids)} wiki IDs loaded")
    click.echo()


def _display_summary(stats: "IngestionStatistics", dry_run: bool) -> None:
    """Display pipeline execution summary."""
    click.echo()
    click.echo("=" * 80)
    click.echo("Ingestion Complete!")
    click.echo("=" * 80)
    click.echo(f"  Articles Processed: {stats.articles_processed}")
    click.echo(f"  Articles Skipped: {stats.articles_skipped}")
    click.echo(f"  Articles Failed: {stats.articles_failed}")
    click.echo(f"  Chunks Created: {stats.chunks_created}")
    click.echo(f"  Chunks Deleted: {stats.chunks_deleted}")

    if not dry_run:
        click.echo(f"  Embeddings Generated: {stats.embeddings_generated}")
        click.echo(f"  Total Cost: ${stats.total_cost:.4f} USD")
        click.echo(f"  Tokens Used: {stats.tokens_used:,}")

    click.echo(f"  Duration: {int(stats.duration_seconds)}s")
    click.echo()
    click.echo("Summary report saved to: logs/ingestion-summary.json")
    click.echo()


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
    "--batch-size",
    type=int,
    default=100,
    help="Number of articles to process per batch (default: 100)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Process without generating embeddings or storing (useful for testing)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-ingest all articles regardless of last_updated timestamp",
)
@click.option(
    "--chroma-path",
    type=str,
    default=None,
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def ingest(  # noqa: PLR0913
    archive_path: Path,
    wiki_ids_file: Path | None,
    batch_size: int,
    dry_run: bool,
    force: bool,
    chroma_path: str | None,
) -> None:
    """Ingest markdown articles into vector database.

    Reads markdown files from the archive directory and processes them
    through the full pipeline: chunk -> extract metadata -> embed -> store.

    Unchanged articles (same last_updated timestamp) are skipped unless --force is used.

    Examples:

        \b
        # Basic usage - process entire archive
        poetry run ingest

        \b
        # Process specific articles
        poetry run ingest --wiki-ids-file data/test-bed-pages.txt

        \b
        # Force re-ingest all articles
        poetry run ingest --force

        \b
        # Dry run (no embeddings, for testing)
        poetry run ingest --dry-run --batch-size 10
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Markdown Ingestion Pipeline")
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

    # Display configuration
    _display_configuration(archive_path, batch_size, dry_run, force, wiki_ids)

    # Initialize pipeline
    try:
        pipeline = IngestionPipeline(
            archive_path=archive_path,
            chroma_path=chroma_path,
        )
    except Exception as e:
        click.echo(f"  Failed to initialize pipeline: {e}", err=True)
        raise click.Abort() from e

    # Run pipeline
    click.echo("Starting ingestion pipeline...")
    click.echo()

    try:
        stats = pipeline.run(
            wiki_ids=wiki_ids,
            batch_size=batch_size,
            dry_run=dry_run,
            force=force,
        )

        # Display summary
        _display_summary(stats, dry_run)

    except KeyboardInterrupt:
        click.echo()
        click.echo("  Pipeline interrupted by user", err=True)
        raise click.Abort() from None

    except Exception as e:
        click.echo()
        click.echo(f"  Pipeline failed: {e}", err=True)
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    ingest()
