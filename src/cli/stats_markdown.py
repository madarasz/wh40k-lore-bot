"""CLI command for markdown archive statistics."""

import statistics
from pathlib import Path

import click
import structlog
from dotenv import load_dotenv

from src.ingestion.markdown_loader import MarkdownLoader

load_dotenv()
logger = structlog.get_logger(__name__)


def _collect_file_stats(loader: MarkdownLoader) -> list[tuple[str, int]]:
    """Collect file statistics from markdown archive."""
    file_stats: list[tuple[str, int]] = []
    for article in loader.load_all():
        file_stats.append((article.title, article.word_count))
    return file_stats


def _display_summary(file_stats: list[tuple[str, int]]) -> None:
    """Display summary statistics."""
    word_counts = [wc for _, wc in file_stats]
    click.echo("-" * 80)
    click.echo("Summary")
    click.echo("-" * 80)
    click.echo(f"  Total Files: {len(file_stats):,}")
    click.echo(f"  Total Words: {sum(word_counts):,}")
    click.echo(f"  Median Word Count: {statistics.median(word_counts):,.0f}")
    click.echo()


def _display_top_lists(sorted_stats: list[tuple[str, int]]) -> None:
    """Display top 10 smallest and largest files."""
    click.echo("-" * 80)
    click.echo("Top 10 Smallest Files (by word count)")
    click.echo("-" * 80)
    for i, (filename, word_count) in enumerate(sorted_stats[:10], 1):
        click.echo(f"  {i:2}. {filename}: {word_count:,} words")
    click.echo()

    click.echo("-" * 80)
    click.echo("Top 10 Largest Files (by word count)")
    click.echo("-" * 80)
    for i, (filename, word_count) in enumerate(sorted_stats[-10:][::-1], 1):
        click.echo(f"  {i:2}. {filename}: {word_count:,} words")
    click.echo()


@click.command()
@click.option(
    "--archive-path",
    type=click.Path(exists=True, path_type=Path),
    default="data/markdown-archive",
    help="Path to markdown archive directory (default: data/markdown-archive)",
)
def stats_markdown(archive_path: Path) -> None:
    """Display markdown archive statistics.

    Shows total file count, top 10 smallest and largest files by word count,
    and median word count across all files.
    """
    click.echo("=" * 80)
    click.echo("WH40K Lore Bot - Markdown Archive Statistics")
    click.echo("=" * 80)
    click.echo()
    click.echo(f"Archive Path: {archive_path}")
    click.echo()

    try:
        loader = MarkdownLoader(archive_path)
        click.echo("Scanning archive...")
        file_stats = _collect_file_stats(loader)

        if not file_stats:
            click.echo("  No markdown files found in archive", err=True)
            raise click.Abort()

        click.echo()
        _display_summary(file_stats)
        _display_top_lists(sorted(file_stats, key=lambda x: x[1]))

    except (FileNotFoundError, click.Abort) as e:
        if isinstance(e, FileNotFoundError):
            click.echo(f"Error: {e}", err=True)
            logger.error("stats_markdown_failed", error=str(e))
        raise click.Abort() from e if isinstance(e, FileNotFoundError) else e
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        raise click.Abort() from None
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error("stats_markdown_failed", error=str(e), exc_info=True)
        raise click.Abort() from e


if __name__ == "__main__":
    stats_markdown()
