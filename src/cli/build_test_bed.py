"""CLI command for building test bed page selection."""

from pathlib import Path

import click
import structlog

from src.ingestion.test_bed_builder import TestBedBuilder

logger = structlog.get_logger(__name__)


@click.command()
@click.argument("xml_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--seed-id",
    default="58",
    help="Seed page ID to start BFS traversal (default: 58 - Blood Angels)",
)
@click.option(
    "--count",
    default=100,
    type=int,
    help="Target number of pages to select (default: 100)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/test-bed-pages.txt",
    help="Output file path (default: data/test-bed-pages.txt)",
)
def build_test_bed(
    xml_path: Path,
    seed_id: str,
    count: int,
    output: Path,
) -> None:
    """Build a test bed of wiki pages using BFS traversal.

    Starts from a seed page and expands to related pages using breadth-first
    traversal with priority given to hub pages (most incoming links).

    Args:
        xml_path: Path to the MediaWiki XML export file
        seed_id: Starting page ID for traversal
        count: Target number of pages to select
        output: Output file path for selected page IDs
    """
    click.echo(f"Building test bed from: {xml_path}")
    click.echo(f"Seed page ID: {seed_id}")
    click.echo(f"Target count: {count}")
    click.echo(f"Output file: {output}")
    click.echo()

    # Create builder
    builder = TestBedBuilder()

    try:
        # Build test bed
        click.echo("Starting BFS traversal...")
        page_ids = builder.build_test_bed(xml_path, seed_id, count)

        click.echo(f"\nSelected {len(page_ids)} pages")
        click.echo()

        # Write output file
        click.echo(f"Writing output to: {output}")
        builder.write_test_bed_file(page_ids, output, xml_path, seed_id)

        # Success message
        click.echo("\nTest bed build complete!")
        click.echo(f"Pages selected: {len(page_ids)}")
        click.echo(f"Output file: {output}")

    except ValueError as e:
        logger.error("test_bed_build_failed", error=str(e))
        click.echo(f"Error: {e}", err=True)
        raise click.Abort from e
    except Exception as e:
        logger.error("test_bed_build_failed", error=str(e), exc_info=True)
        click.echo(f"Unexpected error: {e}", err=True)
        raise click.Abort from e


if __name__ == "__main__":
    build_test_bed()
