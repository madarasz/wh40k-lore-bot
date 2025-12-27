"""CLI command for parsing wiki XML exports."""

from pathlib import Path

import click
import structlog

from src.ingestion.markdown_archive import save_markdown_file
from src.ingestion.wiki_xml_parser import WikiXMLParser

logger = structlog.get_logger(__name__)


@click.command()
@click.argument("xml_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--page-ids-file",
    type=click.Path(exists=True, path_type=Path),
    help="Optional file containing page IDs to filter (one per line)",
)
def parse_wiki(xml_path: Path, page_ids_file: Path | None) -> None:
    """Parse MediaWiki XML export and convert articles to markdown.

    Args:
        xml_path: Path to the XML export file
        page_ids_file: Optional file with page IDs to filter (one per line)
    """
    click.echo(f"Parsing wiki XML from: {xml_path}")

    # Load page IDs filter if provided
    page_ids = None
    if page_ids_file:
        click.echo(f"Loading page ID filter from: {page_ids_file}")
        page_ids = page_ids_file.read_text().strip().split("\n")
        page_ids = [pid.strip() for pid in page_ids if pid.strip()]
        click.echo(f"Filtering to {len(page_ids)} page IDs")

    # Create parser
    parser = WikiXMLParser()

    # Parse and save articles
    articles_saved = 0
    try:
        for article in parser.parse_xml_export(xml_path, page_ids):
            # Save article to markdown archive
            save_markdown_file(article)
            articles_saved += 1

            # Show progress every 100 articles
            if articles_saved % 100 == 0:
                click.echo(f"Saved {articles_saved} articles...")

    except KeyboardInterrupt:
        click.echo("\nParsing interrupted by user")
    except Exception as e:
        logger.error("parsing_failed", error=str(e), exc_info=True)
        click.echo(f"Error: {e}", err=True)
        raise click.Abort from e

    # Final summary
    click.echo("\nParsing complete!")
    click.echo(f"Total articles saved: {articles_saved}")
    click.echo(f"Articles skipped: {parser.articles_skipped}")


if __name__ == "__main__":
    parse_wiki()
