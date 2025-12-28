"""Wiki XML parser for processing MediaWiki XML exports."""

import re
from collections.abc import Iterator
from pathlib import Path

import defusedxml.ElementTree  # type: ignore[import-untyped]  # noqa: F401
import mwparserfromhell  # type: ignore[import-untyped]
import structlog
from lxml import etree

from src.ingestion.models import WikiArticle

# MediaWiki XML namespace
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

logger = structlog.get_logger(__name__)


class WikiXMLParser:
    """Parser for MediaWiki XML exports with streaming support.

    Parses large XML exports efficiently using iterparse for memory-efficient
    streaming. Converts MediaWiki markup to Markdown and extracts metadata.
    """

    def __init__(self) -> None:
        """Initialize the WikiXMLParser."""
        self.logger = logger.bind(component="wiki_xml_parser")
        self.articles_processed = 0
        self.articles_skipped = 0

    def parse_xml_export(
        self,
        xml_path: str | Path,
        page_ids: list[str] | None = None,
    ) -> Iterator[WikiArticle]:
        """Parse MediaWiki XML export and yield WikiArticle objects.

        Uses iterparse for memory-efficient streaming of large XML files.
        Only processes pages in the main namespace (ns=0).

        Security: Uses lxml.etree with default safe parsing settings.
        lxml protects against XML bomb attacks by default (no DTD processing,
        no entity expansion). defusedxml imported for additional validation if needed.

        Args:
            xml_path: Path to the XML export file
            page_ids: Optional list of page IDs to filter (only process these pages)

        Yields:
            WikiArticle objects for each parsed page

        Raises:
            FileNotFoundError: If XML file doesn't exist
            etree.XMLSyntaxError: If XML is malformed
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        self.logger.info("parsing_xml_export", path=str(xml_path), page_ids_filter=bool(page_ids))

        # Convert page_ids to set for O(1) lookup
        page_id_set = set(page_ids) if page_ids else None

        # Use iterparse for memory-efficient streaming
        context = etree.iterparse(
            str(xml_path),
            events=("end",),
            tag=f"{NS}page",
        )

        try:
            for _, elem in context:
                try:
                    article = self._process_page_element(elem, page_id_set)
                    if article:
                        self.articles_processed += 1

                        # Log progress every 100 articles
                        if self.articles_processed % 100 == 0:
                            self.logger.info(
                                "parsing_progress",
                                articles_processed=self.articles_processed,
                                articles_skipped=self.articles_skipped,
                            )

                        yield article
                    else:
                        self.articles_skipped += 1

                except Exception as e:
                    self.logger.error("article_parsing_error", error=str(e))
                finally:
                    # Always clear element to free memory
                    elem.clear()

        finally:
            # Log final statistics
            self.logger.info(
                "parsing_complete",
                total_processed=self.articles_processed,
                total_skipped=self.articles_skipped,
            )

    def _process_page_element(
        self, elem: etree._Element, page_id_set: set[str] | None
    ) -> WikiArticle | None:
        """Process a single page element from XML.

        Args:
            elem: The page element to process
            page_id_set: Set of page IDs to filter, or None for all pages

        Returns:
            WikiArticle if page should be processed, None otherwise
        """
        # Extract and validate basic page data
        page_data = self._extract_page_data(elem)
        if not page_data:
            return None

        page_id, title, timestamp, wikitext = page_data

        # Apply page ID filter if provided
        if page_id_set and page_id not in page_id_set:
            return None

        # Convert wikitext to markdown
        markdown_content = self._convert_wikitext_to_markdown(wikitext)

        # Calculate word count
        word_count = len(markdown_content.split())

        # Create WikiArticle
        return WikiArticle(
            title=title,
            wiki_id=page_id,
            last_updated=timestamp,
            content=markdown_content,
            word_count=word_count,
        )

    def _extract_page_data(self, elem: etree._Element) -> tuple[str, str, str, str] | None:
        """Extract page data from XML element.

        Args:
            elem: The page element to extract data from

        Returns:
            Tuple of (page_id, title, timestamp, wikitext) or None if invalid
        """
        # Extract namespace - only process main articles (ns=0)
        ns_elem = elem.find(f"{NS}ns")
        if ns_elem is None or ns_elem.text != "0":
            return None

        # Extract page ID
        id_elem = elem.find(f"{NS}id")
        if id_elem is None or id_elem.text is None:
            self.logger.warning("missing_page_id")
            return None

        page_id = id_elem.text.strip()

        # Extract title
        title_elem = elem.find(f"{NS}title")
        if title_elem is None or title_elem.text is None:
            self.logger.warning("missing_title", page_id=page_id)
            return None

        title = title_elem.text.strip()

        # Extract revision info
        revision_elem = elem.find(f"{NS}revision")
        if revision_elem is None:
            self.logger.warning("missing_revision", page_id=page_id, title=title)
            return None

        # Extract timestamp
        timestamp_elem = revision_elem.find(f"{NS}timestamp")
        timestamp = (
            timestamp_elem.text if timestamp_elem is not None and timestamp_elem.text else ""
        )

        # Extract wikitext content
        text_elem = revision_elem.find(f"{NS}text")
        if text_elem is None or text_elem.text is None:
            self.logger.warning("missing_text", page_id=page_id, title=title)
            return None

        wikitext = text_elem.text

        return page_id, title, timestamp, wikitext

    def _convert_wikitext_to_markdown(self, wikitext: str) -> str:
        """Convert MediaWiki markup to Markdown.

        Args:
            wikitext: Raw MediaWiki markup text

        Returns:
            Markdown-formatted text
        """
        if not wikitext:
            return ""

        try:
            # Parse wikitext using mwparserfromhell
            parsed = mwparserfromhell.parse(wikitext)
            text = str(parsed)

            # Remove unwanted elements
            text = self._remove_templates_and_special_links(parsed, text)

            # Re-parse after removals
            parsed = mwparserfromhell.parse(text)

            # Convert wiki elements to markdown
            text = self._convert_wiki_elements(parsed, text)

            # Apply regex-based cleanup
            text = self._apply_regex_cleanup(text)

            return text.strip()

        except Exception as e:
            self.logger.error("wikitext_conversion_error", error=str(e))
            return wikitext

    def _remove_templates_and_special_links(
        self, parsed: mwparserfromhell.wikicode.Wikicode, text: str
    ) -> str:
        """Remove templates, File/Image embeds, and Category links.

        Args:
            parsed: Parsed wikicode object
            text: Current text to modify

        Returns:
            Modified text with unwanted elements removed
        """
        # Remove templates ({{...}})
        for template in parsed.filter_templates():
            text = text.replace(str(template), "")

        # Remove File/Image embeds and Category links
        for wikilink in parsed.filter_wikilinks():
            link_title = str(wikilink.title)
            if link_title.startswith(("File:", "Image:", "Category:")):
                text = text.replace(str(wikilink), "")

        return text

    def _convert_wiki_elements(self, parsed: mwparserfromhell.wikicode.Wikicode, text: str) -> str:
        """Convert wiki elements (headings, links, formatting) to markdown.

        Args:
            parsed: Parsed wikicode object
            text: Current text to modify

        Returns:
            Modified text with wiki elements converted to markdown
        """
        # Convert headings
        for heading in parsed.filter_headings():
            level = heading.level
            title_text = str(heading.title).strip()
            markdown_heading = f"{'#' * level} {title_text}"
            text = text.replace(str(heading), markdown_heading)

        # Convert wikilinks [[Link]] and [[Link|Display]]
        for wikilink in parsed.filter_wikilinks():
            link_title = str(wikilink.title)
            # Skip already-removed File/Image/Category links
            if link_title.startswith(("File:", "Image:", "Category:")):
                continue

            if wikilink.text:
                display_text = str(wikilink.text)
                markdown_link = f"[{display_text}]({link_title})"
            else:
                markdown_link = f"[{link_title}]({link_title})"

            text = text.replace(str(wikilink), markdown_link)

        # Convert external links
        for extlink in parsed.filter_external_links():
            if extlink.title:
                url = str(extlink.url)
                title_text = str(extlink.title).strip()
                markdown_link = f"[{title_text}]({url})"
                text = text.replace(str(extlink), markdown_link)

        return text

    def _apply_regex_cleanup(self, text: str) -> str:
        """Apply regex-based cleanup for common wiki markup patterns.

        Args:
            text: Text to clean up

        Returns:
            Cleaned text
        """
        # Convert bold '''text''' -> **text**
        text = re.sub(r"'''(.+?)'''", r"**\1**", text)

        # Convert italic ''text'' -> *text*
        text = re.sub(r"''(.+?)''", r"*\1*", text)

        # Convert unordered lists: * item -> - item
        text = re.sub(r"^\* ", "- ", text, flags=re.MULTILINE)

        # Convert ordered lists: # item -> 1. item
        text = re.sub(r"^# ", "1. ", text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def extract_internal_links(self, wikitext: str) -> list[str]:
        """Extract internal link targets from wikitext.

        Args:
            wikitext: Raw MediaWiki markup text

        Returns:
            Deduplicated list of internal link targets
        """
        if not wikitext:
            return []

        try:
            parsed = mwparserfromhell.parse(wikitext)
            links = []

            for wikilink in parsed.filter_wikilinks():
                link_title = str(wikilink.title).strip()
                # Skip File, Image, and Category links
                if not link_title.startswith(("File:", "Image:", "Category:")):
                    links.append(link_title)

            # Return deduplicated list
            return list(dict.fromkeys(links))

        except Exception as e:
            self.logger.error("link_extraction_error", error=str(e))
            return []
