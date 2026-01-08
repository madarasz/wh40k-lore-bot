"""Test bed builder for selecting a curated subset of wiki pages for testing."""

from collections import defaultdict, deque
from pathlib import Path

import structlog
from lxml import etree

from src.ingestion.wiki_xml_parser import NS, WikiXMLParser

logger = structlog.get_logger(__name__)


class WikiTestBedBuilder:
    """Builds a curated test set of wiki pages using breadth-first traversal.

    Uses BFS starting from a seed page to select ~100 related pages for
    testing and fine-tuning the RAG pipeline before processing the full dataset.
    """

    def __init__(self) -> None:
        """Initialize the WikiTestBedBuilder."""
        self.logger = logger.bind(component="test_bed_builder")
        self.parser = WikiXMLParser()

    def build_test_bed(
        self,
        xml_path: str | Path,
        seed_page_id: str,
        target_count: int = 100,
    ) -> list[str]:
        """Build a test bed of related wiki pages using BFS traversal.

        Args:
            xml_path: Path to the MediaWiki XML export file
            seed_page_id: Starting page ID (e.g., "58" for Blood Angels)
            target_count: Target number of pages to select

        Returns:
            List of page IDs in traversal order

        Raises:
            FileNotFoundError: If XML file doesn't exist
            ValueError: If seed page ID not found in XML
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        self.logger.info(
            "building_test_bed",
            xml_path=str(xml_path),
            seed_page_id=seed_page_id,
            target_count=target_count,
        )

        # Build title→ID mapping for link resolution
        title_to_id = self._build_title_to_id_map(xml_path)
        id_to_title = {v: k for k, v in title_to_id.items()}

        # Verify seed page exists
        if seed_page_id not in id_to_title:
            raise ValueError(f"Seed page ID {seed_page_id} not found in XML")

        # BFS setup
        queue: deque[str] = deque([seed_page_id])
        visited: set[str] = {seed_page_id}
        incoming_links: dict[str, int] = defaultdict(int)
        selected: list[str] = [seed_page_id]

        # Track depth for statistics
        depth_map: dict[str, int] = {seed_page_id: 0}

        # BFS traversal
        while queue and len(selected) < target_count:
            page_id = queue.popleft()
            current_depth = depth_map[page_id]

            # Extract links from this page
            links = self._extract_links_from_page(page_id, xml_path)

            # Add unvisited linked pages
            for link_title in links:
                linked_id = title_to_id.get(link_title)
                if linked_id and linked_id not in visited:
                    visited.add(linked_id)
                    queue.append(linked_id)
                    selected.append(linked_id)
                    incoming_links[linked_id] += 1
                    depth_map[linked_id] = current_depth + 1

                    # Stop if we've reached target count
                    if len(selected) >= target_count:
                        break

            # Reorder queue by incoming link count (hub pages first)
            queue = deque(sorted(queue, key=lambda x: incoming_links[x], reverse=True))

        # Log statistics
        self._log_statistics(selected, depth_map, incoming_links, id_to_title)

        self.logger.info(
            "test_bed_complete",
            pages_selected=len(selected),
            target_count=target_count,
        )

        return selected[:target_count]

    def _build_title_to_id_map(self, xml_path: Path) -> dict[str, str]:
        """Build a mapping of normalized page titles to page IDs.

        Args:
            xml_path: Path to the MediaWiki XML export file

        Returns:
            Dictionary mapping normalized title to page ID
        """
        self.logger.info("building_title_to_id_map", xml_path=str(xml_path))

        title_to_id: dict[str, str] = {}

        # Use iterparse for memory efficiency
        context = etree.iterparse(
            str(xml_path),
            events=("end",),
            tag=f"{NS}page",
        )

        count = 0
        try:
            for _, elem in context:
                try:
                    # Extract namespace - only process main articles (ns=0)
                    ns_elem = elem.find(f"{NS}ns")
                    if ns_elem is None or ns_elem.text != "0":
                        continue

                    # Extract page ID
                    id_elem = elem.find(f"{NS}id")
                    if id_elem is None or id_elem.text is None:
                        continue

                    # Extract title
                    title_elem = elem.find(f"{NS}title")
                    if title_elem is None or title_elem.text is None:
                        continue

                    page_id = id_elem.text.strip()
                    title = title_elem.text.strip()

                    # Normalize title (replace spaces with underscores for link matching)
                    normalized_title = title.replace(" ", "_")
                    title_to_id[normalized_title] = page_id

                    # Also store with original spacing for flexibility
                    title_to_id[title] = page_id

                    count += 1

                    # Log progress every 1000 pages
                    if count % 1000 == 0:
                        self.logger.info("title_mapping_progress", titles_mapped=count)

                finally:
                    # Clear element to free memory
                    elem.clear()

        finally:
            self.logger.info("title_mapping_complete", total_titles=count)

        return title_to_id

    def _extract_links_from_page(self, page_id: str, xml_path: Path) -> list[str]:
        """Extract internal link targets from a specific page.

        Args:
            page_id: Page ID to extract links from
            xml_path: Path to the MediaWiki XML export file

        Returns:
            List of link targets (page titles)
        """
        # Use iterparse to find the specific page
        context = etree.iterparse(
            str(xml_path),
            events=("end",),
            tag=f"{NS}page",
        )

        try:
            for _, elem in context:
                try:
                    # Check if this is the page we want
                    id_elem = elem.find(f"{NS}id")
                    if id_elem is None or id_elem.text is None:
                        continue

                    if id_elem.text.strip() != page_id:
                        continue

                    # Found the page - extract wikitext
                    revision_elem = elem.find(f"{NS}revision")
                    if revision_elem is None:
                        return []

                    text_elem = revision_elem.find(f"{NS}text")
                    if text_elem is None or text_elem.text is None:
                        return []

                    wikitext = text_elem.text

                    # Use WikiXMLParser to extract links
                    links = self.parser.extract_internal_links(wikitext)

                    # Normalize link titles (replace spaces with underscores)
                    normalized_links = [link.replace(" ", "_") for link in links]

                    return normalized_links

                finally:
                    elem.clear()

        finally:
            pass

        return []

    def _log_statistics(
        self,
        selected: list[str],
        depth_map: dict[str, int],
        incoming_links: dict[str, int],
        id_to_title: dict[str, str],
    ) -> None:
        """Log statistics about the test bed selection.

        Args:
            selected: List of selected page IDs
            depth_map: Mapping of page ID to depth from seed
            incoming_links: Mapping of page ID to incoming link count
            id_to_title: Mapping of page ID to title
        """
        # Calculate max depth
        max_depth = max(depth_map.values()) if depth_map else 0

        # Get top 10 most-linked pages
        top_pages = sorted(
            incoming_links.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        top_pages_info = [
            {"id": pid, "title": id_to_title.get(pid, "Unknown"), "links": count}
            for pid, count in top_pages
        ]

        self.logger.info(
            "test_bed_statistics",
            total_pages=len(selected),
            max_depth=max_depth,
            top_pages=top_pages_info,
        )

    def write_test_bed_file(
        self,
        page_ids: list[str],
        output_path: str | Path,
        xml_path: str | Path,
        seed_page_id: str,
    ) -> None:
        """Write test bed page IDs to output file.

        Args:
            page_ids: List of page IDs to write
            output_path: Path to output file
            xml_path: Path to XML file (for title lookup)
            seed_page_id: Seed page ID for marking
        """
        output_path = Path(output_path)
        xml_path = Path(xml_path)

        self.logger.info("writing_test_bed_file", output_path=str(output_path))

        # Build title mapping for comments
        title_to_id = self._build_title_to_id_map(xml_path)
        id_to_title = {v: k for k, v in title_to_id.items()}

        # Write file with comments
        lines: list[str] = []
        for page_id in page_ids:
            title = id_to_title.get(page_id, "Unknown")
            # Add seed marker
            if page_id == seed_page_id:
                lines.append(f"# {title} (seed)")
            else:
                lines.append(f"# {title}")
            lines.append(page_id)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path.write_text("\n".join(lines) + "\n")

        self.logger.info(
            "test_bed_file_written",
            output_path=str(output_path),
            page_count=len(page_ids),
        )
