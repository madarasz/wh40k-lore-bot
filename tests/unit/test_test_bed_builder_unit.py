"""Unit tests for WikiTestBedBuilder page selection."""

from pathlib import Path

import pytest

from src.ingestion.test_bed_builder import WikiTestBedBuilder


@pytest.fixture
def builder() -> WikiTestBedBuilder:
    """Create a WikiTestBedBuilder instance for testing."""
    return WikiTestBedBuilder()


@pytest.fixture
def sample_xml(tmp_path: Path) -> Path:
    """Create a sample MediaWiki XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Blood Angels</title>
    <ns>0</ns>
    <id>58</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
The [[Blood Angels]] are a [[Space Marine]] chapter.
They follow [[Sanguinius]], their [[Primarch]].
They fought in the [[Horus Heresy]].
      </text>
    </revision>
  </page>
  <page>
    <title>Space Marine</title>
    <ns>0</ns>
    <id>100</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
[[Space Marine]]s serve the [[Imperium]].
They are organized into [[Chapter]]s.
      </text>
    </revision>
  </page>
  <page>
    <title>Sanguinius</title>
    <ns>0</ns>
    <id>142</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
[[Sanguinius]] was the [[Primarch]] of the [[Blood Angels]].
He died in the [[Horus Heresy]].
      </text>
    </revision>
  </page>
  <page>
    <title>Primarch</title>
    <ns>0</ns>
    <id>200</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
The [[Primarch]]s were created by the [[Emperor]].
[[Sanguinius]] was one of them.
      </text>
    </revision>
  </page>
  <page>
    <title>Horus Heresy</title>
    <ns>0</ns>
    <id>89</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
The [[Horus Heresy]] was a galaxy-spanning civil war.
[[Sanguinius]] fought in it.
      </text>
    </revision>
  </page>
  <page>
    <title>Imperium</title>
    <ns>0</ns>
    <id>300</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>
The [[Imperium]] is vast.
      </text>
    </revision>
  </page>
  <page>
    <title>Category:Space Marines</title>
    <ns>14</ns>
    <id>999</id>
    <revision>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <text>Category page (should be skipped)</text>
    </revision>
  </page>
</mediawiki>
"""
    xml_file = tmp_path / "test_wiki.xml"
    xml_file.write_text(xml_content)
    return xml_file


class TestTitleToIdMapping:
    """Test title to ID mapping construction."""

    def test_build_title_to_id_map(self, builder: WikiTestBedBuilder, sample_xml: Path) -> None:
        """Test building title→ID mapping from XML."""
        title_to_id = builder._build_title_to_id_map(sample_xml)

        # Check expected mappings
        assert title_to_id["Blood Angels"] == "58"
        assert title_to_id["Blood_Angels"] == "58"
        assert title_to_id["Space Marine"] == "100"
        assert title_to_id["Space_Marine"] == "100"
        assert title_to_id["Sanguinius"] == "142"
        assert title_to_id["Primarch"] == "200"
        assert title_to_id["Horus Heresy"] == "89"
        assert title_to_id["Horus_Heresy"] == "89"

    def test_build_title_to_id_map_skips_non_main_namespace(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that non-main namespace pages are skipped."""
        title_to_id = builder._build_title_to_id_map(sample_xml)

        # Category page (ns=14) should not be included
        assert "Category:Space Marines" not in title_to_id

    def test_build_title_to_id_map_with_empty_xml(
        self, builder: WikiTestBedBuilder, tmp_path: Path
    ) -> None:
        """Test mapping construction with empty XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
</mediawiki>
"""
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text(xml_content)

        title_to_id = builder._build_title_to_id_map(xml_file)
        assert len(title_to_id) == 0


class TestLinkExtraction:
    """Test link extraction from pages."""

    def test_extract_links_from_page(self, builder: WikiTestBedBuilder, sample_xml: Path) -> None:
        """Test extracting links from Blood Angels page."""
        links = builder._extract_links_from_page("58", sample_xml)

        # Blood Angels page links to: Blood Angels, Space Marine, Sanguinius, Primarch, Horus Heresy
        assert "Blood_Angels" in links
        assert "Space_Marine" in links
        assert "Sanguinius" in links
        assert "Primarch" in links
        assert "Horus_Heresy" in links

    def test_extract_links_from_nonexistent_page(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test extracting links from non-existent page returns empty list."""
        links = builder._extract_links_from_page("99999", sample_xml)
        assert links == []

    def test_extract_links_normalizes_titles(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that link titles are normalized (spaces → underscores)."""
        links = builder._extract_links_from_page("58", sample_xml)

        # All links should have underscores instead of spaces
        for link in links:
            assert " " not in link or link.startswith("File:") or link.startswith("Category:")


class TestBFSTraversal:
    """Test breadth-first search traversal."""

    def test_build_test_bed_basic(self, builder: WikiTestBedBuilder, sample_xml: Path) -> None:
        """Test basic test bed construction."""
        page_ids = builder.build_test_bed(sample_xml, "58", target_count=5)

        # Should select 5 pages starting from Blood Angels (58)
        assert len(page_ids) == 5
        assert page_ids[0] == "58"  # Seed page should be first

    def test_build_test_bed_starts_with_seed(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that test bed starts with seed page."""
        page_ids = builder.build_test_bed(sample_xml, "58", target_count=3)

        assert page_ids[0] == "58"

    def test_build_test_bed_respects_target_count(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that target count is respected."""
        page_ids = builder.build_test_bed(sample_xml, "58", target_count=3)
        assert len(page_ids) == 3

        page_ids = builder.build_test_bed(sample_xml, "58", target_count=10)
        assert len(page_ids) <= 10  # May be less if not enough pages

    def test_build_test_bed_with_invalid_seed(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that invalid seed page raises ValueError."""
        with pytest.raises(ValueError, match="Seed page ID 99999 not found"):
            builder.build_test_bed(sample_xml, "99999", target_count=5)

    def test_build_test_bed_includes_linked_pages(
        self, builder: WikiTestBedBuilder, sample_xml: Path
    ) -> None:
        """Test that BFS includes pages linked from seed."""
        page_ids = builder.build_test_bed(sample_xml, "58", target_count=10)

        # Blood Angels (58) links to Space Marine (100), Sanguinius (142), etc.
        # So these should be included
        assert "58" in page_ids  # Seed
        # At least some of the linked pages should be included
        linked_pages = {"100", "142", "200", "89"}
        included_count = sum(1 for pid in linked_pages if pid in page_ids)
        assert included_count > 0

    def test_build_test_bed_with_nonexistent_file(self, builder: WikiTestBedBuilder) -> None:
        """Test that non-existent XML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            builder.build_test_bed("nonexistent.xml", "58", target_count=10)


class TestStatistics:
    """Test statistics logging."""

    def test_log_statistics(self, builder: WikiTestBedBuilder) -> None:
        """Test statistics logging doesn't raise errors."""
        selected = ["58", "100", "142"]
        depth_map = {"58": 0, "100": 1, "142": 1}
        incoming_links = {"100": 2, "142": 3}
        id_to_title = {"58": "Blood Angels", "100": "Space Marine", "142": "Sanguinius"}

        # Should not raise any errors
        builder._log_statistics(selected, depth_map, incoming_links, id_to_title)

    def test_log_statistics_with_empty_data(self, builder: WikiTestBedBuilder) -> None:
        """Test statistics logging with empty data."""
        selected: list[str] = []
        depth_map: dict[str, int] = {}
        incoming_links: dict[str, int] = {}
        id_to_title: dict[str, str] = {}

        # Should not raise any errors
        builder._log_statistics(selected, depth_map, incoming_links, id_to_title)


class TestOutputFile:
    """Test output file generation."""

    def test_write_test_bed_file(
        self, builder: WikiTestBedBuilder, sample_xml: Path, tmp_path: Path
    ) -> None:
        """Test writing test bed file with comments."""
        page_ids = ["58", "100", "142"]
        output_path = tmp_path / "test-bed.txt"

        builder.write_test_bed_file(page_ids, output_path, sample_xml, seed_page_id="58")

        # Check file was created
        assert output_path.exists()

        # Check content
        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Should have comments and IDs
        assert "# Blood Angels (seed)" in content
        assert "58" in lines
        assert "# Space Marine" in content or "# Space_Marine" in content
        assert "100" in lines
        assert "# Sanguinius" in content
        assert "142" in lines

    def test_write_test_bed_file_creates_parent_directory(
        self, builder: WikiTestBedBuilder, sample_xml: Path, tmp_path: Path
    ) -> None:
        """Test that parent directory is created if it doesn't exist."""
        page_ids = ["58", "100"]
        output_path = tmp_path / "nested" / "dir" / "test-bed.txt"

        builder.write_test_bed_file(page_ids, output_path, sample_xml, seed_page_id="58")

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_write_test_bed_file_marks_seed_page(
        self, builder: WikiTestBedBuilder, sample_xml: Path, tmp_path: Path
    ) -> None:
        """Test that seed page is marked with (seed) comment."""
        page_ids = ["58", "100"]
        output_path = tmp_path / "test-bed.txt"

        builder.write_test_bed_file(page_ids, output_path, sample_xml, seed_page_id="58")

        content = output_path.read_text()
        assert "(seed)" in content

        # Only seed page should have (seed) marker
        seed_count = content.count("(seed)")
        assert seed_count == 1
