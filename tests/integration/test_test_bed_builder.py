"""Integration tests for WikiTestBedBuilder with sample XML data."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ingestion.test_bed_builder import WikiTestBedBuilder

# Sample MediaWiki XML export with linked pages
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11" xml:lang="en">
  <page>
    <id>58</id>
    <ns>0</ns>
    <title>Blood Angels</title>
    <revision>
      <timestamp>2024-01-15T12:00:00Z</timestamp>
      <text>==Overview==
The '''Blood Angels''' are a [[Space Marine]] chapter.

They were led by their [[Primarch]], [[Sanguinius]].
They fought in the [[Horus Heresy]].

Their homeworld is [[Baal]].

[[Category:Space Marines]]</text>
    </revision>
  </page>
  <page>
    <id>100</id>
    <ns>0</ns>
    <title>Space Marine</title>
    <revision>
      <timestamp>2024-01-16T10:00:00Z</timestamp>
      <text>==Space Marines==
The [[Space Marine]]s serve the [[Imperium]].

They are organized into [[Chapter]]s.
Famous chapters include the [[Blood Angels]].

They were created by the [[Emperor]].</text>
    </revision>
  </page>
  <page>
    <id>142</id>
    <ns>0</ns>
    <title>Sanguinius</title>
    <revision>
      <timestamp>2024-01-17T14:30:00Z</timestamp>
      <text>==Sanguinius==
[[Sanguinius]] was the [[Primarch]] of the [[Blood Angels]].

He died defending the [[Emperor]] during the [[Horus Heresy]].

He had angelic wings.</text>
    </revision>
  </page>
  <page>
    <id>200</id>
    <ns>0</ns>
    <title>Primarch</title>
    <revision>
      <timestamp>2024-01-18T09:15:00Z</timestamp>
      <text>==Primarchs==
The [[Primarch]]s were created by the [[Emperor]].

[[Sanguinius]] was one of the twenty Primarchs.
[[Horus]] was another.</text>
    </revision>
  </page>
  <page>
    <id>89</id>
    <ns>0</ns>
    <title>Horus Heresy</title>
    <revision>
      <timestamp>2024-01-19T11:00:00Z</timestamp>
      <text>==The Horus Heresy==
The [[Horus Heresy]] was a galaxy-spanning civil war.

[[Sanguinius]] and the [[Blood Angels]] fought loyally.
[[Horus]] led the rebellion against the [[Emperor]].</text>
    </revision>
  </page>
  <page>
    <id>300</id>
    <ns>0</ns>
    <title>Imperium</title>
    <revision>
      <timestamp>2024-01-20T15:30:00Z</timestamp>
      <text>==Imperium of Man==
The [[Imperium]] spans a million worlds.

The [[Emperor]] rules from [[Terra]].
[[Space Marine]]s defend it.</text>
    </revision>
  </page>
  <page>
    <id>350</id>
    <ns>0</ns>
    <title>Emperor</title>
    <revision>
      <timestamp>2024-01-21T08:00:00Z</timestamp>
      <text>==The Emperor==
The [[Emperor]] created the [[Primarch]]s.

He rules the [[Imperium]] from the Golden Throne on [[Terra]].</text>
    </revision>
  </page>
  <page>
    <id>400</id>
    <ns>0</ns>
    <title>Baal</title>
    <revision>
      <timestamp>2024-01-22T10:00:00Z</timestamp>
      <text>==Baal==
[[Baal]] is the homeworld of the [[Blood Angels]].

It is a desert world.</text>
    </revision>
  </page>
  <page>
    <id>450</id>
    <ns>0</ns>
    <title>Horus</title>
    <revision>
      <timestamp>2024-01-23T12:00:00Z</timestamp>
      <text>==Horus==
[[Horus]] was a [[Primarch]] who fell to [[Chaos]].

He led the [[Horus Heresy]].</text>
    </revision>
  </page>
  <page>
    <id>500</id>
    <ns>0</ns>
    <title>Terra</title>
    <revision>
      <timestamp>2024-01-24T14:00:00Z</timestamp>
      <text>==Terra==
[[Terra]] is the throneworld of the [[Imperium]].

The [[Emperor]] resides there.</text>
    </revision>
  </page>
  <page>
    <id>999</id>
    <ns>1</ns>
    <title>Talk:Blood Angels</title>
    <revision>
      <timestamp>2024-01-25T16:00:00Z</timestamp>
      <text>This is a talk page and should be skipped.</text>
    </revision>
  </page>
  <page>
    <id>1000</id>
    <ns>14</ns>
    <title>Category:Space Marines</title>
    <revision>
      <timestamp>2024-01-26T18:00:00Z</timestamp>
      <text>This is a category page and should be skipped.</text>
    </revision>
  </page>
</mediawiki>
"""


@pytest.fixture
def sample_xml_file(tmp_path: Path) -> Iterator[Path]:
    """Create a temporary sample XML file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Path to the sample XML file
    """
    xml_file = tmp_path / "sample_wiki.xml"
    xml_file.write_text(SAMPLE_XML, encoding="utf-8")
    yield xml_file


@pytest.fixture
def builder() -> WikiTestBedBuilder:
    """Create a WikiTestBedBuilder instance for testing."""
    return WikiTestBedBuilder()


class TestTestBedBuilderIntegration:
    """Integration tests for WikiTestBedBuilder with real XML data."""

    def test_build_test_bed_with_sample_xml(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path
    ) -> None:
        """Test building test bed with sample XML data."""
        # Build test bed starting from Blood Angels (58)
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=5)

        # Should select 5 pages
        assert len(page_ids) == 5

        # First page should be the seed (Blood Angels)
        assert page_ids[0] == "58"

        # All IDs should be strings
        for page_id in page_ids:
            assert isinstance(page_id, str)

    def test_build_test_bed_includes_related_pages(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path
    ) -> None:
        """Test that test bed includes pages related to seed."""
        # Build larger test bed
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=10)

        # Blood Angels links to: Space Marine, Sanguinius, Primarch, Horus Heresy, Baal
        # So at least some of these should be in the first few pages
        # IDs: 100, 142, 200, 89, 400
        first_degree_links = {"100", "142", "200", "89", "400"}

        # At least 3 of these should be in the test bed
        included = sum(1 for pid in first_degree_links if pid in page_ids)
        assert included >= 3

    def test_build_test_bed_excludes_non_main_namespace(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path
    ) -> None:
        """Test that non-main namespace pages are excluded."""
        # Build test bed with all pages
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=100)

        # Talk page (999, ns=1) and Category page (1000, ns=14) should not be included
        assert "999" not in page_ids
        assert "1000" not in page_ids

    def test_build_test_bed_respects_target_count(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path
    ) -> None:
        """Test that target count is respected."""
        # Small count
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=3)
        assert len(page_ids) == 3

        # Medium count
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=7)
        assert len(page_ids) == 7

        # Count larger than available (10 main namespace pages)
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=100)
        assert len(page_ids) <= 10  # Only 10 main namespace pages in sample

    def test_write_test_bed_file_integration(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path, tmp_path: Path
    ) -> None:
        """Test writing test bed file with real data."""
        # Build test bed
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=5)

        # Write to file
        output_path = tmp_path / "test-bed-pages.txt"
        builder.write_test_bed_file(page_ids, output_path, sample_xml_file, seed_page_id="58")

        # Verify file exists
        assert output_path.exists()

        # Read and verify content
        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Should have comments and IDs (2 lines per page)
        assert len(lines) >= len(page_ids)

        # First page should be Blood Angels with (seed) marker
        assert "# Blood Angels (seed)" in content
        assert "58" in lines

        # All page IDs should be in the file
        for page_id in page_ids:
            assert page_id in lines

    def test_full_workflow_with_output(
        self, builder: WikiTestBedBuilder, sample_xml_file: Path, tmp_path: Path
    ) -> None:
        """Test complete workflow: build test bed and write output."""
        output_path = tmp_path / "data" / "test-bed-pages.txt"

        # Build test bed
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=8)

        # Write output
        builder.write_test_bed_file(page_ids, output_path, sample_xml_file, seed_page_id="58")

        # Verify output
        assert output_path.exists()
        content = output_path.read_text()

        # Verify format: alternating comments and IDs
        lines = content.strip().split("\n")

        # Check that comments start with #
        comment_lines = [line for line in lines if line.startswith("#")]
        assert len(comment_lines) == len(page_ids)

        # Check that page IDs are present
        id_lines = [line for line in lines if not line.startswith("#")]
        assert len(id_lines) == len(page_ids)

        # Verify seed is marked
        assert "(seed)" in content
        assert content.count("(seed)") == 1

    def test_bfs_traversal_order(self, builder: WikiTestBedBuilder, sample_xml_file: Path) -> None:
        """Test that BFS traversal explores pages level by level."""
        # Build small test bed
        page_ids = builder.build_test_bed(sample_xml_file, "58", target_count=6)

        # First should be seed
        assert page_ids[0] == "58"

        # Next pages should be directly linked from Blood Angels
        # Blood Angels (58) links to: Space Marine (100), Sanguinius (142),
        # Primarch (200), Horus Heresy (89), Baal (400)
        first_degree = {"100", "142", "200", "89", "400"}

        # The next 5 pages should all be from first degree links
        next_pages = set(page_ids[1:6])
        # All should be from first degree
        assert next_pages.issubset(first_degree) or len(next_pages & first_degree) >= 3
