"""Integration tests for XML parsing with sample XML data."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.ingestion.markdown_archive import save_markdown_file
from src.ingestion.wiki_xml_parser import WikiXMLParser

# Sample MediaWiki XML export
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11" xml:lang="en">
  <page>
    <id>1</id>
    <ns>0</ns>
    <title>Blood Angels</title>
    <revision>
      <timestamp>2024-01-15T12:00:00Z</timestamp>
      <text>==Overview==
The '''Blood Angels''' are one of the twenty Space Marine Chapters.

===History===
Founded during the [[Great Crusade]], they are led by [[Sanguinius]].

* Noble warriors
* Red armor
* [[Baal]] is their homeworld

[[Category:Space Marines]]
[[File:BloodAngels.jpg|thumb]]</text>
    </revision>
  </page>
  <page>
    <id>2</id>
    <ns>1</ns>
    <title>Talk:Space Marines</title>
    <revision>
      <timestamp>2024-01-16T10:00:00Z</timestamp>
      <text>This is a talk page and should be skipped.</text>
    </revision>
  </page>
  <page>
    <id>3</id>
    <ns>0</ns>
    <title>Imperium</title>
    <revision>
      <timestamp>2024-01-17T14:30:00Z</timestamp>
      <text>The ''Imperium of Man'' is vast.

[http://example.com Official Site]

# First point
# Second point</text>
    </revision>
  </page>
  <page>
    <id>4</id>
    <ns>0</ns>
    <title>Chaos</title>
    <revision>
      <timestamp>2024-01-18T09:15:00Z</timestamp>
      <text>===Dark Powers===
[[Chaos Gods|The Ruinous Powers]] corrupt all.

{{Quote|Beware the daemon}}</text>
    </revision>
  </page>
  <page>
    <id>5</id>
    <ns>6</ns>
    <title>File:SomeImage.png</title>
    <revision>
      <timestamp>2024-01-19T08:00:00Z</timestamp>
      <text>File description - should be skipped.</text>
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
def archive_path(tmp_path: Path) -> Path:
    """Create a temporary archive directory for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the temporary archive directory
    """
    archive = tmp_path / "markdown-archive"
    archive.mkdir(parents=True, exist_ok=True)
    return archive


class TestXMLParsingIntegration:
    """Integration tests for end-to-end XML parsing."""

    def test_parse_sample_xml_end_to_end(self, sample_xml_file: Path, archive_path: Path) -> None:
        """Test complete XML parsing pipeline with sample data."""
        parser = WikiXMLParser()

        # Parse XML and collect articles
        articles = list(parser.parse_xml_export(sample_xml_file))

        # Should only get ns=0 pages (3 articles: Blood Angels, Imperium, Chaos)
        assert len(articles) == 3
        assert parser.articles_processed == 3
        assert parser.articles_skipped == 2  # Talk page and File page

        # Verify article titles
        titles = [article.title for article in articles]
        assert "Blood Angels" in titles
        assert "Imperium" in titles
        assert "Chaos" in titles
        assert "Talk:Space Marines" not in titles  # Should be filtered out
        assert "File:SomeImage.png" not in titles  # Should be filtered out

    def test_namespace_filtering(self, sample_xml_file: Path) -> None:
        """Test that only ns=0 pages are processed."""
        parser = WikiXMLParser()
        articles = list(parser.parse_xml_export(sample_xml_file))

        # All articles should have namespace 0 (main articles)
        for article in articles:
            # Verify these are main articles, not talk pages or file pages
            assert not article.title.startswith("Talk:")
            assert not article.title.startswith("File:")

    def test_markdown_conversion_quality(self, sample_xml_file: Path) -> None:
        """Test quality of MediaWiki to Markdown conversion."""
        parser = WikiXMLParser()
        articles = {article.title: article for article in parser.parse_xml_export(sample_xml_file)}

        # Check Blood Angels article
        blood_angels = articles["Blood Angels"]
        assert "## Overview" in blood_angels.content
        assert "### History" in blood_angels.content
        assert "**Blood Angels**" in blood_angels.content  # Bold
        assert "- Noble warriors" in blood_angels.content  # List
        assert "[Sanguinius](Sanguinius)" in blood_angels.content  # Internal link
        assert "[[Category:" not in blood_angels.content  # Category removed
        assert "[[File:" not in blood_angels.content  # File removed

        # Check Imperium article
        imperium = articles["Imperium"]
        assert "*Imperium of Man*" in imperium.content  # Italic
        assert "[Official Site](http://example.com)" in imperium.content  # External link
        assert "1. First point" in imperium.content  # Ordered list

        # Check Chaos article
        chaos = articles["Chaos"]
        assert "### Dark Powers" in chaos.content  # Heading level 3
        assert "{{Quote" not in chaos.content  # Template removed

    def test_page_id_filtering(self, sample_xml_file: Path) -> None:
        """Test filtering by page IDs."""
        parser = WikiXMLParser()

        # Filter to only page IDs 1 and 3 (Blood Angels and Imperium)
        articles = list(parser.parse_xml_export(sample_xml_file, page_ids=["1", "3"]))

        assert len(articles) == 2
        titles = [article.title for article in articles]
        assert "Blood Angels" in titles
        assert "Imperium" in titles
        assert "Chaos" not in titles  # ID 4, should be filtered out

    def test_save_to_markdown_archive(self, sample_xml_file: Path, archive_path: Path) -> None:
        """Test saving parsed articles to markdown archive."""
        parser = WikiXMLParser()

        # Parse and save articles
        for article in parser.parse_xml_export(sample_xml_file):
            save_markdown_file(article, archive_path)

        # Verify files were created
        assert (archive_path / "Blood_Angels.md").exists()
        assert (archive_path / "Imperium.md").exists()
        assert (archive_path / "Chaos.md").exists()

        # Verify file content includes YAML frontmatter
        blood_angels_content = (archive_path / "Blood_Angels.md").read_text()
        assert "---\n" in blood_angels_content  # YAML frontmatter delimiters
        assert "title: Blood Angels" in blood_angels_content
        assert "wiki_id: '1'" in blood_angels_content
        assert "## Overview" in blood_angels_content

    def test_metadata_extraction(self, sample_xml_file: Path) -> None:
        """Test metadata extraction from parsed articles."""
        parser = WikiXMLParser()
        articles = {article.title: article for article in parser.parse_xml_export(sample_xml_file)}

        # Check Blood Angels metadata
        blood_angels = articles["Blood Angels"]
        assert blood_angels.wiki_id == "1"
        assert blood_angels.last_updated == "2024-01-15T12:00:00Z"
        assert blood_angels.word_count > 0
        assert isinstance(blood_angels.word_count, int)

        # Check Imperium metadata
        imperium = articles["Imperium"]
        assert imperium.wiki_id == "3"
        assert imperium.last_updated == "2024-01-17T14:30:00Z"

    def test_empty_page_ids_list(self, sample_xml_file: Path) -> None:
        """Test that empty page_ids list is treated as no filter (all articles)."""
        parser = WikiXMLParser()
        articles = list(parser.parse_xml_export(sample_xml_file, page_ids=[]))

        # Empty filter should process all articles (same as None)
        assert len(articles) == 3

    def test_nonexistent_page_ids(self, sample_xml_file: Path) -> None:
        """Test filtering with page IDs that don't exist."""
        parser = WikiXMLParser()
        articles = list(parser.parse_xml_export(sample_xml_file, page_ids=["999", "888"]))

        # No articles should match
        assert len(articles) == 0

    def test_parser_statistics(self, sample_xml_file: Path) -> None:
        """Test parser statistics tracking."""
        parser = WikiXMLParser()
        list(parser.parse_xml_export(sample_xml_file))

        assert parser.articles_processed == 3  # 3 main namespace articles
        assert parser.articles_skipped == 2  # 2 non-main namespace pages


class TestPageIDFileFiltering:
    """Test page ID filtering from file."""

    def test_page_ids_from_file(self, sample_xml_file: Path, tmp_path: Path) -> None:
        """Test loading page IDs from file and filtering."""
        # Create page IDs file
        page_ids_file = tmp_path / "page_ids.txt"
        page_ids_file.write_text("1\n3\n")

        # Load page IDs
        page_ids = page_ids_file.read_text().strip().split("\n")
        page_ids = [pid.strip() for pid in page_ids if pid.strip()]

        # Parse with filter
        parser = WikiXMLParser()
        articles = list(parser.parse_xml_export(sample_xml_file, page_ids=page_ids))

        assert len(articles) == 2
        titles = [article.title for article in articles]
        assert "Blood Angels" in titles
        assert "Imperium" in titles


class TestRedirectHandlingIntegration:
    """Integration tests for end-to-end redirect handling."""

    def test_end_to_end_redirect_handling(self, tmp_path: Path) -> None:
        """Test complete redirect handling pipeline with redirects and link resolution."""
        # Create sample XML with redirects and articles that link to them
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Redirect Source</title>
    <ns>0</ns>
    <id>1</id>
    <redirect title="Target Article" />
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>#REDIRECT [[Target Article]]</text>
    </revision>
  </page>
  <page>
    <title>Mankind</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="Humans" />
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>#REDIRECT [[Humans]]</text>
    </revision>
  </page>
  <page>
    <title>Article With Links</title>
    <ns>0</ns>
    <id>3</id>
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>==Overview==
See [[Redirect Source]] for more info.
The [[Mankind|human race]] is vast.
Normal link to [[Target Article]] works too.</text>
    </revision>
  </page>
  <page>
    <title>Target Article</title>
    <ns>0</ns>
    <id>4</id>
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>This is the canonical article.</text>
    </revision>
  </page>
  <page>
    <title>Humans</title>
    <ns>0</ns>
    <id>5</id>
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>Human article content.</text>
    </revision>
  </page>
</mediawiki>"""

        xml_file = tmp_path / "redirect_test.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        parser = WikiXMLParser()
        articles = list(parser.parse_xml_export(xml_file))

        # Verify redirect pages are skipped
        titles = [article.title for article in articles]
        assert "Redirect Source" not in titles  # Redirect should be skipped
        assert "Mankind" not in titles  # Redirect should be skipped
        assert "Article With Links" in titles
        assert "Target Article" in titles
        assert "Humans" in titles

        # Should have 3 articles (2 redirects skipped)
        assert len(articles) == 3
        assert parser.redirects_found == 2
        assert parser.redirects_skipped == 2

        # Verify links in "Article With Links" are resolved
        article_with_links = next(a for a in articles if a.title == "Article With Links")

        # Link to "Redirect Source" should be resolved to "Target Article"
        assert "[Target Article](Target Article)" in article_with_links.content
        assert "[Redirect Source]" not in article_with_links.content

        # Link with display text should preserve display text but resolve target
        assert "[human race](Humans)" in article_with_links.content
        assert "[Mankind]" not in article_with_links.content

        # Links resolved count should be tracked
        assert parser.links_resolved >= 2
