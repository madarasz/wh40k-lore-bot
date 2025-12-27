"""Unit tests for WikiXMLParser MediaWiki to Markdown conversion."""

import pytest

from src.ingestion.wiki_xml_parser import WikiXMLParser


@pytest.fixture
def parser() -> WikiXMLParser:
    """Create a WikiXMLParser instance for testing."""
    return WikiXMLParser()


class TestWikitextToMarkdownConversion:
    """Test MediaWiki markup to Markdown conversion."""

    def test_heading_conversion_level_2(self, parser: WikiXMLParser) -> None:
        """Test conversion of ==Heading== to ## Heading."""
        wikitext = "==Blood Angels=="
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "## Blood Angels" in result

    def test_heading_conversion_level_3(self, parser: WikiXMLParser) -> None:
        """Test conversion of ===Heading=== to ### Heading."""
        wikitext = "===Chapter History==="
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "### Chapter History" in result

    def test_heading_conversion_level_4(self, parser: WikiXMLParser) -> None:
        """Test conversion of ====Heading==== to #### Heading."""
        wikitext = "====The Horus Heresy===="
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "#### The Horus Heresy" in result

    def test_bold_conversion(self, parser: WikiXMLParser) -> None:
        """Test conversion of '''bold''' to **bold**."""
        wikitext = "The '''Primarch''' was Sanguinius."
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "**Primarch**" in result

    def test_italic_conversion(self, parser: WikiXMLParser) -> None:
        """Test conversion of ''italic'' to *italic*."""
        wikitext = "The ''Codex Astartes'' is sacred."
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "*Codex Astartes*" in result

    def test_simple_internal_link(self, parser: WikiXMLParser) -> None:
        """Test conversion of [[Link]] to [Link](Link)."""
        wikitext = "The [[Imperium]] is vast."
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "[Imperium](Imperium)" in result

    def test_internal_link_with_display_text(self, parser: WikiXMLParser) -> None:
        """Test conversion of [[Link|Display]] to [Display](Link)."""
        wikitext = "[[Blood Angels|The Blood Angels]] are noble."
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "[The Blood Angels](Blood Angels)" in result

    def test_external_link_conversion(self, parser: WikiXMLParser) -> None:
        """Test conversion of [http://url text] to [text](http://url)."""
        wikitext = "[http://example.com Official Site]"
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "[Official Site](http://example.com)" in result

    def test_unordered_list_conversion(self, parser: WikiXMLParser) -> None:
        """Test conversion of * item to - item."""
        wikitext = "* First item\n* Second item"
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "- First item" in result
        assert "- Second item" in result

    def test_ordered_list_conversion(self, parser: WikiXMLParser) -> None:
        """Test conversion of # item to 1. item."""
        wikitext = "# First step\n# Second step"
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "1. First step" in result
        assert "1. Second step" in result

    def test_template_removal(self, parser: WikiXMLParser) -> None:
        """Test removal of {{Template}} markup."""
        wikitext = "Text {{Quote|Some quote}} more text."
        result = parser._convert_wikitext_to_markdown(wikitext)
        # Template should be removed
        assert "{{Quote" not in result
        assert "Text" in result
        assert "more text" in result

    def test_file_embed_removal(self, parser: WikiXMLParser) -> None:
        """Test removal of [[File:...]] embeds."""
        wikitext = "Text [[File:Image.jpg|thumb|Caption]] more text."
        result = parser._convert_wikitext_to_markdown(wikitext)
        # File embed should be removed
        assert "[[File:" not in result
        assert "Text" in result
        assert "more text" in result

    def test_image_embed_removal(self, parser: WikiXMLParser) -> None:
        """Test removal of [[Image:...]] embeds."""
        wikitext = "Text [[Image:Photo.png]] more text."
        result = parser._convert_wikitext_to_markdown(wikitext)
        # Image embed should be removed
        assert "[[Image:" not in result
        assert "Text" in result
        assert "more text" in result

    def test_category_removal(self, parser: WikiXMLParser) -> None:
        """Test removal of [[Category:...]] links."""
        wikitext = "Text content. [[Category:Space Marines]]"
        result = parser._convert_wikitext_to_markdown(wikitext)
        # Category should be removed
        assert "[[Category:" not in result
        assert "Text content" in result

    def test_complex_nested_formatting(self, parser: WikiXMLParser) -> None:
        """Test conversion of complex nested formatting."""
        wikitext = "==Overview==\nThe '''[[Blood Angels]]''' are ''legendary''."
        result = parser._convert_wikitext_to_markdown(wikitext)
        assert "## Overview" in result
        assert "**" in result  # Bold marker
        assert "*" in result  # Italic marker (not part of **)
        assert "[Blood Angels](Blood Angels)" in result or "Blood Angels" in result

    def test_empty_wikitext(self, parser: WikiXMLParser) -> None:
        """Test handling of empty wikitext."""
        result = parser._convert_wikitext_to_markdown("")
        assert result == ""

    def test_whitespace_cleanup(self, parser: WikiXMLParser) -> None:
        """Test cleanup of excessive whitespace."""
        wikitext = "Line 1\n\n\n\nLine 2"
        result = parser._convert_wikitext_to_markdown(wikitext)
        # Should reduce to max 2 newlines
        assert "\n\n\n" not in result


class TestInternalLinkExtraction:
    """Test extraction of internal links from wikitext."""

    def test_simple_link_extraction(self, parser: WikiXMLParser) -> None:
        """Test extraction of simple [[Link]] format."""
        wikitext = "The [[Imperium]] is vast."
        links = parser.extract_internal_links(wikitext)
        assert "Imperium" in links

    def test_link_with_display_text_extraction(self, parser: WikiXMLParser) -> None:
        """Test extraction from [[Link|Display]] format (extract Link only)."""
        wikitext = "[[Blood Angels|The Angels]] are noble."
        links = parser.extract_internal_links(wikitext)
        assert "Blood Angels" in links
        assert "The Angels" not in links  # Display text should not be extracted

    def test_multiple_links_extraction(self, parser: WikiXMLParser) -> None:
        """Test extraction of multiple links."""
        wikitext = "[[Imperium]] and [[Space Marines]] and [[Chaos]]."
        links = parser.extract_internal_links(wikitext)
        assert "Imperium" in links
        assert "Space Marines" in links
        assert "Chaos" in links

    def test_link_deduplication(self, parser: WikiXMLParser) -> None:
        """Test deduplication of repeated links."""
        wikitext = "[[Imperium]] again [[Imperium]] and [[Imperium]]."
        links = parser.extract_internal_links(wikitext)
        # Should only appear once
        assert links.count("Imperium") == 1

    def test_skip_file_links(self, parser: WikiXMLParser) -> None:
        """Test that File: links are skipped."""
        wikitext = "[[File:Image.jpg]] and [[Imperium]]."
        links = parser.extract_internal_links(wikitext)
        assert "Imperium" in links
        assert not any("File:" in link for link in links)

    def test_skip_image_links(self, parser: WikiXMLParser) -> None:
        """Test that Image: links are skipped."""
        wikitext = "[[Image:Photo.png]] and [[Space Marines]]."
        links = parser.extract_internal_links(wikitext)
        assert "Space Marines" in links
        assert not any("Image:" in link for link in links)

    def test_skip_category_links(self, parser: WikiXMLParser) -> None:
        """Test that Category: links are skipped."""
        wikitext = "[[Category:Space Marines]] content [[Imperium]]."
        links = parser.extract_internal_links(wikitext)
        assert "Imperium" in links
        assert not any("Category:" in link for link in links)

    def test_empty_wikitext_extraction(self, parser: WikiXMLParser) -> None:
        """Test link extraction from empty wikitext."""
        links = parser.extract_internal_links("")
        assert links == []

    def test_no_links_in_wikitext(self, parser: WikiXMLParser) -> None:
        """Test link extraction from text without links."""
        wikitext = "Just plain text with no links."
        links = parser.extract_internal_links(wikitext)
        assert links == []
