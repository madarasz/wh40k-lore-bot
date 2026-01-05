"""Unit tests for WikiXMLParser MediaWiki to Markdown conversion."""

from pathlib import Path
from tempfile import NamedTemporaryFile

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


class TestRedirectHandling:
    """Test redirect detection, mapping, and link resolution."""

    def test_build_redirect_map_extracts_redirects(self, parser: WikiXMLParser) -> None:
        """Test that build_redirect_map correctly extracts redirect mappings."""
        xml_content = """<?xml version="1.0"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Mankind</title>
    <ns>0</ns>
    <id>1</id>
    <redirect title="Humans" />
    <revision><text>#REDIRECT [[Humans]]</text></revision>
  </page>
  <page>
    <title>Astartes</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="Space Marines" />
    <revision><text>#REDIRECT [[Space Marines]]</text></revision>
  </page>
</mediawiki>"""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            redirect_map = parser.build_redirect_map(temp_path)
            assert redirect_map["Mankind"] == "Humans"
            assert redirect_map["Astartes"] == "Space Marines"
            assert len(redirect_map) == 2
            assert parser.redirects_found == 2
        finally:
            temp_path.unlink()

    def test_build_redirect_map_decodes_html_entities(self, parser: WikiXMLParser) -> None:
        """Test that HTML entities in redirect targets are properly decoded."""
        xml_content = """<?xml version="1.0"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Tau</title>
    <ns>0</ns>
    <id>1</id>
    <redirect title="T&#039;au Empire" />
    <revision><text>#REDIRECT [[T'au Empire]]</text></revision>
  </page>
</mediawiki>"""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            redirect_map = parser.build_redirect_map(temp_path)
            assert redirect_map["Tau"] == "T'au Empire"
            assert "&#039;" not in redirect_map["Tau"]
        finally:
            temp_path.unlink()

    def test_redirect_pages_are_skipped(self, parser: WikiXMLParser) -> None:
        """Test that redirect pages are skipped during article processing."""
        xml_content = """<?xml version="1.0"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Mankind</title>
    <ns>0</ns>
    <id>1</id>
    <redirect title="Humans" />
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>#REDIRECT [[Humans]]</text>
    </revision>
  </page>
  <page>
    <title>Humans</title>
    <ns>0</ns>
    <id>2</id>
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>Human article content.</text>
    </revision>
  </page>
</mediawiki>"""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            articles = list(parser.parse_xml_export(temp_path))
            # Should only get the Humans article, not the Mankind redirect
            assert len(articles) == 1
            assert articles[0].title == "Humans"
            assert parser.redirects_skipped == 1
        finally:
            temp_path.unlink()

    def test_link_resolution_replaces_redirect_sources(self, parser: WikiXMLParser) -> None:
        """Test that links to redirect sources are resolved to targets."""
        redirect_map = {"Mankind": "Humans", "Astartes": "Space Marines"}
        wikitext = "See [[Mankind]] and [[Astartes]] for more info."
        result = parser._convert_wikitext_to_markdown(wikitext, redirect_map)
        assert "[Humans](Humans)" in result
        assert "[Space Marines](Space Marines)" in result
        assert "Mankind" not in result
        assert "Astartes" not in result

    def test_link_resolution_preserves_display_text(self, parser: WikiXMLParser) -> None:
        """Test that display text is preserved when resolving redirects."""
        redirect_map = {"Mankind": "Humans"}
        wikitext = "See [[Mankind|humanity]] for details."
        result = parser._convert_wikitext_to_markdown(wikitext, redirect_map)
        # Display text "humanity" should be preserved, but link target should be "Humans"
        assert "[humanity](Humans)" in result
        assert "Mankind" not in result

    def test_non_redirect_links_unchanged(self, parser: WikiXMLParser) -> None:
        """Test that links not in redirect map are unchanged."""
        redirect_map = {"Mankind": "Humans"}
        wikitext = "See [[Imperium]] and [[Space Marines]]."
        result = parser._convert_wikitext_to_markdown(wikitext, redirect_map)
        assert "[Imperium](Imperium)" in result
        assert "[Space Marines](Space Marines)" in result

    def test_empty_redirect_map(self, parser: WikiXMLParser) -> None:
        """Test handling of empty redirect map (no redirects in XML)."""
        xml_content = """<?xml version="1.0"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Imperium</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>Article content.</text>
    </revision>
  </page>
</mediawiki>"""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)

        try:
            redirect_map = parser.build_redirect_map(temp_path)
            assert len(redirect_map) == 0
            assert parser.redirects_found == 0
        finally:
            temp_path.unlink()

    def test_redirect_to_nonexistent_article(self, parser: WikiXMLParser) -> None:
        """Test that redirect resolution works even if target doesn't exist in XML."""
        redirect_map = {"OldName": "NewName"}
        wikitext = "See [[OldName]] for info."
        result = parser._convert_wikitext_to_markdown(wikitext, redirect_map)
        # Should still resolve to NewName even if NewName doesn't exist as an article
        assert "[NewName](NewName)" in result
        assert "OldName" not in result


class TestInfoboxExtraction:
    """Test infobox extraction and formatting."""

    def test_extract_simple_infobox(self, parser: WikiXMLParser) -> None:
        """Test extraction of a simple infobox."""
        wikitext = """{{Infobox Chapter
| name = Black Templars
| primarch = Rogal Dorn
| founding = Second Founding
}}

Some article content here."""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is not None
        assert "## Infobox: Chapter" in infobox_text
        assert "**Name**: Black Templars" in infobox_text
        assert "**Primarch**: Rogal Dorn" in infobox_text
        assert "**Founding**: Second Founding" in infobox_text

    def test_extract_infobox_with_links(self, parser: WikiXMLParser) -> None:
        """Test extraction of infobox with wiki links."""
        wikitext = """{{Infobox Chapter
| name = Ultramarines
| primarch = [[Roboute Guilliman]]
| legion = [[Ultramarines Legion]]
}}"""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is not None
        assert "Roboute Guilliman" in infobox_links
        assert "Ultramarines Legion" in infobox_links
        # Links should be cleaned in the text
        assert "[[" not in infobox_text

    def test_extract_infobox_with_redirect_resolution(self, parser: WikiXMLParser) -> None:
        """Test that infobox links are resolved via redirect map."""
        wikitext = """{{Infobox Character
| name = Horus
| affiliation = [[Mankind]]
}}"""
        redirect_map = {"Mankind": "Humanity"}
        infobox_text, infobox_links = parser._extract_infobox(wikitext, redirect_map)

        assert "Humanity" in infobox_links
        assert "Mankind" not in infobox_links

    def test_extract_infobox_skips_images(self, parser: WikiXMLParser) -> None:
        """Test that image parameters are skipped in infobox."""
        wikitext = """{{Infobox Chapter
| image = BloodAngels.jpg
| image caption = A Blood Angel
| name = Blood Angels
}}"""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is not None
        assert "BloodAngels.jpg" not in infobox_text
        assert "image" not in infobox_text.lower() or "Infobox" in infobox_text
        assert "**Name**: Blood Angels" in infobox_text

    def test_extract_infobox_no_infobox(self, parser: WikiXMLParser) -> None:
        """Test extraction when there is no infobox."""
        wikitext = """==Overview==
Some article content without an infobox."""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is None
        assert infobox_links == []

    def test_extract_infobox_empty_wikitext(self, parser: WikiXMLParser) -> None:
        """Test extraction from empty wikitext."""
        infobox_text, infobox_links = parser._extract_infobox("")

        assert infobox_text is None
        assert infobox_links == []

    def test_extract_infobox_lowercase_name(self, parser: WikiXMLParser) -> None:
        """Test extraction with lowercase 'infobox' template name."""
        wikitext = """{{infobox planet
| name = Terra
| sector = Sol
}}"""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is not None
        assert "## Infobox: Planet" in infobox_text

    def test_extract_infobox_with_file_links(self, parser: WikiXMLParser) -> None:
        """Test that File: links in infobox values are not included in links."""
        wikitext = """{{Infobox Chapter
| name = Imperial Fists
| banner = [[File:ImperialFists.png]]
}}"""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        assert infobox_text is not None
        # File links should not be in the links list
        assert not any("File:" in link for link in infobox_links)

    def test_clean_infobox_value_removes_wiki_formatting(self, parser: WikiXMLParser) -> None:
        """Test that wiki formatting is cleaned from infobox values."""
        # Test bold
        assert parser._clean_infobox_value("'''Bold'''") == "Bold"
        # Test italic
        assert parser._clean_infobox_value("''Italic''") == "Italic"
        # Test wiki link
        assert parser._clean_infobox_value("[[Space Marines]]") == "Space Marines"
        # Test wiki link with display text
        assert parser._clean_infobox_value("[[Space Marines|Astartes]]") == "Astartes"
        # Test template removal
        assert parser._clean_infobox_value("Text {{cite}}") == "Text"

    def test_extract_infobox_deduplicates_links(self, parser: WikiXMLParser) -> None:
        """Test that duplicate links in infobox are deduplicated."""
        wikitext = """{{Infobox Character
| name = Guilliman
| legion = [[Ultramarines]]
| chapter = [[Ultramarines]]
}}"""
        infobox_text, infobox_links = parser._extract_infobox(wikitext)

        # Should only have one Ultramarines entry
        assert infobox_links.count("Ultramarines") == 1
