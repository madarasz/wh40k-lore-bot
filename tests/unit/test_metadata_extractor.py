"""Unit tests for metadata extraction."""

import pytest

from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import Chunk


class TestCharacterNameExtraction:
    """Test character name extraction functionality."""

    def test_extract_single_character(self) -> None:
        """Test extraction of a single character name from wiki link."""
        extractor = MetadataExtractor()
        text = "[[Roboute Guilliman]] is the Primarch of the Ultramarines."
        names = extractor._extract_character_names(text)
        assert names == ["Roboute Guilliman"]

    def test_extract_multiple_characters(self) -> None:
        """Test extraction of multiple character names."""
        extractor = MetadataExtractor()
        text = """
        [[Roboute Guilliman]] met with [[Lion El'Jonson]] and [[Sanguinius]]
        to discuss the fate of [[Horus]].
        """
        names = extractor._extract_character_names(text)
        assert len(names) == 4
        assert "Roboute Guilliman" in names
        assert "Sanguinius" in names

    def test_extract_with_display_text(self) -> None:
        """Test extraction from wiki links with display text."""
        extractor = MetadataExtractor()
        text = "[[Roboute Guilliman|the Avenging Son]] returned to lead the Imperium."
        names = extractor._extract_character_names(text)
        assert names == ["Roboute Guilliman"]

    def test_extract_top_5_most_mentioned(self) -> None:
        """Test that only top 5 most mentioned names are returned."""
        extractor = MetadataExtractor()
        text = """
        [[Name1]] [[Name1]] [[Name1]] [[Name1]] [[Name1]]
        [[Name2]] [[Name2]] [[Name2]] [[Name2]]
        [[Name3]] [[Name3]] [[Name3]]
        [[Name4]] [[Name4]]
        [[Name5]]
        [[Name6]]
        [[Name7]]
        """
        names = extractor._extract_character_names(text)
        assert len(names) == 5
        assert "Name1" in names
        assert "Name2" in names
        assert "Name3" in names
        assert "Name4" in names
        assert "Name5" in names

    def test_no_characters_extracted(self) -> None:
        """Test that empty list is returned when no wiki links found."""
        extractor = MetadataExtractor()
        text = "This text has no wiki links."
        names = extractor._extract_character_names(text)
        assert names == []


class TestSourceBookExtraction:
    """Test source book extraction functionality."""

    def test_extract_codex_reference(self) -> None:
        """Test extraction of Codex reference."""
        extractor = MetadataExtractor()
        text = "As detailed in Codex: Space Marines, the chapter follows strict protocols."
        books = extractor._extract_source_books(text)
        assert "Space Marines" in books

    def test_extract_horus_heresy_book(self) -> None:
        """Test extraction of Horus Heresy book reference."""
        extractor = MetadataExtractor()
        text = "Described in Horus Heresy: Book 1: Betrayal, the events unfolded."
        books = extractor._extract_source_books(text)
        assert any("Betrayal" in book for book in books)

    def test_extract_white_dwarf(self) -> None:
        """Test extraction of White Dwarf reference."""
        extractor = MetadataExtractor()
        text = "Originally published in White Dwarf 450."
        books = extractor._extract_source_books(text)
        assert "450" in books

    def test_extract_multiple_sources(self) -> None:
        """Test extraction of multiple source books."""
        extractor = MetadataExtractor()
        text = """
        Codex: Space Marines and Index: Imperium provide details.
        White Dwarf 450 expanded on this.
        """
        books = extractor._extract_source_books(text)
        assert len(books) >= 2

    def test_no_source_books_found(self) -> None:
        """Test that empty list is returned when no sources found."""
        extractor = MetadataExtractor()
        text = "This text has no source book references."
        books = extractor._extract_source_books(text)
        assert books == []


class TestExtractMetadata:
    """Test main extract_metadata method."""

    def test_extract_complete_metadata(self) -> None:
        """Test extraction of complete metadata from a realistic chunk."""
        extractor = MetadataExtractor()
        chunk = Chunk(
            chunk_text="""
            During the Great Crusade, [[Roboute Guilliman]] led the Ultramarines
            in numerous battles against xenos threats. The Space Marines employed
            advanced weapons and power armor. As detailed in Codex: Ultramarines,
            their tactics were unmatched.
            """,
            article_title="Roboute Guilliman",
            section_path="History > Great Crusade",
            chunk_index=0,
        )

        metadata = extractor.extract_metadata(chunk)

        assert "Roboute Guilliman" in metadata["character_names"]
        assert "Ultramarines" in metadata["source_books"]

    def test_extract_metadata_all_fields_present(self) -> None:
        """Test that all metadata fields are present in output."""
        extractor = MetadataExtractor()
        chunk = Chunk(
            chunk_text="Simple test text.",
            article_title="Test",
            section_path="Test",
            chunk_index=0,
        )

        metadata = extractor.extract_metadata(chunk)

        assert "character_names" in metadata
        assert "source_books" in metadata

    def test_extract_metadata_invalid_chunk(self) -> None:
        """Test that ValueError is raised for invalid chunk."""
        extractor = MetadataExtractor()

        with pytest.raises(ValueError):
            extractor.extract_metadata(None)  # type: ignore

    def test_extract_metadata_empty_text(self) -> None:
        """Test that ValueError is raised for empty chunk text."""
        extractor = MetadataExtractor()
        chunk = Chunk(
            chunk_text="",
            article_title="Test",
            section_path="Test",
            chunk_index=0,
        )

        with pytest.raises(ValueError):
            extractor.extract_metadata(chunk)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_handling(self) -> None:
        """Test that empty strings are handled gracefully."""
        extractor = MetadataExtractor()
        assert extractor._extract_character_names("") == []
        assert extractor._extract_source_books("") == []

    def test_special_characters_handling(self) -> None:
        """Test that special characters don't break extraction."""
        extractor = MetadataExtractor()
        text = "The [[T'au]] and [[O'Shovah]] fought with advanced tech."
        names = extractor._extract_character_names(text)
        assert "T'au" in names or "O'Shovah" in names

    def test_very_long_text(self) -> None:
        """Test that very long text is processed correctly."""
        extractor = MetadataExtractor()
        text = "[[Character]] " * 1000
        names = extractor._extract_character_names(text)
        assert names == ["Character"]
