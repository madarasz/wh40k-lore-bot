"""Unit tests for metadata extraction."""

import pytest

from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import Chunk


class TestFactionDetection:
    """Test faction detection functionality."""

    def test_detect_single_faction(self) -> None:
        """Test detection of a single faction."""
        extractor = MetadataExtractor()
        text = (
            "The Ultramarines are a Space Marines chapter known for "
            "their adherence to the Codex Astartes."
        )
        faction = extractor._detect_faction(text)
        assert faction == "Space Marines"

    def test_detect_most_frequent_faction(self) -> None:
        """Test that most frequent faction is returned."""
        extractor = MetadataExtractor()
        text = """
        The Tyranids invaded the world. The Imperial Guard defended bravely.
        The Tyranids overwhelmed them. More Tyranids arrived as reinforcements.
        The Tyranids consumed everything.
        """
        faction = extractor._detect_faction(text)
        assert faction == "Tyranids"

    def test_faction_case_insensitive(self) -> None:
        """Test that faction detection is case-insensitive."""
        extractor = MetadataExtractor()
        text = "The SPACE MARINES fought the orks in brutal combat."
        faction = extractor._detect_faction(text)
        assert faction == "Space Marines"

    def test_faction_aliases(self) -> None:
        """Test that faction aliases are normalized."""
        extractor = MetadataExtractor()
        text = "The Astra Militarum deployed their forces."
        faction = extractor._detect_faction(text)
        assert faction == "Imperial Guard"

    def test_no_faction_detected(self) -> None:
        """Test that None is returned when no faction found."""
        extractor = MetadataExtractor()
        text = "This is generic text with no faction mentions."
        faction = extractor._detect_faction(text)
        assert faction is None

    def test_no_substring_collision(self) -> None:
        """Test that shorter faction names inside longer ones are not double-counted.

        For example, "Eldar" should not match inside "Craftworld Eldar" or "Dark Eldar".
        The function should use word-boundary matching to prevent substring collisions.
        """
        extractor = MetadataExtractor()
        text = """
        The Craftworld Eldar are known for their psychic powers.
        The Dark Eldar raid from Commorragh.
        The Craftworld Eldar have farseers.
        """
        faction = extractor._detect_faction(text)
        # "Craftworld Eldar" appears 2 times, "Dark Eldar" appears 1 time
        # "Eldar" should NOT be counted separately (3 times) as it's part of longer names
        # Result should be "Eldar" (normalized from "Craftworld Eldar" and "Dark Eldar")
        assert faction == "Eldar"

    def test_word_boundary_matching(self) -> None:
        """Test that factions are matched with word boundaries.

        Ensures that faction names only match as complete words, not as substrings.
        """
        extractor = MetadataExtractor()
        # "Tau" should only match the standalone word, not in "Centaur" or similar
        text = """
        The Tau Empire expanded their borders.
        The Tau are known for their greater good philosophy.
        The centaur-like creatures were not involved.
        """
        faction = extractor._detect_faction(text)
        assert faction == "Tau"


class TestEraDetection:
    """Test era detection functionality."""

    def test_detect_single_era(self) -> None:
        """Test detection of a single era."""
        extractor = MetadataExtractor()
        text = "During the Great Crusade, the Emperor led his armies across the galaxy."
        eras = extractor._detect_eras(text)
        assert eras == ["Great Crusade"]

    def test_detect_multiple_eras(self) -> None:
        """Test detection of multiple eras."""
        extractor = MetadataExtractor()
        text = """
        The Great Crusade united humanity, but the Horus Heresy tore it apart.
        The Indomitus Crusade seeks to reclaim what was lost.
        """
        eras = extractor._detect_eras(text)
        assert "Great Crusade" in eras
        assert "Horus Heresy" in eras
        assert "Indomitus Crusade" in eras

    def test_eras_deduplicated(self) -> None:
        """Test that duplicate eras are removed."""
        extractor = MetadataExtractor()
        text = "The Horus Heresy began. The Horus Heresy raged. The Horus Heresy ended."
        eras = extractor._detect_eras(text)
        assert eras == ["Horus Heresy"]

    def test_no_eras_detected(self) -> None:
        """Test that empty list is returned when no eras found."""
        extractor = MetadataExtractor()
        text = "This text has no era mentions."
        eras = extractor._detect_eras(text)
        assert eras == []


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


class TestContentTypeClassification:
    """Test content type classification functionality."""

    def test_classify_as_character(self) -> None:
        """Test classification as 'character' with high name density."""
        extractor = MetadataExtractor()
        text = "Generic text"
        character_names = ["Name1", "Name2", "Name3"]
        content_type = extractor._classify_content_type(text, character_names)
        assert content_type == "character"

    def test_classify_as_military(self) -> None:
        """Test classification as 'military' based on keywords."""
        extractor = MetadataExtractor()
        text = """
        The battle raged for days. War consumed the planet.
        Military tactics and strategy determined the campaign outcome.
        The siege was brutal.
        """
        character_names = []
        content_type = extractor._classify_content_type(text, character_names)
        assert content_type == "military"

    def test_classify_as_technology(self) -> None:
        """Test classification as 'technology' based on keywords."""
        extractor = MetadataExtractor()
        text = """
        The weapon was a marvel of technology. Power armor protected the warriors.
        Bolters and plasma weapons were standard equipment.
        The vehicle carried advanced tech.
        """
        character_names = []
        content_type = extractor._classify_content_type(text, character_names)
        assert content_type == "technology"

    def test_classify_as_lore_default(self) -> None:
        """Test classification defaults to 'lore' when no clear type."""
        extractor = MetadataExtractor()
        text = "This is general lore text without specific keywords."
        character_names = []
        content_type = extractor._classify_content_type(text, character_names)
        assert content_type == "lore"

    def test_character_type_overrides_keywords(self) -> None:
        """Test that character type takes precedence over keyword matching."""
        extractor = MetadataExtractor()
        text = "Battle and war and tactics and weapons and armor."
        character_names = ["Name1", "Name2", "Name3", "Name4"]
        content_type = extractor._classify_content_type(text, character_names)
        assert content_type == "character"


class TestSpoilerDetection:
    """Test spoiler detection functionality."""

    def test_detect_spoiler_keyword(self) -> None:
        """Test detection of spoiler keyword."""
        extractor = MetadataExtractor()
        text = "Warning: spoiler ahead for recent lore developments."
        spoiler = extractor._detect_spoiler(text)
        assert spoiler is True

    def test_detect_edition_spoiler(self) -> None:
        """Test detection of edition-based spoilers."""
        extractor = MetadataExtractor()
        text = "In 10th edition, new developments occurred."
        spoiler = extractor._detect_spoiler(text)
        assert spoiler is True

    def test_no_spoiler_detected(self) -> None:
        """Test that False is returned when no spoilers found."""
        extractor = MetadataExtractor()
        text = "This is standard canon lore from established sources."
        spoiler = extractor._detect_spoiler(text)
        assert spoiler is False


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

        # Ultramarines is mentioned twice, Space Marines once, so Ultramarines is detected
        assert metadata["faction"] == "Ultramarines"
        assert "Great Crusade" in metadata["eras"]
        assert "Roboute Guilliman" in metadata["character_names"]
        assert metadata["content_type"] in ["military", "technology", "lore"]
        assert metadata["spoiler_flag"] is False
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

        assert "faction" in metadata
        assert "eras" in metadata
        assert "character_names" in metadata
        assert "content_type" in metadata
        assert "spoiler_flag" in metadata
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
        assert extractor._detect_faction("") is None
        assert extractor._detect_eras("") == []
        assert extractor._extract_character_names("") == []
        assert extractor._classify_content_type("", []) == "lore"
        assert extractor._detect_spoiler("") is False
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
        text = "Space Marines " * 1000
        faction = extractor._detect_faction(text)
        assert faction == "Space Marines"
