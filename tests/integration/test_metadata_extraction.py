"""Integration tests for metadata extraction with real wiki chunks."""

from pathlib import Path

import pytest

from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import Chunk


class TestMetadataExtractionWithRealData:
    """Test metadata extraction with real wiki article chunks."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create a metadata extractor instance."""
        return MetadataExtractor()

    @pytest.fixture
    def sample_space_marine_chunk(self) -> Chunk:
        """Create a sample chunk from a Space Marine article."""
        return Chunk(
            chunk_text="""
            During the Great Crusade, [[Roboute Guilliman]] led the Ultramarines
            Space Marines chapter in numerous battles across the galaxy. The Ultramarines
            employed advanced power armor and bolter weapons. As detailed in Codex: Ultramarines,
            their tactical doctrine became the gold standard for all Space Marines chapters.
            The campaign against the Orks was particularly brutal.
            """,
            article_title="Roboute Guilliman",
            section_path="History > Great Crusade",
            chunk_index=0,
        )

    @pytest.fixture
    def sample_horus_heresy_chunk(self) -> Chunk:
        """Create a sample chunk from Horus Heresy era."""
        return Chunk(
            chunk_text="""
            The Horus Heresy was the greatest betrayal in the history of the Imperium.
            [[Horus]], Warmaster of the Emperor's armies and primarch of the Luna Wolves,
            turned against the Emperor of Mankind. The traitor legions, including the
            [[World Eaters]] and [[Death Guard]], fought against the loyalist Space Marines.
            This conflict is detailed in Horus Heresy: Book 1: Betrayal.
            """,
            article_title="Horus Heresy",
            section_path="Overview",
            chunk_index=0,
        )

    @pytest.fixture
    def sample_technology_chunk(self) -> Chunk:
        """Create a sample chunk focused on technology."""
        return Chunk(
            chunk_text="""
            The bolter is the standard weapon of the Adeptus Astartes. This mass-reactive
            firearm fires .75 caliber rounds. Space Marine armor, also known as power armour,
            provides exceptional protection. The vehicle pool includes Land Raiders and
            Rhino transports. All this equipment represents the peak of Imperial technology.
            More details can be found in Codex: Space Marines.
            """,
            article_title="Space Marine Equipment",
            section_path="Weapons and Armor",
            chunk_index=0,
        )

    def test_extract_metadata_from_space_marine_chunk(
        self,
        extractor: MetadataExtractor,
        sample_space_marine_chunk: Chunk,
    ) -> None:
        """Test metadata extraction from a Space Marines chunk."""
        metadata = extractor.extract_metadata(sample_space_marine_chunk)

        # Verify character extraction
        assert "Roboute Guilliman" in metadata["character_names"]

        # Verify source book extraction
        assert "Ultramarines" in metadata["source_books"]

    def test_extract_metadata_from_heresy_chunk(
        self,
        extractor: MetadataExtractor,
        sample_horus_heresy_chunk: Chunk,
    ) -> None:
        """Test metadata extraction from a Horus Heresy chunk."""
        metadata = extractor.extract_metadata(sample_horus_heresy_chunk)

        # Verify character extraction
        assert "Horus" in metadata["character_names"]

        # Verify source book extraction
        assert any("Betrayal" in book for book in metadata["source_books"])

    def test_extract_metadata_from_technology_chunk(
        self,
        extractor: MetadataExtractor,
        sample_technology_chunk: Chunk,
    ) -> None:
        """Test metadata extraction from a technology-focused chunk."""
        metadata = extractor.extract_metadata(sample_technology_chunk)

        # Verify source book extraction
        assert "Space Marines" in metadata["source_books"]

    def test_extract_metadata_statistics(
        self,
        extractor: MetadataExtractor,
    ) -> None:
        """Test metadata extraction on multiple chunks and verify statistics."""
        # Create multiple test chunks with different characteristics
        chunks = [
            Chunk(
                chunk_text="The [[Ultramarines]] Space Marines fought bravely.",
                article_title="Test1",
                section_path="Test",
                chunk_index=0,
            ),
            Chunk(
                chunk_text="The [[Tyranids]] invaded the sector with [[Hive Fleet Leviathan]].",
                article_title="Test2",
                section_path="Test",
                chunk_index=0,
            ),
            Chunk(
                chunk_text="During the crusade, many worlds were liberated.",
                article_title="Test3",
                section_path="Test",
                chunk_index=0,
            ),
        ]

        # Extract metadata for all chunks
        metadata_results = [extractor.extract_metadata(chunk) for chunk in chunks]

        # Verify all extractions succeeded
        assert len(metadata_results) == 3

        # Verify character extraction worked for chunks with wiki links
        chars_detected = sum(len(m["character_names"]) for m in metadata_results)
        assert chars_detected >= 3  # Ultramarines, Tyranids, Hive Fleet Leviathan

    @pytest.mark.skipif(
        not Path("data/markdown-archive").exists(),
        reason="Markdown archive not available",
    )
    def test_extract_from_real_wiki_file(
        self,
        extractor: MetadataExtractor,
    ) -> None:
        """Test metadata extraction from a real wiki markdown file.

        This test uses actual wiki data if available.
        """
        # Try to read a real wiki file
        archive_path = Path("data/markdown-archive")
        wiki_files = list(archive_path.glob("*.md"))

        if not wiki_files:
            pytest.skip("No wiki files available in markdown archive")

        # Use the first available file
        test_file = wiki_files[0]
        content = test_file.read_text(encoding="utf-8")

        # Extract just the article content (skip frontmatter)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2]

        # Create a chunk from the first 1000 characters
        test_chunk = Chunk(
            chunk_text=content[:1000],
            article_title=test_file.stem,
            section_path="Test",
            chunk_index=0,
        )

        # Extract metadata
        metadata = extractor.extract_metadata(test_chunk)

        # Verify basic structure
        assert "character_names" in metadata
        assert "source_books" in metadata

        # Log results for manual verification
        print(f"\nMetadata extracted from {test_file.name}:")
        print(f"  Character Names: {metadata['character_names']}")
        print(f"  Source Books: {metadata['source_books']}")

    @pytest.mark.skipif(
        not Path("data/markdown-archive").exists(),
        reason="Markdown archive not available",
    )
    def test_extract_from_multiple_real_files(
        self,
        extractor: MetadataExtractor,
    ) -> None:
        """Test metadata extraction from multiple real wiki files."""
        archive_path = Path("data/markdown-archive")
        wiki_files = list(archive_path.glob("*.md"))[:5]  # Test first 5 files

        if not wiki_files:
            pytest.skip("No wiki files available in markdown archive")

        results = []

        for wiki_file in wiki_files:
            content = wiki_file.read_text(encoding="utf-8")

            # Extract article content (skip frontmatter)
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2]

            # Create chunk from first portion
            chunk = Chunk(
                chunk_text=content[:1500],
                article_title=wiki_file.stem,
                section_path="Test",
                chunk_index=0,
            )

            metadata = extractor.extract_metadata(chunk)
            results.append((wiki_file.name, metadata))

        # Verify we got results for all files
        assert len(results) == len(wiki_files)

        # Calculate statistics
        total_characters = sum(len(m["character_names"]) for _, m in results)
        total_books = sum(len(m["source_books"]) for _, m in results)

        print(f"\nMetadata extraction statistics ({len(wiki_files)} files):")
        print(f"  Total character names extracted: {total_characters}")
        print(f"  Total source books found: {total_books}")
