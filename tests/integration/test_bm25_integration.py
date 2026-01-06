"""Integration tests for BM25Repository with realistic wiki content."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.ingestion.models import Chunk
from src.repositories.bm25_repository import BM25Repository


def build_chunk_lookup(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Build lookup map from chunk_id to Chunk for testing."""
    lookup = {}
    for chunk in chunks:
        # Use same ID generation logic as BM25Repository
        if chunk.wiki_page_id is not None:
            chunk_id = f"{chunk.wiki_page_id}_{chunk.chunk_index}"
        else:
            chunk_id = f"{chunk.article_title}_{chunk.chunk_index}"
        lookup[chunk_id] = chunk
    return lookup


@pytest.fixture
def realistic_chunks() -> list[Chunk]:
    """Create realistic WH40K wiki chunks for integration testing.

    Returns 50+ chunks covering various WH40K topics with proper nouns,
    faction names, character names, and event names.
    """
    chunks: list[Chunk] = []

    # Roboute Guilliman chunks
    guilliman_chunks = [
        Chunk(
            chunk_text="Roboute Guilliman is the Primarch of the Ultramarines Space Marine "
            "Chapter and the Lord Commander of the Imperium. He is one of the twenty "
            "Primarchs created by the Emperor before the Great Crusade.",
            article_title="Roboute Guilliman",
            section_path="Overview",
            chunk_index=0,
            links=["Ultramarines", "Primarch", "Emperor"],
        ),
        Chunk(
            chunk_text="During the Horus Heresy, Guilliman remained loyal to the Emperor and "
            "fought against the traitor forces. He was gravely wounded by his brother "
            "Fulgrim and placed in stasis for ten thousand years.",
            article_title="Roboute Guilliman",
            section_path="History > Horus Heresy",
            chunk_index=1,
            links=["Horus Heresy", "Fulgrim"],
        ),
        Chunk(
            chunk_text="Guilliman awakened in the 41st Millennium and became the Lord Commander "
            "of the Imperium, leading humanity in the Era Indomitus. He launched the "
            "Indomitus Crusade to reunite the Imperium.",
            article_title="Roboute Guilliman",
            section_path="History > Return",
            chunk_index=2,
            links=["Indomitus Crusade"],
        ),
    ]
    chunks.extend(guilliman_chunks)

    # Ultramarines chunks
    ultramarines_chunks = [
        Chunk(
            chunk_text="The Ultramarines are a First Founding Space Marine Chapter and "
            "the largest loyalist chapter. They were created from the gene-seed of "
            "Roboute Guilliman and follow the Codex Astartes closely.",
            article_title="Ultramarines",
            section_path="Overview",
            chunk_index=0,
            links=["Roboute Guilliman", "Codex Astartes"],
        ),
        Chunk(
            chunk_text="The Ultramarines homeworld is Macragge, a fortress world in the "
            "Ultramar sector. Macragge was devastated during the First Tyrannic War "
            "when Hive Fleet Behemoth attacked.",
            article_title="Ultramarines",
            section_path="Homeworld",
            chunk_index=1,
            links=["Macragge", "Hive Fleet Behemoth"],
        ),
        Chunk(
            chunk_text="Chapter Master Marneus Calgar leads the Ultramarines and is one of "
            "the greatest warriors in the Imperium. He fought at the Battle of Macragge "
            "and was gravely wounded by a Carnifex.",
            article_title="Ultramarines",
            section_path="Leadership",
            chunk_index=2,
            links=["Marneus Calgar", "Battle of Macragge"],
        ),
    ]
    chunks.extend(ultramarines_chunks)

    # Horus Heresy chunks
    horus_chunks = [
        Chunk(
            chunk_text="The Horus Heresy was a galaxy-spanning civil war that nearly destroyed "
            "the Imperium of Man. It began when Horus, the Warmaster, was corrupted by "
            "Chaos and turned against the Emperor.",
            article_title="Horus Heresy",
            section_path="Overview",
            chunk_index=0,
            links=["Horus", "Emperor", "Chaos"],
        ),
        Chunk(
            chunk_text="Horus was the most favored son of the Emperor and was appointed Warmaster "
            "to lead the Great Crusade. His fall to Chaos was orchestrated by the Chaos "
            "Gods through manipulation and false visions.",
            article_title="Horus Heresy",
            section_path="Fall of Horus",
            chunk_index=1,
            links=["Horus", "Chaos Gods"],
        ),
        Chunk(
            chunk_text="The Heresy culminated in the Siege of Terra, where Horus led his traitor "
            "forces against the Imperial Palace. The Emperor fought Horus aboard his "
            "flagship and defeated him, but was mortally wounded.",
            article_title="Horus Heresy",
            section_path="Siege of Terra",
            chunk_index=2,
            links=["Siege of Terra", "Imperial Palace"],
        ),
    ]
    chunks.extend(horus_chunks)

    # Additional faction chunks
    additional_chunks = [
        # Blood Angels
        Chunk(
            chunk_text="The Blood Angels are a First Founding Space Marine Chapter descended from "
            "the Primarch Sanguinius. They suffer from the Red Thirst and Black Rage, "
            "genetic flaws inherited from their Primarch's death.",
            article_title="Blood Angels",
            section_path="Overview",
            chunk_index=0,
            links=["Sanguinius", "Red Thirst", "Black Rage"],
        ),
        # Dark Angels
        Chunk(
            chunk_text="The Dark Angels were the First Legion and are led by Supreme Grand Master "
            "Azrael. They hunt the Fallen, traitor Dark Angels from the Horus Heresy "
            "who turned to Chaos.",
            article_title="Dark Angels",
            section_path="Overview",
            chunk_index=0,
            links=["Azrael", "The Fallen"],
        ),
        # Space Wolves
        Chunk(
            chunk_text="The Space Wolves are the VI Legion founded by Primarch Leman Russ. "
            "They are known for their fierce nature and use of wolf-themed iconography. "
            "Their homeworld is the frozen death world of Fenris.",
            article_title="Space Wolves",
            section_path="Overview",
            chunk_index=0,
            links=["Leman Russ", "Fenris"],
        ),
        # Imperial Fists
        Chunk(
            chunk_text="The Imperial Fists are masters of siege warfare and defensive tactics. "
            "Founded by Primarch Rogal Dorn, they defended the Imperial Palace during "
            "the Siege of Terra.",
            article_title="Imperial Fists",
            section_path="Overview",
            chunk_index=0,
            links=["Rogal Dorn", "Siege of Terra"],
        ),
        # Tyranids
        Chunk(
            chunk_text="The Tyranids are an extragalactic alien race that consumes all biomass. "
            "Hive Fleet Behemoth was the first major Tyranid incursion, attacking "
            "Ultramar and nearly destroying Macragge.",
            article_title="Tyranids",
            section_path="Hive Fleets",
            chunk_index=0,
            links=["Hive Fleet Behemoth", "Macragge"],
        ),
        Chunk(
            chunk_text="Hive Fleet Leviathan is the largest Tyranid hive fleet and poses an "
            "existential threat to the Imperium. It approaches from below the galactic "
            "plane and has consumed hundreds of worlds.",
            article_title="Tyranids",
            section_path="Hive Fleet Leviathan",
            chunk_index=1,
            links=["Hive Fleet Leviathan"],
        ),
        # Chaos chunks
        Chunk(
            chunk_text="The Chaos Gods are malevolent entities from the Warp: Khorne the Blood "
            "God, Nurgle the Plague Lord, Tzeentch the Changer of Ways, and Slaanesh "
            "the Prince of Pleasure.",
            article_title="Chaos Gods",
            section_path="The Four Powers",
            chunk_index=0,
            links=["Khorne", "Nurgle", "Tzeentch", "Slaanesh"],
        ),
        # Necrons
        Chunk(
            chunk_text="The Necrons are an ancient race that transferred their consciousness into "
            "immortal metal bodies. They sleep in tomb worlds across the galaxy and are "
            "awakening to reclaim their empire.",
            article_title="Necrons",
            section_path="Overview",
            chunk_index=0,
            links=["Tomb Worlds"],
        ),
    ]
    chunks.extend(additional_chunks)

    # Add more generic lore chunks to reach 50+
    for i in range(36):
        chunks.append(
            Chunk(
                chunk_text=f"The Imperium of Man spans a million worlds across the galaxy. "
                f"World {i + 1} is defended by the Imperial Guard and Adeptus Astartes. "
                f"The Emperor protects humanity from xenos threats and Chaos corruption.",
                article_title=f"Imperial World {i + 1}",
                section_path="Defense",
                chunk_index=0,
                links=["Imperium", "Imperial Guard", "Adeptus Astartes"],
            )
        )

    return chunks


@pytest.fixture
def repository() -> BM25Repository:
    """Create a BM25Repository instance for testing."""
    return BM25Repository(tokenize_lowercase=True)


class TestBM25Integration:
    """Integration tests for BM25Repository with realistic data."""

    def test_build_index_with_realistic_chunks(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test building index with 50+ realistic WH40K chunks."""
        assert len(realistic_chunks) >= 50

        repository.build_index(realistic_chunks)

        assert repository.is_index_built()
        assert len(repository.chunk_ids) >= 50
        assert repository.bm25 is not None

        # Verify index stats
        stats = repository.get_index_stats()
        assert stats["chunk_count"] >= 50

    def test_search_proper_nouns_guilliman(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test search for 'Guilliman' ranks exact matches highest."""
        repository.build_index(realistic_chunks)
        chunk_lookup = build_chunk_lookup(realistic_chunks)

        results = repository.search("Guilliman", top_k=10)

        # Should return results
        assert len(results) > 0

        # Top results should be from Guilliman or Ultramarines articles
        top_5_titles = [chunk_lookup[chunk_id].article_title for chunk_id, _ in results[:5]]
        assert any("Guilliman" in title or "Ultramarines" in title for title in top_5_titles)

        # Scores should be descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

        # Top score should be significantly higher than average
        top_score = scores[0]
        avg_score = sum(scores) / len(scores)
        assert top_score > avg_score * 2

    def test_search_proper_nouns_ultramarines(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test search for 'Ultramarines' finds relevant chunks."""
        repository.build_index(realistic_chunks)
        chunk_lookup = build_chunk_lookup(realistic_chunks)

        results = repository.search("Ultramarines", top_k=10)

        # Should return results
        assert len(results) > 0

        # Top results should mention Ultramarines
        top_texts = [chunk_lookup[chunk_id].chunk_text.lower() for chunk_id, _ in results[:5]]
        assert any("ultramarines" in text for text in top_texts)

    def test_search_multi_word_query(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test multi-word query 'Horus Heresy' ranks appropriately."""
        repository.build_index(realistic_chunks)
        chunk_lookup = build_chunk_lookup(realistic_chunks)

        results = repository.search("Horus Heresy", top_k=10)

        # Should return results
        assert len(results) > 0

        # Top results should contain both keywords or be from Horus Heresy article
        top_5_results = results[:5]
        relevant_count = sum(
            1
            for chunk_id, _ in top_5_results
            if "horus" in chunk_lookup[chunk_id].chunk_text.lower()
            or "Horus" in chunk_lookup[chunk_id].article_title
        )
        assert relevant_count >= 3

    def test_exact_match_beats_partial_match(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Verify BM25 ranks exact keyword matches higher than partial matches."""
        repository.build_index(realistic_chunks)
        chunk_lookup = build_chunk_lookup(realistic_chunks)

        # Search for distinctive proper noun
        results = repository.search("Macragge", top_k=20)

        # Find chunks that contain exact "Macragge" vs just "Ultramarines"
        exact_matches = [
            (chunk_id, score)
            for chunk_id, score in results
            if "macragge" in chunk_lookup[chunk_id].chunk_text.lower()
        ]
        partial_matches = [
            (chunk_id, score)
            for chunk_id, score in results
            if "macragge" not in chunk_lookup[chunk_id].chunk_text.lower()
            and "ultramarines" in chunk_lookup[chunk_id].chunk_text.lower()
        ]

        # Verify we have exact matches with positive scores
        assert exact_matches, "Should find chunks containing 'Macragge'"
        exact_scores = [score for _, score in exact_matches]
        assert any(score > 0 for score in exact_scores), "Exact matches should have positive scores"

        # Compare average scores: exact matches should score higher on average
        if exact_matches and partial_matches:
            avg_exact_score = sum(score for _, score in exact_matches) / len(exact_matches)
            avg_partial_score = sum(score for _, score in partial_matches) / len(partial_matches)
            assert avg_exact_score > avg_partial_score

    def test_persistence_roundtrip(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test save, load, and search consistency with realistic data."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25_integration_test.pkl"

            # Build index
            repository.build_index(realistic_chunks)
            original_stats = repository.get_index_stats()

            # Perform search before save
            query = "Guilliman Ultramarines"
            original_results = repository.search(query, top_k=10)

            # Save index
            repository.save_index(index_path)
            assert index_path.exists()

            # Load into new repository
            new_repo = BM25Repository()
            new_repo.load_index(index_path)

            # Verify stats match
            loaded_stats = new_repo.get_index_stats()
            assert loaded_stats["chunk_count"] == original_stats["chunk_count"]

            # Verify search results match
            loaded_results = new_repo.search(query, top_k=10)
            assert len(loaded_results) == len(original_results)

            # Chunk IDs and scores should be identical
            for (orig_chunk_id, orig_score), (load_chunk_id, load_score) in zip(
                original_results, loaded_results, strict=True
            ):
                assert orig_chunk_id == load_chunk_id
                assert orig_score == load_score

    def test_large_scale_search_performance(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test search performance with 50+ chunks."""
        repository.build_index(realistic_chunks)

        # Perform multiple searches
        queries = [
            "Guilliman",
            "Horus Heresy",
            "Ultramarines Macragge",
            "Tyranids Hive Fleet",
            "Chaos Gods",
            "Space Marine Chapter",
        ]

        for query in queries:
            results = repository.search(query, top_k=20)
            # All searches should complete and return results
            assert len(results) > 0
            # Results should be properly scored
            scores = [score for _, score in results]
            assert all(isinstance(score, float) for score in scores)
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)

    def test_update_index_with_new_chunks(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test updating index with additional chunks."""
        # Build initial index with first 30 chunks
        initial_chunks = realistic_chunks[:30]
        repository.build_index(initial_chunks)

        initial_stats = repository.get_index_stats()
        assert initial_stats["chunk_count"] == 30

        # Update with all chunks
        repository.update_index(realistic_chunks)

        updated_stats = repository.get_index_stats()
        assert updated_stats["chunk_count"] >= 50

        # Search should work with updated index
        results = repository.search("Guilliman")
        assert len(results) > 0

    def test_cleanup_after_test(
        self,
        repository: BM25Repository,
        realistic_chunks: list[Chunk],
    ) -> None:
        """Test that temporary files are cleaned up."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "cleanup_test.pkl"

            repository.build_index(realistic_chunks)
            repository.save_index(index_path)

            assert index_path.exists()

        # After exiting context, tmpdir should be cleaned up
        assert not index_path.exists()
