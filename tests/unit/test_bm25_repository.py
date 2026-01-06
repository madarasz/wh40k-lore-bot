"""Unit tests for BM25Repository."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.ingestion.models import Chunk
from src.repositories.bm25_repository import BM25Repository


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_text="Roboute Guilliman is the Primarch of the Ultramarines",
            article_title="Roboute Guilliman",
            section_path="Introduction",
            chunk_index=0,
            links=["Ultramarines", "Primarch"],
            wiki_page_id="12345",
        ),
        Chunk(
            chunk_text="The Ultramarines are a Space Marine chapter founded by Guilliman",
            article_title="Ultramarines",
            section_path="History",
            chunk_index=0,
            links=["Roboute Guilliman"],
            wiki_page_id="23456",
        ),
        Chunk(
            chunk_text="Macragge is the homeworld of the Ultramarines",
            article_title="Macragge",
            section_path="Overview",
            chunk_index=0,
            links=["Ultramarines"],
            wiki_page_id="34567",
        ),
        Chunk(
            chunk_text="The Horus Heresy was a galactic civil war",
            article_title="Horus Heresy",
            section_path="Overview",
            chunk_index=0,
            links=["Horus"],
            wiki_page_id="45678",
        ),
        Chunk(
            chunk_text="Horus was the Warmaster who betrayed the Emperor",
            article_title="Horus",
            section_path="Fall to Chaos",
            chunk_index=0,
            links=["Emperor", "Chaos"],
            wiki_page_id="56789",
        ),
    ]


@pytest.fixture
def repository() -> BM25Repository:
    """Create a BM25Repository instance."""
    return BM25Repository(tokenize_lowercase=True)


class TestTokenization:
    """Test tokenization functionality."""

    def test_tokenize_simple_text(self, repository: BM25Repository) -> None:
        """Test basic tokenization."""
        text = "Hello World"
        tokens = repository._tokenize(text)
        assert tokens == ["hello", "world"]

    def test_tokenize_lowercase(self, repository: BM25Repository) -> None:
        """Test lowercase conversion."""
        text = "Roboute GUILLIMAN"
        tokens = repository._tokenize(text)
        assert tokens == ["roboute", "guilliman"]

    def test_tokenize_whitespace_split(self, repository: BM25Repository) -> None:
        """Test whitespace-based splitting."""
        text = "multiple   spaces\ttabs\nnewlines"
        tokens = repository._tokenize(text)
        assert tokens == ["multiple", "spaces", "tabs", "newlines"]

    def test_tokenize_empty_string(self, repository: BM25Repository) -> None:
        """Test empty string handling."""
        assert repository._tokenize("") == []
        assert repository._tokenize("   ") == []

    def test_tokenize_no_lowercase(self) -> None:
        """Test tokenization without lowercasing."""
        repo = BM25Repository(tokenize_lowercase=False)
        text = "Roboute GUILLIMAN"
        tokens = repo._tokenize(text)
        assert tokens == ["Roboute", "GUILLIMAN"]


class TestIndexBuilding:
    """Test index building functionality."""

    def test_build_index_success(
        self, repository: BM25Repository, sample_chunks: list[Chunk]
    ) -> None:
        """Test successful index building."""
        repository.build_index(sample_chunks)

        assert repository.bm25 is not None
        assert len(repository.chunk_ids) == 5

    def test_build_index_chunk_ids(
        self, repository: BM25Repository, sample_chunks: list[Chunk]
    ) -> None:
        """Test chunk IDs are created correctly."""
        repository.build_index(sample_chunks)

        # Check chunk IDs are in expected format (wiki_page_id_chunk_index)
        expected_ids = [
            "12345_0",
            "23456_0",
            "34567_0",
            "45678_0",
            "56789_0",
        ]
        assert repository.chunk_ids == expected_ids

    def test_build_index_empty_chunks(self, repository: BM25Repository) -> None:
        """Test building index with empty chunks list raises error."""
        with pytest.raises(ValueError, match="Cannot build index from empty chunks list"):
            repository.build_index([])

    def test_is_index_built(self, repository: BM25Repository, sample_chunks: list[Chunk]) -> None:
        """Test index built status check."""
        assert not repository.is_index_built()

        repository.build_index(sample_chunks)
        assert repository.is_index_built()


class TestSearch:
    """Test search functionality."""

    @pytest.fixture(autouse=True)
    def setup_index(self, repository: BM25Repository, sample_chunks: list[Chunk]) -> None:
        """Build index before each search test."""
        repository.build_index(sample_chunks)

    def test_search_exact_match(self, repository: BM25Repository) -> None:
        """Test exact keyword match returns high score."""
        results = repository.search("Guilliman")

        # Should find results
        assert len(results) > 0
        # First result should be a chunk_id string
        assert isinstance(results[0][0], str)
        # Score should be positive
        assert results[0][1] > 0
        # Should return chunk IDs for Guilliman chunks (12345_0 or 23456_0)
        top_chunk_ids = [r[0] for r in results[:2]]
        assert any(cid in ["12345_0", "23456_0"] for cid in top_chunk_ids)

    def test_search_partial_match(self, repository: BM25Repository) -> None:
        """Test partial keyword match."""
        results = repository.search("Ultramarines homeworld")

        # Should find chunks with either word
        assert len(results) > 0
        # Results should be chunk_id strings
        assert all(isinstance(r[0], str) for r in results)
        # Macragge chunk (34567_0) should rank high (contains "homeworld")
        top_chunk_ids = [r[0] for r in results[:3]]
        assert "34567_0" in top_chunk_ids

    def test_search_no_match(self, repository: BM25Repository) -> None:
        """Test search with no matching keywords."""
        results = repository.search("xyzabc nonexistent keywords")

        # Should return results but with low scores
        assert len(results) > 0
        # Scores should be very low or zero
        assert all(score <= 1.0 for _, score in results)

    def test_search_score_ordering(self, repository: BM25Repository) -> None:
        """Test results are sorted by score descending."""
        results = repository.search("Guilliman Ultramarines")

        # Verify descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_limiting(self, repository: BM25Repository) -> None:
        """Test top_k limits number of results."""
        results = repository.search("the", top_k=3)

        assert len(results) == 3

    def test_search_empty_query(self, repository: BM25Repository) -> None:
        """Test empty query raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            repository.search("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            repository.search("   ")

    def test_search_before_index_built(self) -> None:
        """Test search before building index raises error."""
        repo = BM25Repository()
        with pytest.raises(ValueError, match="Index not built"):
            repo.search("test")


class TestIndexPersistence:
    """Test save and load functionality."""

    def test_save_and_load_index(
        self, repository: BM25Repository, sample_chunks: list[Chunk]
    ) -> None:
        """Test roundtrip save and load."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.pkl"

            # Build and save
            repository.build_index(sample_chunks)
            repository.save_index(index_path)

            assert index_path.exists()

            # Load into new repository
            new_repo = BM25Repository()
            new_repo.load_index(index_path)

            # Verify loaded data
            assert new_repo.is_index_built()
            assert len(new_repo.chunk_ids) == len(repository.chunk_ids)
            assert new_repo.chunk_ids == repository.chunk_ids

            # Verify search works the same
            query = "Guilliman"
            original_results = repository.search(query)
            loaded_results = new_repo.search(query)

            assert len(original_results) == len(loaded_results)
            # Chunk IDs and scores should match
            for orig, loaded in zip(original_results, loaded_results, strict=True):
                assert orig[0] == loaded[0]  # chunk_id
                assert orig[1] == loaded[1]  # score

    def test_save_without_index(self, repository: BM25Repository) -> None:
        """Test save without building index raises error."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.pkl"
            with pytest.raises(ValueError, match="Index not built"):
                repository.save_index(index_path)

    def test_load_nonexistent_file(self, repository: BM25Repository) -> None:
        """Test load from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            repository.load_index(Path("/nonexistent/path/index.pkl"))

    def test_save_creates_directory(
        self, repository: BM25Repository, sample_chunks: list[Chunk]
    ) -> None:
        """Test save creates parent directories."""
        with TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "subdir" / "nested" / "index.pkl"

            repository.build_index(sample_chunks)
            repository.save_index(index_path)

            assert index_path.exists()
            assert index_path.parent.exists()


class TestUpdateIndex:
    """Test index update functionality."""

    def test_update_index(self, repository: BM25Repository, sample_chunks: list[Chunk]) -> None:
        """Test update rebuilds index."""
        # Build initial index
        repository.build_index(sample_chunks[:3])
        assert len(repository.chunk_ids) == 3

        # Update with different chunks
        repository.update_index(sample_chunks)
        assert len(repository.chunk_ids) == 5

        # Verify search works with new index
        results = repository.search("Horus")
        assert len(results) > 0
        # Should find Horus chunks (45678_0 or 56789_0)
        chunk_ids = [r[0] for r in results]
        assert any(cid in ["45678_0", "56789_0"] for cid in chunk_ids)

    def test_update_empty_chunks(self, repository: BM25Repository) -> None:
        """Test update with empty chunks raises error."""
        with pytest.raises(ValueError, match="Cannot build index from empty chunks list"):
            repository.update_index([])


class TestIndexStats:
    """Test index statistics functionality."""

    def test_get_stats_no_index(self, repository: BM25Repository) -> None:
        """Test stats when no index built."""
        stats = repository.get_index_stats()

        assert stats["total_chunks"] == 0
        assert stats["unique_tokens"] == 0
        assert stats["index_built"] is False
        assert "index_path" in stats

    def test_get_stats_with_index(
        self, repository: BM25Repository, sample_chunks: list[Chunk]
    ) -> None:
        """Test stats when index is built."""
        repository.build_index(sample_chunks)
        stats = repository.get_index_stats()

        assert stats["total_chunks"] == 5
        assert stats["unique_tokens"] > 0  # Should have extracted unique tokens
        assert stats["index_built"] is True
        assert "index_path" in stats
