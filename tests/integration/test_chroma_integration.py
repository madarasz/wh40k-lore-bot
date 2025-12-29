"""Integration tests for ChromaVectorStore with real Chroma database."""

import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore

# Test storage path
TEST_STORAGE_PATH = "test-data/chroma-integration/"
TEST_COLLECTION_NAME = "test-wh40k-lore"


@pytest.fixture
def vector_store():
    """Create real ChromaVectorStore for testing.

    Each test gets a completely separate storage path to avoid database locking.
    """
    # Use unique storage path per test to avoid locking
    unique_id = f"{int(time.time() * 1000000)}-{uuid.uuid4().hex[:8]}"
    unique_path = f"{TEST_STORAGE_PATH}{unique_id}/"

    store = ChromaVectorStore(
        storage_path=unique_path,
        collection_name=TEST_COLLECTION_NAME,
    )
    yield store

    # Clean up storage path completely
    try:
        path = Path(unique_path)
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


@pytest.fixture
def test_chunks():
    """Create 100 test chunks with various metadata."""
    chunks = []
    factions = ["Space Marines", "Orks", "Eldar", "Chaos", "Tyranids"]
    eras = ["Great Crusade", "Horus Heresy", "War of the Beast", "Age of Apostasy", "Current"]
    content_types = ["lore", "tactics", "units"]

    for i in range(100):
        chunk = WikiChunk(
            id=f"test-chunk-{i}",
            wiki_page_id=f"page-{i % 10}",
            article_title=f"Article {i % 20}",
            section_path=f"Section {i % 5} > Subsection {i % 3}",
            chunk_text=f"This is test chunk {i} with various content about Warhammer 40k lore.",
            chunk_index=i % 10,
            metadata_json={
                "faction": factions[i % len(factions)],
                "era": eras[i % len(eras)] if i % 3 != 0 else None,  # Some without era
                "spoiler_flag": i % 4 == 0,  # Every 4th chunk is a spoiler
                "content_type": content_types[i % len(content_types)],
            },
        )
        # Remove era if None
        if chunk.metadata_json["era"] is None:
            del chunk.metadata_json["era"]
        chunks.append(chunk)

    return chunks


@pytest.fixture
def test_embeddings():
    """Create 100 test embeddings (1536-dim).

    Embeddings are designed so that similar indices have similar embeddings.
    """
    embeddings = []
    base_vector = np.random.rand(1536).astype(np.float32)

    for _i in range(100):
        # Add small random variation to base vector
        variation = np.random.rand(1536).astype(np.float32) * 0.1
        embedding = base_vector + variation
        # Normalize to unit vector (for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

    return embeddings


class TestChromaIntegration:
    """Integration tests with real Chroma database."""

    def test_add_and_count(self, vector_store, test_chunks, test_embeddings):
        """Test adding chunks and counting them."""
        # Initially empty
        assert vector_store.count() == 0

        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Verify count
        assert vector_store.count() == 100

    def test_query_without_filters(self, vector_store, test_chunks, test_embeddings):
        """Test basic query without filters."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query with first embedding
        query_embedding = test_embeddings[0]
        results = vector_store.query(query_embedding, n_results=10)

        # Should return 10 results
        assert len(results) == 10

        # First result should be the exact match (chunk 0)
        chunk, distance = results[0]
        assert chunk.id == "test-chunk-0"
        assert distance < 0.01  # Very close to 0 for exact match

        # All results should have WikiChunk objects and distance scores
        for chunk, distance in results:
            assert isinstance(chunk, WikiChunk)
            assert isinstance(distance, float)
            # Valid distance range for cosine (allow tiny negative due to float precision)
            assert -0.001 <= distance <= 2

    def test_query_with_faction_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query with faction filter."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for Space Marines only
        query_embedding = test_embeddings[0]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"faction": "Space Marines"},
        )

        # Should return results (20 Space Marines chunks total)
        assert len(results) > 0
        assert len(results) <= 10

        # All results should be Space Marines
        for chunk, _distance in results:
            assert chunk.metadata_json["faction"] == "Space Marines"

    def test_query_with_era_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query with era filter."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for Horus Heresy only
        query_embedding = test_embeddings[1]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"era": "Horus Heresy"},
        )

        # Should return results
        assert len(results) > 0

        # All results should be from Horus Heresy era
        for chunk, _distance in results:
            assert chunk.metadata_json.get("era") == "Horus Heresy"

    def test_query_with_spoiler_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query excluding spoilers."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query excluding spoilers
        query_embedding = test_embeddings[2]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"spoiler_flag": False},
        )

        # Should return results
        assert len(results) > 0

        # All results should not be spoilers
        for chunk, _distance in results:
            assert chunk.metadata_json["spoiler_flag"] is False

    def test_query_with_compound_filters(self, vector_store, test_chunks, test_embeddings):
        """Test query with multiple filters (AND logic)."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for Space Marines without spoilers
        query_embedding = test_embeddings[3]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={
                "faction": "Space Marines",
                "spoiler_flag": False,
            },
        )

        # Should return results
        assert len(results) > 0

        # All results should match all filters
        for chunk, _distance in results:
            assert chunk.metadata_json["faction"] == "Space Marines"
            assert chunk.metadata_json["spoiler_flag"] is False

    def test_get_by_id(self, vector_store, test_chunks, test_embeddings):
        """Test retrieving chunk by ID."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Get specific chunk
        chunk = vector_store.get_by_id("test-chunk-42")

        assert chunk is not None
        assert chunk.id == "test-chunk-42"
        assert chunk.article_title == "Article 2"  # 42 % 20 = 2
        assert chunk.chunk_text.startswith("This is test chunk 42")

    def test_get_by_id_nonexistent(self, vector_store, test_chunks, test_embeddings):
        """Test retrieving nonexistent chunk."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Get nonexistent chunk
        chunk = vector_store.get_by_id("nonexistent-id")

        assert chunk is None

    def test_persistence(self, test_chunks, test_embeddings):
        """Test that data persists across sessions."""
        # Use unique storage path for this test
        unique_id = f"{int(time.time() * 1000000)}-{uuid.uuid4().hex[:8]}"
        unique_path = f"{TEST_STORAGE_PATH}persistence-{unique_id}/"

        try:
            # Create store and add chunks
            store1 = ChromaVectorStore(
                storage_path=unique_path,
                collection_name=TEST_COLLECTION_NAME,
            )
            store1.add_chunks(test_chunks[:50], test_embeddings[:50])
            assert store1.count() == 50

            # Create new store instance (simulates restart)
            store2 = ChromaVectorStore(
                storage_path=unique_path,
                collection_name=TEST_COLLECTION_NAME,
            )

            # Data should still be there
            assert store2.count() == 50

            # Should be able to query
            query_embedding = test_embeddings[0]
            results = store2.query(query_embedding, n_results=5)
            assert len(results) == 5

            # Clean up
            store2.delete_collection()
        finally:
            # Clean up storage path
            path = Path(unique_path)
            if path.exists():
                shutil.rmtree(path)

    def test_cosine_similarity_scores(self, vector_store, test_chunks, test_embeddings):
        """Test that cosine similarity scores are reasonable."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query with embedding
        query_embedding = test_embeddings[10]
        results = vector_store.query(query_embedding, n_results=20)

        # Scores should be in ascending order (lower = more similar)
        distances = [distance for _, distance in results]
        assert distances == sorted(distances)

        # First result should be very close (exact match)
        assert results[0][1] < 0.01

        # Scores should be in valid range for cosine distance
        # Allow tiny negative values due to floating-point precision
        for _, distance in results:
            assert -0.001 <= distance <= 2

    def test_batch_insertion(self, vector_store):
        """Test insertion of >1000 chunks (multiple batches)."""
        # Create 2500 chunks
        chunks = []
        embeddings = []
        for i in range(2500):
            chunk = WikiChunk(
                id=f"batch-chunk-{i}",
                wiki_page_id="page-1",
                article_title="Batch Test",
                section_path="Test",
                chunk_text=f"Batch chunk {i}",
                chunk_index=i,
                metadata_json={
                    "spoiler_flag": False,
                    "content_type": "lore",
                },
            )
            chunks.append(chunk)
            embeddings.append(np.random.rand(1536).astype(np.float32))

        # Add all chunks (should use 3 batches)
        vector_store.add_chunks(chunks, embeddings)

        # Verify all were added
        assert vector_store.count() == 2500

        # Verify we can query
        query_embedding = embeddings[0]
        results = vector_store.query(query_embedding, n_results=10)
        assert len(results) == 10

    def test_empty_results_with_strict_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query that returns no results due to strict filtering."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query with impossible filter combination
        query_embedding = test_embeddings[0]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"faction": "NonExistentFaction"},
        )

        # Should return empty list
        assert results == []

    def test_metadata_preservation(self, vector_store, test_chunks, test_embeddings):
        """Test that all metadata fields are preserved correctly."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Get chunk with all metadata fields
        chunk = vector_store.get_by_id("test-chunk-0")

        assert chunk is not None
        assert chunk.metadata_json["faction"] == "Space Marines"
        assert chunk.metadata_json["spoiler_flag"] is True  # chunk 0: 0 % 4 == 0
        assert chunk.metadata_json["content_type"] == "lore"

        # Get chunk with optional era field
        chunk_with_era = vector_store.get_by_id("test-chunk-1")
        assert chunk_with_era is not None
        assert "era" in chunk_with_era.metadata_json

        # Get chunk without era field (index divisible by 3)
        chunk_no_era = vector_store.get_by_id("test-chunk-3")
        assert chunk_no_era is not None
        assert "era" not in chunk_no_era.metadata_json
