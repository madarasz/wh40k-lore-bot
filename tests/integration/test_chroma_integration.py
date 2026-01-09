"""Integration tests for ChromaVectorStore with real Chroma database."""

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.rag.vector_store import ChromaVectorStore, ChunkData

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
    """Create 100 test chunks with various metadata.

    Note: Schema was refactored to only include article_last_updated and links.
    Filtering is done via top-level fields: wiki_page_id, article_title, section_path.
    """
    chunks: list[ChunkData] = []
    section_paths = ["Infobox", "History", "Organization", "Notable Members", "Equipment"]

    for i in range(100):
        # New schema: only article_last_updated and links in metadata
        metadata: dict[str, Any] = {
            "article_last_updated": f"2024-01-{(i % 28) + 1:02d}T12:00:00Z",
        }
        # Add links for some chunks
        if i % 3 == 0:
            metadata["links"] = [f"Link_{i}_1", f"Link_{i}_2"]

        chunk: ChunkData = {
            "id": f"test-chunk-{i}",
            "wiki_page_id": f"page-{i % 10}",
            "article_title": f"Article {i % 20}",
            "section_path": section_paths[i % len(section_paths)],
            "chunk_text": f"This is test chunk {i} with various content about Warhammer 40k lore.",
            "chunk_index": i % 10,
            "metadata": metadata,
        }
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
        assert chunk["id"] == "test-chunk-0"
        assert distance < 0.01  # Very close to 0 for exact match

        # All results should have ChunkData dicts and distance scores
        for chunk, distance in results:
            assert isinstance(chunk, dict)
            assert isinstance(distance, float)
            # Valid distance range for cosine (allow tiny negative due to float precision)
            assert -0.001 <= distance <= 2

    def test_query_with_wiki_page_id_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query with wiki_page_id filter."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for page-0 only (10 chunks with this wiki_page_id)
        query_embedding = test_embeddings[0]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"wiki_page_id": "page-0"},
        )

        # Should return results (10 chunks with wiki_page_id="page-0")
        assert len(results) > 0
        assert len(results) <= 10

        # All results should have wiki_page_id="page-0"
        for chunk, _distance in results:
            assert chunk["wiki_page_id"] == "page-0"

    def test_query_with_section_path_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query with section_path filter."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for History sections only (20 chunks with this section_path)
        query_embedding = test_embeddings[1]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"section_path": "History"},
        )

        # Should return results
        assert len(results) > 0

        # All results should have section_path="History"
        for chunk, _distance in results:
            assert chunk["section_path"] == "History"

    def test_query_with_article_title_filter(self, vector_store, test_chunks, test_embeddings):
        """Test query with article_title filter."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for Article 0 only (5 chunks with this title)
        query_embedding = test_embeddings[2]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"article_title": "Article 0"},
        )

        # Should return results
        assert len(results) > 0

        # All results should have article_title="Article 0"
        for chunk, _distance in results:
            assert chunk["article_title"] == "Article 0"

    def test_query_with_compound_filters(self, vector_store, test_chunks, test_embeddings):
        """Test query with multiple filters (AND logic)."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Query for specific wiki_page_id and section_path
        query_embedding = test_embeddings[3]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={
                "wiki_page_id": "page-0",
                "section_path": "Infobox",
            },
        )

        # Should return results (page-0 has chunks at indices 0,10,20,30,40,50,60,70,80,90
        # and Infobox is at indices 0,5,10,15,... so intersection is at 0,10,20,...)
        assert len(results) > 0

        # All results should match all filters
        for chunk, _distance in results:
            assert chunk["wiki_page_id"] == "page-0"
            assert chunk["section_path"] == "Infobox"

    def test_get_by_id(self, vector_store, test_chunks, test_embeddings):
        """Test retrieving chunk by ID."""
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Get specific chunk
        chunk = vector_store.get_by_id("test-chunk-42")

        assert chunk is not None
        assert chunk["id"] == "test-chunk-42"
        assert chunk["article_title"] == "Article 2"  # 42 % 20 = 2
        assert chunk["chunk_text"].startswith("This is test chunk 42")

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

    @pytest.mark.long
    def test_batch_insertion(self, vector_store):
        """Test insertion of >1000 chunks (multiple batches)."""
        # Create 2500 chunks
        chunks: list[ChunkData] = []
        embeddings = []
        for i in range(2500):
            chunk: ChunkData = {
                "id": f"batch-chunk-{i}",
                "wiki_page_id": "page-1",
                "article_title": "Batch Test",
                "section_path": "Test",
                "chunk_text": f"Batch chunk {i}",
                "chunk_index": i,
                "metadata": {
                    "article_last_updated": "2024-01-01T12:00:00Z",
                },
            }
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

        # Query with non-existent wiki_page_id
        query_embedding = test_embeddings[0]
        results = vector_store.query(
            query_embedding,
            n_results=10,
            filters={"wiki_page_id": "nonexistent-page"},
        )

        # Should return empty list
        assert results == []

    def test_metadata_preservation(self, vector_store, test_chunks, test_embeddings):
        """Test that all metadata fields are preserved correctly.

        Note: Schema was refactored to only include article_last_updated and links.
        """
        # Add chunks
        vector_store.add_chunks(test_chunks, test_embeddings)

        # Get chunk with all metadata fields (chunk 0 has links since 0 % 3 == 0)
        chunk = vector_store.get_by_id("test-chunk-0")

        assert chunk is not None
        assert chunk["metadata"]["article_last_updated"] == "2024-01-01T12:00:00Z"
        assert chunk["metadata"]["links"] == ["Link_0_1", "Link_0_2"]

        # Get chunk without links (chunk 1: 1 % 3 != 0)
        chunk_no_links = vector_store.get_by_id("test-chunk-1")
        assert chunk_no_links is not None
        assert chunk_no_links["metadata"]["article_last_updated"] == "2024-01-02T12:00:00Z"
        assert (
            "links" not in chunk_no_links["metadata"] or chunk_no_links["metadata"]["links"] == []
        )

        # Get chunk with links (chunk 3: 3 % 3 == 0)
        chunk_with_links = vector_store.get_by_id("test-chunk-3")
        assert chunk_with_links is not None
        assert chunk_with_links["metadata"]["links"] == ["Link_3_1", "Link_3_2"]
