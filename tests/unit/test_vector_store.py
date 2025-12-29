"""Unit tests for ChromaVectorStore."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore
from src.utils.exceptions import VectorStoreError


@pytest.fixture
def mock_chroma_client():
    """Mock Chroma client and collection."""
    with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
        # Mock client instance
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock collection
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        yield mock_client, mock_collection


@pytest.fixture
def vector_store(mock_chroma_client):
    """Create ChromaVectorStore with mocked Chroma."""
    mock_client, mock_collection = mock_chroma_client
    store = ChromaVectorStore(
        storage_path="test-data/chroma-db/",
        collection_name="test-collection",
    )
    return store


@pytest.fixture
def sample_chunks():
    """Create sample WikiChunk objects for testing."""
    chunks = [
        WikiChunk(
            id="chunk-1",
            wiki_page_id="page-1",
            article_title="Blood Angels",
            section_path="History > Founding",
            chunk_text="The Blood Angels are a Space Marine chapter...",
            chunk_index=0,
            metadata_json={
                "faction": "Space Marines",
                "era": "Great Crusade",
                "spoiler_flag": False,
                "content_type": "lore",
            },
        ),
        WikiChunk(
            id="chunk-2",
            wiki_page_id="page-1",
            article_title="Blood Angels",
            section_path="History > Horus Heresy",
            chunk_text="During the Horus Heresy, Sanguinius...",
            chunk_index=1,
            metadata_json={
                "faction": "Space Marines",
                "era": "Horus Heresy",
                "spoiler_flag": True,
                "content_type": "lore",
            },
        ),
        WikiChunk(
            id="chunk-3",
            wiki_page_id="page-2",
            article_title="Orks",
            section_path="Overview",
            chunk_text="Orks are a warlike, crude, and highly aggressive race...",
            chunk_index=0,
            metadata_json={
                "faction": "Orks",
                "spoiler_flag": False,
                "content_type": "lore",
            },
        ),
    ]
    return chunks


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings (1536-dim)."""
    return [
        np.random.rand(1536).astype(np.float32),
        np.random.rand(1536).astype(np.float32),
        np.random.rand(1536).astype(np.float32),
    ]


class TestChromaVectorStoreInit:
    """Test ChromaVectorStore initialization."""

    def test_init_success(self, mock_chroma_client):
        """Test successful initialization."""
        mock_client, mock_collection = mock_chroma_client

        store = ChromaVectorStore(
            storage_path="test-data/chroma-db/",
            collection_name="test-collection",
        )

        assert store.storage_path == "test-data/chroma-db/"
        assert store.collection_name == "test-collection"
        assert store.client == mock_client
        assert store.collection == mock_collection

        # Verify collection created with correct metadata
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test-collection",
            metadata={"hnsw:space": "cosine"},
        )

    def test_init_with_defaults(self, mock_chroma_client):
        """Test initialization with default parameters."""
        store = ChromaVectorStore()

        assert store.storage_path == "data/chroma-db/"
        assert store.collection_name == "wh40k-lore"

    def test_init_failure(self):
        """Test initialization failure handling."""
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(VectorStoreError, match="Failed to initialize"):
                ChromaVectorStore()


class TestAddChunks:
    """Test add_chunks method."""

    def test_add_chunks_success(self, vector_store, sample_chunks, sample_embeddings):
        """Test successful chunk insertion."""
        vector_store.add_chunks(sample_chunks, sample_embeddings)

        # Verify collection.add was called
        vector_store.collection.add.assert_called_once()

        # Verify correct data format
        call_args = vector_store.collection.add.call_args
        assert call_args.kwargs["ids"] == ["chunk-1", "chunk-2", "chunk-3"]
        assert len(call_args.kwargs["embeddings"]) == 3
        assert len(call_args.kwargs["metadatas"]) == 3
        assert len(call_args.kwargs["documents"]) == 3

        # Verify metadata structure
        metadata_0 = call_args.kwargs["metadatas"][0]
        assert metadata_0["article_title"] == "Blood Angels"
        assert metadata_0["faction"] == "Space Marines"
        assert metadata_0["era"] == "Great Crusade"
        assert metadata_0["spoiler_flag"] is False

    def test_add_chunks_length_mismatch(self, vector_store, sample_chunks, sample_embeddings):
        """Test error when chunks and embeddings length don't match."""
        with pytest.raises(VectorStoreError, match="length mismatch"):
            vector_store.add_chunks(sample_chunks, sample_embeddings[:2])

    def test_add_chunks_empty_list(self, vector_store):
        """Test add_chunks with empty list."""
        vector_store.add_chunks([], [])

        # Should not call collection.add
        vector_store.collection.add.assert_not_called()

    def test_add_chunks_batching(self, vector_store):
        """Test batch processing for large number of chunks."""
        # Create 2500 chunks (should result in 3 batches of 1000, 1000, 500)
        chunks = []
        embeddings = []
        for i in range(2500):
            chunk = WikiChunk(
                id=f"chunk-{i}",
                wiki_page_id="page-1",
                article_title="Test Article",
                section_path="Test Section",
                chunk_text=f"Test content {i}",
                chunk_index=i,
                metadata_json={
                    "spoiler_flag": False,
                    "content_type": "lore",
                },
            )
            chunks.append(chunk)
            embeddings.append(np.random.rand(1536).astype(np.float32))

        vector_store.add_chunks(chunks, embeddings)

        # Verify collection.add called 3 times (3 batches)
        assert vector_store.collection.add.call_count == 3

        # Verify batch sizes
        call_args_list = vector_store.collection.add.call_args_list
        assert len(call_args_list[0].kwargs["ids"]) == 1000
        assert len(call_args_list[1].kwargs["ids"]) == 1000
        assert len(call_args_list[2].kwargs["ids"]) == 500

    def test_add_chunks_optional_metadata(self, vector_store):
        """Test adding chunks with optional metadata fields missing."""
        chunk = WikiChunk(
            id="chunk-1",
            wiki_page_id="page-1",
            article_title="Test Article",
            section_path="Test Section",
            chunk_text="Test content",
            chunk_index=0,
            metadata_json={
                "spoiler_flag": False,
                "content_type": "lore",
                # faction and era are optional and missing
            },
        )
        embedding = np.random.rand(1536).astype(np.float32)

        vector_store.add_chunks([chunk], [embedding])

        # Verify metadata doesn't include optional fields
        call_args = vector_store.collection.add.call_args
        metadata = call_args.kwargs["metadatas"][0]
        assert "faction" not in metadata
        assert "era" not in metadata
        assert metadata["spoiler_flag"] is False

    def test_add_chunks_failure(self, vector_store, sample_chunks, sample_embeddings):
        """Test handling of insertion failure."""
        vector_store.collection.add.side_effect = Exception("Insertion failed")

        with pytest.raises(VectorStoreError, match="Failed to add chunks"):
            vector_store.add_chunks(sample_chunks, sample_embeddings)


class TestQuery:
    """Test query method."""

    def test_query_success(self, vector_store):
        """Test successful query."""
        # Mock query results
        vector_store.collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "distances": [[0.15, 0.23]],
            "metadatas": [
                [
                    {
                        "article_title": "Blood Angels",
                        "section_path": "History",
                        "chunk_index": 0,
                        "faction": "Space Marines",
                        "spoiler_flag": False,
                        "content_type": "lore",
                    },
                    {
                        "article_title": "Orks",
                        "section_path": "Overview",
                        "chunk_index": 0,
                        "spoiler_flag": False,
                        "content_type": "lore",
                    },
                ]
            ],
            "documents": [["Blood Angels text...", "Orks text..."]],
        }

        query_embedding = np.random.rand(1536).astype(np.float32)
        results = vector_store.query(query_embedding, n_results=10)

        # Verify results
        assert len(results) == 2
        chunk1, score1 = results[0]
        chunk2, score2 = results[1]

        assert chunk1.id == "chunk-1"
        assert chunk1.article_title == "Blood Angels"
        assert score1 == 0.15

        assert chunk2.id == "chunk-2"
        assert chunk2.article_title == "Orks"
        assert score2 == 0.23

    def test_query_with_filters(self, vector_store):
        """Test query with metadata filters."""
        vector_store.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }

        query_embedding = np.random.rand(1536).astype(np.float32)
        filters = {"faction": "Space Marines", "spoiler_flag": False}

        vector_store.query(query_embedding, n_results=10, filters=filters)

        # Verify filters passed to collection.query
        # Compound filters are converted to $and format
        call_args = vector_store.collection.query.call_args
        expected_filters = {"$and": [{"faction": "Space Marines"}, {"spoiler_flag": False}]}
        assert call_args.kwargs["where"] == expected_filters

    def test_query_empty_results(self, vector_store):
        """Test query with no results."""
        # Mock empty results
        vector_store.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }

        query_embedding = np.random.rand(1536).astype(np.float32)
        results = vector_store.query(query_embedding, n_results=10)

        assert results == []

    def test_query_failure(self, vector_store):
        """Test handling of query failure."""
        vector_store.collection.query.side_effect = Exception("Query failed")

        query_embedding = np.random.rand(1536).astype(np.float32)

        with pytest.raises(VectorStoreError, match="Failed to query"):
            vector_store.query(query_embedding, n_results=10)


class TestUtilityMethods:
    """Test utility methods."""

    def test_count_success(self, vector_store):
        """Test successful count."""
        vector_store.collection.count.return_value = 42

        count = vector_store.count()

        assert count == 42
        vector_store.collection.count.assert_called_once()

    def test_count_failure(self, vector_store):
        """Test count failure handling."""
        vector_store.collection.count.side_effect = Exception("Count failed")

        with pytest.raises(VectorStoreError, match="Failed to get collection count"):
            vector_store.count()

    def test_delete_collection_success(self, vector_store):
        """Test successful collection deletion."""
        vector_store.delete_collection()

        vector_store.client.delete_collection.assert_called_once_with(name="test-collection")

    def test_delete_collection_failure(self, vector_store):
        """Test collection deletion failure."""
        vector_store.client.delete_collection.side_effect = Exception("Delete failed")

        with pytest.raises(VectorStoreError, match="Failed to delete collection"):
            vector_store.delete_collection()

    def test_get_by_id_success(self, vector_store):
        """Test successful get by ID."""
        vector_store.collection.get.return_value = {
            "ids": ["chunk-1"],
            "metadatas": [
                {
                    "article_title": "Blood Angels",
                    "section_path": "History",
                    "chunk_index": 0,
                    "faction": "Space Marines",
                    "spoiler_flag": False,
                    "content_type": "lore",
                }
            ],
            "documents": ["Blood Angels text..."],
        }

        chunk = vector_store.get_by_id("chunk-1")

        assert chunk is not None
        assert chunk.id == "chunk-1"
        assert chunk.article_title == "Blood Angels"
        assert chunk.metadata_json["faction"] == "Space Marines"

    def test_get_by_id_not_found(self, vector_store):
        """Test get by ID when chunk not found."""
        vector_store.collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
        }

        chunk = vector_store.get_by_id("nonexistent-id")

        assert chunk is None

    def test_get_by_id_failure(self, vector_store):
        """Test get by ID failure handling."""
        vector_store.collection.get.side_effect = Exception("Get failed")

        with pytest.raises(VectorStoreError, match="Failed to get chunk by ID"):
            vector_store.get_by_id("chunk-1")


class TestMetadataConversion:
    """Test metadata conversion methods."""

    def test_chunk_to_metadata_all_fields(self, vector_store):
        """Test converting WikiChunk to metadata with all fields."""
        chunk = WikiChunk(
            id="chunk-1",
            wiki_page_id="page-1",
            article_title="Blood Angels",
            section_path="History > Founding",
            chunk_text="Test content",
            chunk_index=5,
            metadata_json={
                "faction": "Space Marines",
                "era": "Great Crusade",
                "spoiler_flag": True,
                "content_type": "lore",
            },
        )

        metadata = vector_store._chunk_to_metadata(chunk)

        assert metadata["article_title"] == "Blood Angels"
        assert metadata["section_path"] == "History > Founding"
        assert metadata["chunk_index"] == 5
        assert metadata["faction"] == "Space Marines"
        assert metadata["era"] == "Great Crusade"
        assert metadata["spoiler_flag"] is True
        assert metadata["content_type"] == "lore"

    def test_chunk_to_metadata_optional_fields_missing(self, vector_store):
        """Test converting WikiChunk with optional fields missing."""
        chunk = WikiChunk(
            id="chunk-1",
            wiki_page_id="page-1",
            article_title="Test Article",
            section_path="Test Section",
            chunk_text="Test content",
            chunk_index=0,
            metadata_json={
                "spoiler_flag": False,
                "content_type": "lore",
            },
        )

        metadata = vector_store._chunk_to_metadata(chunk)

        assert "faction" not in metadata
        assert "era" not in metadata
        assert metadata["spoiler_flag"] is False
        assert metadata["content_type"] == "lore"

    def test_metadata_to_chunk(self, vector_store):
        """Test converting metadata back to WikiChunk."""
        metadata = {
            "article_title": "Blood Angels",
            "section_path": "History",
            "chunk_index": 3,
            "faction": "Space Marines",
            "era": "Great Crusade",
            "spoiler_flag": True,
            "content_type": "lore",
        }

        chunk = vector_store._metadata_to_chunk(
            chunk_id="chunk-1",
            metadata=metadata,
            document="Test content",
        )

        assert chunk.id == "chunk-1"
        assert chunk.article_title == "Blood Angels"
        assert chunk.section_path == "History"
        assert chunk.chunk_index == 3
        assert chunk.chunk_text == "Test content"
        assert chunk.metadata_json["faction"] == "Space Marines"
        assert chunk.metadata_json["era"] == "Great Crusade"
        assert chunk.metadata_json["spoiler_flag"] is True
        assert chunk.metadata_json["content_type"] == "lore"
