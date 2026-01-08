"""Unit tests for ContextExpander."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.context_expander import ContextExpander
from src.rag.vector_store import ChunkData
from src.utils.exceptions import ValidationError


@pytest.fixture
def mock_vector_store():
    """Create a mock ChromaVectorStore."""
    store = MagicMock()
    store.collection = MagicMock()
    store.metadata_to_chunk = MagicMock()
    return store


@pytest.fixture
def sample_chunks() -> list[ChunkData]:
    """Create sample chunks with links."""
    return [
        {
            "id": "chunk1",
            "wiki_page_id": "page1",
            "article_title": "Roboute Guilliman",
            "section_path": "Biography",
            "chunk_text": "Roboute Guilliman is the Primarch of the Ultramarines.",
            "chunk_index": 0,
            "metadata": {
                "links": ["Ultramarines", "Primarch"],
                "article_last_updated": "2024-01-01",
            },
        },
        {
            "id": "chunk2",
            "wiki_page_id": "page1",
            "article_title": "Roboute Guilliman",
            "section_path": "Biography",
            "chunk_text": "He is one of the most important figures in the Imperium.",
            "chunk_index": 1,
            "metadata": {
                "links": ["Imperium of Man"],
                "article_last_updated": "2024-01-01",
            },
        },
    ]


@pytest.fixture
def sample_linked_chunks() -> dict[str, list[ChunkData]]:
    """Create sample linked chunks for each article."""
    return {
        "Ultramarines": [
            {
                "id": "ultra1",
                "wiki_page_id": "page2",
                "article_title": "Ultramarines",
                "section_path": "Overview",
                "chunk_text": "The Ultramarines are a Space Marine chapter.",
                "chunk_index": 0,
                "metadata": {
                    "links": ["Space Marines"],
                    "article_last_updated": "2024-01-01",
                },
            },
            {
                "id": "ultra2",
                "wiki_page_id": "page2",
                "article_title": "Ultramarines",
                "section_path": "History",
                "chunk_text": "They are known for their discipline.",
                "chunk_index": 1,
                "metadata": {
                    "links": [],
                    "article_last_updated": "2024-01-01",
                },
            },
        ],
        "Primarch": [
            {
                "id": "primarch1",
                "wiki_page_id": "page3",
                "article_title": "Primarch",
                "section_path": "Overview",
                "chunk_text": "Primarchs are genetically enhanced super-beings.",
                "chunk_index": 0,
                "metadata": {
                    "links": ["Emperor of Mankind"],
                    "article_last_updated": "2024-01-01",
                },
            },
            {
                "id": "primarch2",
                "wiki_page_id": "page3",
                "article_title": "Primarch",
                "section_path": "Creation",
                "chunk_text": "Created by the Emperor.",
                "chunk_index": 1,
                "metadata": {
                    "links": [],
                    "article_last_updated": "2024-01-01",
                },
            },
        ],
        "Imperium of Man": [
            {
                "id": "imperium1",
                "wiki_page_id": "page4",
                "article_title": "Imperium of Man",
                "section_path": "Overview",
                "chunk_text": "The Imperium is a galaxy-spanning empire.",
                "chunk_index": 0,
                "metadata": {
                    "links": ["Emperor of Mankind"],
                    "article_last_updated": "2024-01-01",
                },
            },
        ],
        "Space Marines": [
            {
                "id": "sm1",
                "wiki_page_id": "page5",
                "article_title": "Space Marines",
                "section_path": "Overview",
                "chunk_text": "Space Marines are enhanced super-soldiers.",
                "chunk_index": 0,
                "metadata": {
                    "links": [],
                    "article_last_updated": "2024-01-01",
                },
            },
        ],
        "Emperor of Mankind": [
            {
                "id": "emperor1",
                "wiki_page_id": "page6",
                "article_title": "Emperor of Mankind",
                "section_path": "Overview",
                "chunk_text": "The Emperor is the ruler of the Imperium.",
                "chunk_index": 0,
                "metadata": {
                    "links": [],
                    "article_last_updated": "2024-01-01",
                },
            },
        ],
    }


class TestContextExpanderInitialization:
    """Tests for ContextExpander initialization."""

    def test_init_with_defaults(self, mock_vector_store):
        """Test initialization with default values."""
        expander = ContextExpander(vector_store=mock_vector_store)

        assert expander.enabled is True
        assert expander.expansion_depth == 1
        assert expander.max_chunks == 30

    def test_init_with_custom_values(self, mock_vector_store):
        """Test initialization with custom values."""
        expander = ContextExpander(
            vector_store=mock_vector_store,
            enabled=False,
            expansion_depth=2,
            max_chunks=50,
        )

        assert expander.enabled is False
        assert expander.expansion_depth == 2
        assert expander.max_chunks == 50

    def test_init_with_invalid_depth(self, mock_vector_store):
        """Test initialization with invalid expansion depth."""
        with pytest.raises(ValidationError, match="Expansion depth must be 0-2"):
            ContextExpander(
                vector_store=mock_vector_store,
                expansion_depth=3,
            )

    def test_init_with_negative_max_chunks(self, mock_vector_store):
        """Test initialization with negative max_chunks."""
        with pytest.raises(ValidationError, match="Max chunks cannot be negative"):
            ContextExpander(
                vector_store=mock_vector_store,
                max_chunks=-1,
            )

    @patch.dict("os.environ", {"CONTEXT_EXPANSION_ENABLED": "false"})
    def test_init_from_env_disabled(self, mock_vector_store):
        """Test initialization with disabled expansion from environment."""
        expander = ContextExpander(vector_store=mock_vector_store)
        assert expander.enabled is False

    @patch.dict("os.environ", {"CONTEXT_EXPANSION_DEPTH": "2"})
    def test_init_from_env_depth(self, mock_vector_store):
        """Test initialization with depth from environment."""
        expander = ContextExpander(vector_store=mock_vector_store)
        assert expander.expansion_depth == 2

    @patch.dict("os.environ", {"CONTEXT_EXPANSION_MAX_CHUNKS": "50"})
    def test_init_from_env_max_chunks(self, mock_vector_store):
        """Test initialization with max_chunks from environment."""
        expander = ContextExpander(vector_store=mock_vector_store)
        assert expander.max_chunks == 50


class TestExpandContextDisabled:
    """Tests for expand_context when disabled."""

    @pytest.mark.asyncio
    async def test_expansion_disabled(self, mock_vector_store, sample_chunks):
        """Test that expansion is skipped when disabled."""
        expander = ContextExpander(vector_store=mock_vector_store, enabled=False)

        result = await expander.expand_context(sample_chunks)

        assert result == sample_chunks
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_expansion_depth_zero(self, mock_vector_store, sample_chunks):
        """Test that expansion is skipped when depth=0."""
        expander = ContextExpander(vector_store=mock_vector_store)

        result = await expander.expand_context(sample_chunks, expansion_depth=0)

        assert result == sample_chunks
        assert len(result) == 2


class TestExpandContextDepth1:
    """Tests for expand_context with depth=1."""

    @pytest.mark.asyncio
    async def test_depth1_expansion(self, mock_vector_store, sample_chunks, sample_linked_chunks):
        """Test depth=1 expansion with multiple links."""

        def mock_get(where, limit, include):
            article_title = where["article_title"]
            chunks = sample_linked_chunks.get(article_title, [])[:limit]

            if not chunks:
                return {"ids": [], "metadatas": [], "documents": []}

            return {
                "ids": [c["id"] for c in chunks],
                "metadatas": [c["metadata"] for c in chunks],
                "documents": [c["chunk_text"] for c in chunks],
            }

        def mock_metadata_to_chunk(chunk_id, metadata, document):
            # Find the chunk in sample_linked_chunks
            for chunks in sample_linked_chunks.values():
                for chunk in chunks:
                    if chunk["id"] == chunk_id:
                        return chunk
            return None

        mock_vector_store.collection.get = MagicMock(side_effect=mock_get)
        mock_vector_store.metadata_to_chunk = mock_metadata_to_chunk

        expander = ContextExpander(vector_store=mock_vector_store, expansion_depth=1)

        result = await expander.expand_context(sample_chunks)

        # Should have initial 2 chunks + 2 from Ultramarines + 2 from Primarch + 1 from Imperium
        assert len(result) == 7
        assert result[0]["id"] == "chunk1"
        assert result[1]["id"] == "chunk2"

        # Check that linked chunks were added
        result_ids = {chunk["id"] for chunk in result}
        assert "ultra1" in result_ids
        assert "ultra2" in result_ids
        assert "primarch1" in result_ids
        assert "primarch2" in result_ids
        assert "imperium1" in result_ids

    @pytest.mark.asyncio
    async def test_depth1_deduplication(self, mock_vector_store, sample_linked_chunks):
        """Test that deduplication prevents duplicates."""
        # Create initial chunks that include a chunk that's also in linked results
        initial_chunks: list[ChunkData] = [
            {
                "id": "chunk1",
                "wiki_page_id": "page1",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Test content",
                "chunk_index": 0,
                "metadata": {"links": ["Ultramarines"]},
            },
            {
                "id": "ultra1",  # Duplicate ID
                "wiki_page_id": "page2",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Already have this",
                "chunk_index": 1,
                "metadata": {"links": []},
            },
        ]

        def mock_get(where, limit, include):
            if where["article_title"] == "Ultramarines":
                chunks = sample_linked_chunks["Ultramarines"][:limit]
                return {
                    "ids": [c["id"] for c in chunks],
                    "metadatas": [c["metadata"] for c in chunks],
                    "documents": [c["chunk_text"] for c in chunks],
                }
            return {"ids": [], "metadatas": [], "documents": []}

        def mock_metadata_to_chunk(chunk_id, metadata, document):
            for chunks in sample_linked_chunks.values():
                for chunk in chunks:
                    if chunk["id"] == chunk_id:
                        return chunk
            return None

        mock_vector_store.collection.get = MagicMock(side_effect=mock_get)
        mock_vector_store.metadata_to_chunk = mock_metadata_to_chunk

        expander = ContextExpander(vector_store=mock_vector_store)

        result = await expander.expand_context(initial_chunks)

        # Should have 2 initial + 1 (ultra2, but not ultra1 since it's a duplicate)
        assert len(result) == 3

        # Check IDs
        result_ids = [chunk["id"] for chunk in result]
        assert result_ids.count("ultra1") == 1  # Only once
        assert "ultra2" in result_ids

    @pytest.mark.asyncio
    async def test_depth1_no_links(self, mock_vector_store):
        """Test expansion when chunks have no links."""
        chunks_no_links: list[ChunkData] = [
            {
                "id": "chunk1",
                "wiki_page_id": "page1",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Test content",
                "chunk_index": 0,
                "metadata": {"links": []},
            },
        ]

        expander = ContextExpander(vector_store=mock_vector_store)

        result = await expander.expand_context(chunks_no_links)

        # Should only have the initial chunk
        assert len(result) == 1
        assert result[0]["id"] == "chunk1"

    @pytest.mark.asyncio
    async def test_depth1_linked_articles_not_found(self, mock_vector_store, sample_chunks):
        """Test expansion when linked articles not found in vector DB."""

        def mock_get(where, limit, include):
            # Return empty results for all queries
            return {"ids": [], "metadatas": [], "documents": []}

        mock_vector_store.collection.get = MagicMock(side_effect=mock_get)

        expander = ContextExpander(vector_store=mock_vector_store)

        result = await expander.expand_context(sample_chunks)

        # Should only have the initial chunks
        assert len(result) == 2
        assert result[0]["id"] == "chunk1"
        assert result[1]["id"] == "chunk2"


class TestExpandContextDepth2:
    """Tests for expand_context with depth=2."""

    @pytest.mark.asyncio
    async def test_depth2_expansion(self, mock_vector_store, sample_chunks, sample_linked_chunks):
        """Test depth=2 expansion."""

        def mock_get(where, limit, include):
            article_title = where["article_title"]
            chunks = sample_linked_chunks.get(article_title, [])[:limit]

            if not chunks:
                return {"ids": [], "metadatas": [], "documents": []}

            return {
                "ids": [c["id"] for c in chunks],
                "metadatas": [c["metadata"] for c in chunks],
                "documents": [c["chunk_text"] for c in chunks],
            }

        def mock_metadata_to_chunk(chunk_id, metadata, document):
            for chunks in sample_linked_chunks.values():
                for chunk in chunks:
                    if chunk["id"] == chunk_id:
                        return chunk
            return None

        mock_vector_store.collection.get = MagicMock(side_effect=mock_get)
        mock_vector_store.metadata_to_chunk = mock_metadata_to_chunk

        expander = ContextExpander(vector_store=mock_vector_store, expansion_depth=2)

        result = await expander.expand_context(sample_chunks)

        # Should have:
        # - 2 initial chunks
        # - Depth 1: 2 Ultramarines + 2 Primarch + 1 Imperium = 5
        # - Depth 2: 1 Space Marines + 1 Emperor = 2
        # Total: 9 chunks
        assert len(result) >= 8  # At least depth 1 + some depth 2

        result_ids = {chunk["id"] for chunk in result}

        # Check depth-1 chunks are present
        assert "ultra1" in result_ids
        assert "primarch1" in result_ids

        # Check depth-2 chunks are present (from links in depth-1 chunks)
        assert "sm1" in result_ids or "emperor1" in result_ids


class TestExpandContextMaxChunksLimit:
    """Tests for max chunks limit enforcement."""

    @pytest.mark.asyncio
    async def test_max_chunks_limit_enforced(
        self, mock_vector_store, sample_chunks, sample_linked_chunks
    ):
        """Test that max chunks limit is enforced."""

        def mock_get(where, limit, include):
            article_title = where["article_title"]
            chunks = sample_linked_chunks.get(article_title, [])[:limit]

            if not chunks:
                return {"ids": [], "metadatas": [], "documents": []}

            return {
                "ids": [c["id"] for c in chunks],
                "metadatas": [c["metadata"] for c in chunks],
                "documents": [c["chunk_text"] for c in chunks],
            }

        def mock_metadata_to_chunk(chunk_id, metadata, document):
            for chunks in sample_linked_chunks.values():
                for chunk in chunks:
                    if chunk["id"] == chunk_id:
                        return chunk
            return None

        mock_vector_store.collection.get = MagicMock(side_effect=mock_get)
        mock_vector_store.metadata_to_chunk = mock_metadata_to_chunk

        # Set very low max_chunks limit
        expander = ContextExpander(vector_store=mock_vector_store, expansion_depth=2, max_chunks=5)

        result = await expander.expand_context(sample_chunks)

        # Should not exceed max_chunks
        assert len(result) <= 5


class TestExtractLinks:
    """Tests for _extract_links helper method."""

    def test_extract_links_from_chunks(self, mock_vector_store):
        """Test extracting links from chunks."""
        chunks: list[ChunkData] = [
            {
                "id": "chunk1",
                "wiki_page_id": "page1",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Test",
                "chunk_index": 0,
                "metadata": {"links": ["Link1", "Link2"]},
            },
            {
                "id": "chunk2",
                "wiki_page_id": "page1",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Test",
                "chunk_index": 1,
                "metadata": {"links": ["Link2", "Link3"]},
            },
        ]

        expander = ContextExpander(vector_store=mock_vector_store)
        links = expander._extract_links(chunks)

        # Should return unique links in order
        assert links == ["Link1", "Link2", "Link3"]

    def test_extract_links_no_metadata(self, mock_vector_store):
        """Test extracting links when chunks have no metadata."""
        chunks: list[ChunkData] = [
            {
                "id": "chunk1",
                "wiki_page_id": "page1",
                "article_title": "Test",
                "section_path": "Overview",
                "chunk_text": "Test",
                "chunk_index": 0,
                "metadata": {},
            },
        ]

        expander = ContextExpander(vector_store=mock_vector_store)
        links = expander._extract_links(chunks)

        assert links == []


class TestExpandContextInvalidDepth:
    """Tests for invalid expansion depth."""

    @pytest.mark.asyncio
    async def test_invalid_depth_raises_error(self, mock_vector_store, sample_chunks):
        """Test that invalid expansion depth raises ValidationError."""
        expander = ContextExpander(vector_store=mock_vector_store)

        with pytest.raises(ValidationError, match="Expansion depth must be 0-2"):
            await expander.expand_context(sample_chunks, expansion_depth=5)
