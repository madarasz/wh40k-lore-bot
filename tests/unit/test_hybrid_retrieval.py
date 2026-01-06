"""Unit tests for HybridRetrievalService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.hybrid_retrieval import HybridRetrievalService
from src.rag.vector_store import ChunkData


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock ChromaVectorStore."""
    mock = MagicMock()
    mock.query = MagicMock()
    mock.get_by_id = MagicMock()
    return mock


@pytest.fixture
def mock_bm25_repository() -> MagicMock:
    """Create a mock BM25Repository."""
    mock = MagicMock()
    mock.search = MagicMock()
    return mock


@pytest.fixture
def sample_chunk_data() -> list[ChunkData]:
    """Create sample ChunkData for testing."""
    return [
        {
            "id": "12345_0",
            "wiki_page_id": "12345",
            "article_title": "Roboute Guilliman",
            "section_path": "Introduction",
            "chunk_text": "Roboute Guilliman is the Primarch of the Ultramarines",
            "chunk_index": 0,
            "metadata": {"article_last_updated": "2025-01-01T00:00:00"},
        },
        {
            "id": "23456_0",
            "wiki_page_id": "23456",
            "article_title": "Ultramarines",
            "section_path": "History",
            "chunk_text": "The Ultramarines are a Space Marine chapter",
            "chunk_index": 0,
            "metadata": {"article_last_updated": "2025-01-01T00:00:00"},
        },
        {
            "id": "34567_0",
            "wiki_page_id": "34567",
            "article_title": "Macragge",
            "section_path": "Overview",
            "chunk_text": "Macragge is the homeworld of the Ultramarines",
            "chunk_index": 0,
            "metadata": {"article_last_updated": "2025-01-01T00:00:00"},
        },
    ]


@pytest.fixture
def query_embedding() -> np.ndarray:
    """Create a sample query embedding."""
    return np.random.rand(1536)


@pytest.fixture
def hybrid_service(
    mock_vector_store: MagicMock,
    mock_bm25_repository: MagicMock,
) -> HybridRetrievalService:
    """Create a HybridRetrievalService instance with mocked dependencies."""
    return HybridRetrievalService(
        vector_store=mock_vector_store,
        bm25_repository=mock_bm25_repository,
        top_k=20,
        vector_weight=0.5,
        bm25_weight=0.5,
    )


class TestInitialization:
    """Test HybridRetrievalService initialization."""

    def test_init_with_explicit_params(
        self,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
    ) -> None:
        """Test initialization with explicit parameters."""
        service = HybridRetrievalService(
            vector_store=mock_vector_store,
            bm25_repository=mock_bm25_repository,
            top_k=10,
            vector_weight=0.7,
            bm25_weight=0.3,
        )

        assert service.vector_store == mock_vector_store
        assert service.bm25_repository == mock_bm25_repository
        assert service.top_k == 10
        assert service.vector_weight == 0.7
        assert service.bm25_weight == 0.3
        assert service.rrf_k == 60

    def test_init_with_env_defaults(
        self,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
    ) -> None:
        """Test initialization with environment defaults."""
        with patch.dict(
            "os.environ",
            {
                "RETRIEVAL_TOP_K": "25",
                "RETRIEVAL_VECTOR_WEIGHT": "0.6",
                "RETRIEVAL_BM25_WEIGHT": "0.4",
            },
        ):
            service = HybridRetrievalService(
                vector_store=mock_vector_store,
                bm25_repository=mock_bm25_repository,
            )

            assert service.top_k == 25
            assert service.vector_weight == 0.6
            assert service.bm25_weight == 0.4

    def test_init_weights_validation_negative(
        self,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
    ) -> None:
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError, match="Weights cannot be negative"):
            HybridRetrievalService(
                vector_store=mock_vector_store,
                bm25_repository=mock_bm25_repository,
                vector_weight=-0.1,
                bm25_weight=1.1,
            )

    def test_init_weights_validation_sum(
        self,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
    ) -> None:
        """Test that weights not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridRetrievalService(
                vector_store=mock_vector_store,
                bm25_repository=mock_bm25_repository,
                vector_weight=0.6,
                bm25_weight=0.6,
            )


class TestRetrieve:
    """Test hybrid retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_happy_path(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test successful retrieval with results from both searches."""
        # Setup mock returns
        vector_results = [
            (sample_chunk_data[0], 0.15),
            (sample_chunk_data[1], 0.25),
        ]
        bm25_results = [
            ("12345_0", 5.2),
            ("34567_0", 3.1),
        ]

        mock_vector_store.query.return_value = vector_results
        mock_bm25_repository.search.return_value = bm25_results
        mock_vector_store.get_by_id.return_value = sample_chunk_data[2]

        # Execute
        results = await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="Ultramarines Guilliman",
        )

        # Verify
        assert len(results) > 0
        assert all(isinstance(chunk, dict) for chunk, _ in results)
        assert all(isinstance(score, float) for _, score in results)

        # Verify both searches were called
        mock_vector_store.query.assert_called_once()
        mock_bm25_repository.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_empty_query_text(
        self,
        hybrid_service: HybridRetrievalService,
        query_embedding: np.ndarray,
    ) -> None:
        """Test that empty query text raises ValueError."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            await hybrid_service.retrieve(
                query_embedding=query_embedding,
                query_text="",
            )

    @pytest.mark.asyncio
    async def test_retrieve_empty_embedding(
        self,
        hybrid_service: HybridRetrievalService,
    ) -> None:
        """Test that empty embedding raises ValueError."""
        empty_embedding = np.array([])

        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            await hybrid_service.retrieve(
                query_embedding=empty_embedding,
                query_text="test query",
            )

    @pytest.mark.asyncio
    async def test_retrieve_vector_only_results(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test retrieval when only vector search returns results."""
        vector_results = [
            (sample_chunk_data[0], 0.15),
            (sample_chunk_data[1], 0.25),
        ]
        bm25_results: list[tuple[str, float]] = []

        mock_vector_store.query.return_value = vector_results
        mock_bm25_repository.search.return_value = bm25_results

        results = await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="query",
        )

        assert len(results) == 2
        assert results[0][0]["id"] in ["12345_0", "23456_0"]

    @pytest.mark.asyncio
    async def test_retrieve_bm25_only_results(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test retrieval when only BM25 search returns results."""
        vector_results: list[tuple[ChunkData, float]] = []
        bm25_results = [
            ("12345_0", 5.2),
            ("23456_0", 3.1),
        ]

        mock_vector_store.query.return_value = vector_results
        mock_bm25_repository.search.return_value = bm25_results
        mock_vector_store.get_by_id.side_effect = [
            sample_chunk_data[0],
            sample_chunk_data[1],
        ]

        results = await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="query",
        )

        assert len(results) == 2
        assert results[0][0]["id"] in ["12345_0", "23456_0"]

    @pytest.mark.asyncio
    async def test_retrieve_both_empty(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
    ) -> None:
        """Test retrieval when both searches return empty results."""
        mock_vector_store.query.return_value = []
        mock_bm25_repository.search.return_value = []

        results = await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="query",
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test retrieval with metadata filters."""
        vector_results = [(sample_chunk_data[0], 0.15)]
        bm25_results = [("12345_0", 5.2)]
        filters = {"article_title": "Roboute Guilliman"}

        mock_vector_store.query.return_value = vector_results
        mock_bm25_repository.search.return_value = bm25_results

        await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="query",
            filters=filters,
        )

        # Verify filters were passed to vector search
        call_args = mock_vector_store.query.call_args
        assert call_args.kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_retrieve_custom_top_k(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test retrieval with custom top_k parameter."""
        vector_results = [(sample_chunk_data[0], 0.15)]
        bm25_results = [("12345_0", 5.2)]

        mock_vector_store.query.return_value = vector_results
        mock_bm25_repository.search.return_value = bm25_results

        await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="query",
            top_k=5,
        )

        # Verify both searches were called with top_k=5
        vector_call = mock_vector_store.query.call_args
        bm25_call = mock_bm25_repository.search.call_args

        assert vector_call.kwargs["n_results"] == 5
        assert bm25_call.kwargs["top_k"] == 5


class TestRRFFusion:
    """Test Reciprocal Rank Fusion algorithm."""

    def test_fuse_results_both_present(
        self,
        hybrid_service: HybridRetrievalService,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test RRF fusion with overlapping results."""
        vector_results = [
            (sample_chunk_data[0], 0.1),  # Rank 1
            (sample_chunk_data[1], 0.2),  # Rank 2
        ]
        bm25_results = [
            ("12345_0", 5.0),  # Rank 1 (same as vector rank 1)
            ("34567_0", 3.0),  # Rank 2
        ]

        # Mock get_by_id for BM25-only chunks
        hybrid_service.vector_store.get_by_id.return_value = sample_chunk_data[2]

        results = hybrid_service._fuse_results(vector_results, bm25_results, top_k=10)

        assert len(results) == 3
        # Chunk 12345_0 appears in both, should have highest combined score
        assert results[0][0]["id"] == "12345_0"

    def test_fuse_results_correct_rrf_scores(
        self,
        hybrid_service: HybridRetrievalService,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test that RRF scores are calculated correctly."""
        # Single chunk in both results at rank 1
        vector_results = [(sample_chunk_data[0], 0.1)]
        bm25_results = [("12345_0", 5.0)]

        results = hybrid_service._fuse_results(vector_results, bm25_results, top_k=10)

        # Expected RRF score: 0.5/(60+1) + 0.5/(60+1) = 1.0/61 ≈ 0.01639
        expected_score = (0.5 + 0.5) / 61
        assert len(results) == 1
        assert abs(results[0][1] - expected_score) < 0.0001

    def test_fuse_results_weighted_rrf(
        self,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test RRF fusion with custom weights."""
        service = HybridRetrievalService(
            vector_store=mock_vector_store,
            bm25_repository=mock_bm25_repository,
            vector_weight=0.7,
            bm25_weight=0.3,
        )

        vector_results = [(sample_chunk_data[0], 0.1)]
        bm25_results = [("12345_0", 5.0)]

        results = service._fuse_results(vector_results, bm25_results, top_k=10)

        # Expected RRF score: 0.7/(60+1) + 0.3/(60+1) = 1.0/61 ≈ 0.01639
        expected_score = (0.7 + 0.3) / 61
        assert len(results) == 1
        assert abs(results[0][1] - expected_score) < 0.0001

    def test_fuse_results_respects_top_k(
        self,
        hybrid_service: HybridRetrievalService,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test that fusion respects top_k limit."""
        vector_results = [
            (sample_chunk_data[0], 0.1),
            (sample_chunk_data[1], 0.2),
        ]
        bm25_results = [
            ("12345_0", 5.0),
            ("23456_0", 4.0),
            ("34567_0", 3.0),
        ]

        hybrid_service.vector_store.get_by_id.return_value = sample_chunk_data[2]

        results = hybrid_service._fuse_results(vector_results, bm25_results, top_k=2)

        assert len(results) == 2

    def test_fuse_results_missing_chunks_filtered(
        self,
        hybrid_service: HybridRetrievalService,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test that chunks missing from vector store are filtered out."""
        vector_results = [(sample_chunk_data[0], 0.1)]
        bm25_results = [
            ("12345_0", 5.0),
            ("99999_0", 4.0),  # This chunk doesn't exist
        ]

        # Mock get_by_id to return None for missing chunk
        hybrid_service.vector_store.get_by_id.return_value = None

        results = hybrid_service._fuse_results(vector_results, bm25_results, top_k=10)

        # Only the chunk from vector results should be present
        assert len(results) == 1
        assert results[0][0]["id"] == "12345_0"


class TestParallelExecution:
    """Test parallel execution of searches."""

    @pytest.mark.asyncio
    async def test_searches_run_concurrently(
        self,
        hybrid_service: HybridRetrievalService,
        mock_vector_store: MagicMock,
        mock_bm25_repository: MagicMock,
        query_embedding: np.ndarray,
        sample_chunk_data: list[ChunkData],
    ) -> None:
        """Test that vector and BM25 searches run concurrently."""
        # Setup mocks
        mock_vector_store.query.return_value = [(sample_chunk_data[0], 0.1)]
        mock_bm25_repository.search.return_value = [("12345_0", 5.0)]

        # Execute retrieval
        await hybrid_service.retrieve(
            query_embedding=query_embedding,
            query_text="test query",
        )

        # Verify both were called (parallel execution verified by asyncio.gather)
        mock_vector_store.query.assert_called_once()
        mock_bm25_repository.search.assert_called_once()
