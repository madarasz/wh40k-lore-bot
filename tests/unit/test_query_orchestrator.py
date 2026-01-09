"""Unit tests for QueryOrchestrator."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from src.llm.structured_output import LLMStructuredResponse
from src.orchestration.query_orchestrator import (
    DEFAULT_BOT_PERSONALITY,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    QueryOrchestrator,
    QueryRequest,
    QueryResponse,
    RetrievalMetadata,
    RetrievalResult,
)
from src.rag.vector_store import ChunkData
from src.utils.exceptions import (
    ConfigurationError,
    LLMProviderError,
    RetrievalError,
)


class TestQueryRequest:
    """Tests for QueryRequest dataclass."""

    def test_query_request_basic(self) -> None:
        """Test basic QueryRequest creation."""
        request = QueryRequest(query_text="Who is Guilliman?")
        assert request.query_text == "Who is Guilliman?"
        assert request.user_id is None
        assert request.server_id is None

    def test_query_request_with_all_fields(self) -> None:
        """Test QueryRequest with all fields populated."""
        request = QueryRequest(
            query_text="Who is Guilliman?",
            user_id="user123",
            server_id="server456",
        )
        assert request.query_text == "Who is Guilliman?"
        assert request.user_id == "user123"
        assert request.server_id == "server456"


class TestQueryResponse:
    """Tests for QueryResponse dataclass."""

    def test_query_response_basic(self) -> None:
        """Test basic QueryResponse creation."""
        response = QueryResponse(
            answer="Guilliman is a Primarch.",
            personality_reply="The Emperor protects.",
            sources=["https://wh40k.fandom.com/wiki/Roboute_Guilliman"],
            smalltalk=False,
            language="EN",
            metadata={"latency_ms": 1000},
        )
        assert response.answer == "Guilliman is a Primarch."
        assert response.personality_reply == "The Emperor protects."
        assert len(response.sources) == 1
        assert response.smalltalk is False
        assert response.language == "EN"
        assert response.error is None

    def test_query_response_with_error(self) -> None:
        """Test QueryResponse with error."""
        response = QueryResponse(
            answer="",
            personality_reply="",
            sources=[],
            smalltalk=False,
            language="EN",
            metadata={"latency_ms": 100},
            error="Query too short.",
        )
        assert response.error == "Query too short."


@pytest.fixture
def mock_embedding_generator():
    """Create mock EmbeddingGenerator."""
    generator = MagicMock()
    generator.generate_embeddings = MagicMock(
        return_value=[np.array([0.1] * 1536, dtype=np.float32)]
    )
    return generator


@pytest.fixture
def mock_hybrid_retrieval():
    """Create mock HybridRetrievalService."""
    service = MagicMock()
    service.retrieve = AsyncMock()
    return service


@pytest.fixture
def mock_context_expander():
    """Create mock ContextExpander."""
    expander = MagicMock()
    expander.expand_context = AsyncMock()
    return expander


@pytest.fixture
def mock_llm_router():
    """Create mock MultiLLMRouter."""
    router = MagicMock()
    router.generate_structured = AsyncMock()
    router.default_model = "claude-sonnet-4-5"
    return router


@pytest.fixture
def mock_response_formatter():
    """Create mock ResponseFormatter."""
    formatter = MagicMock()
    formatter.format_cli_response = MagicMock(return_value="Formatted response")
    return formatter


@pytest.fixture
def sample_chunks() -> list[ChunkData]:
    """Create sample chunks for testing."""
    return [
        {
            "id": "chunk1",
            "wiki_page_id": "page1",
            "article_title": "Roboute Guilliman",
            "section_path": "Biography",
            "chunk_text": "Roboute Guilliman is the Primarch of the Ultramarines.",
            "chunk_index": 0,
            "metadata": {
                "links": ["Ultramarines"],
                "article_last_updated": "2024-01-01",
            },
        },
        {
            "id": "chunk2",
            "wiki_page_id": "page2",
            "article_title": "Ultramarines",
            "section_path": "Overview",
            "chunk_text": "The Ultramarines are a Space Marine chapter.",
            "chunk_index": 0,
            "metadata": {
                "links": [],
                "article_last_updated": "2024-01-01",
            },
        },
    ]


@pytest.fixture
def sample_retrieval_results(sample_chunks: list[ChunkData]) -> list[tuple[ChunkData, float]]:
    """Create sample retrieval results with scores."""
    return [(sample_chunks[0], 0.85), (sample_chunks[1], 0.75)]


@pytest.fixture
def mock_lore_response() -> LLMStructuredResponse:
    """Create mock LLM response for lore question."""
    return LLMStructuredResponse(
        answer="Roboute Guilliman is the Primarch of the Ultramarines Legion.",
        personality_reply="The Emperor protects.",
        sources=["https://warhammer40k.fandom.com/wiki/Roboute_Guilliman"],
        smalltalk=False,
        language="EN",
    )


@pytest.fixture
def mock_smalltalk_response() -> LLMStructuredResponse:
    """Create mock LLM response for smalltalk."""
    return LLMStructuredResponse(
        answer=None,
        personality_reply="Greetings, citizen. How may I serve the Imperium?",
        sources=None,
        smalltalk=True,
        language="EN",
    )


@pytest.fixture
def orchestrator(
    mock_embedding_generator,
    mock_hybrid_retrieval,
    mock_context_expander,
    mock_llm_router,
    mock_response_formatter,
) -> QueryOrchestrator:
    """Create QueryOrchestrator with all mocked dependencies."""
    return QueryOrchestrator(
        embedding_generator=mock_embedding_generator,
        hybrid_retrieval=mock_hybrid_retrieval,
        context_expander=mock_context_expander,
        llm_router=mock_llm_router,
        response_formatter=mock_response_formatter,
    )


class TestQueryOrchestratorInit:
    """Tests for QueryOrchestrator initialization."""

    def test_orchestrator_initialization(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
        mock_llm_router,
        mock_response_formatter,
    ) -> None:
        """Test orchestrator initializes with defaults when env vars not set."""
        # Clear any env vars that might be set by .env file
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            # Also need to patch individual env vars since they may be cached
            with patch("os.getenv") as mock_getenv:
                mock_getenv.side_effect = lambda key, default=None: default
                orchestrator = QueryOrchestrator(
                    embedding_generator=mock_embedding_generator,
                    hybrid_retrieval=mock_hybrid_retrieval,
                    context_expander=mock_context_expander,
                    llm_router=mock_llm_router,
                    response_formatter=mock_response_formatter,
                )
                assert orchestrator.timeout_seconds == DEFAULT_QUERY_TIMEOUT_SECONDS
                assert orchestrator.personality == DEFAULT_BOT_PERSONALITY

    def test_orchestrator_custom_config(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
        mock_llm_router,
        mock_response_formatter,
    ) -> None:
        """Test orchestrator with custom configuration."""
        with patch.dict(
            "os.environ",
            {
                "QUERY_TIMEOUT_SECONDS": "20",
                "BOT_PERSONALITY": "grimdark",
            },
        ):
            orchestrator = QueryOrchestrator(
                embedding_generator=mock_embedding_generator,
                hybrid_retrieval=mock_hybrid_retrieval,
                context_expander=mock_context_expander,
                llm_router=mock_llm_router,
                response_formatter=mock_response_formatter,
            )
            assert orchestrator.timeout_seconds == 20
            assert orchestrator.personality == "grimdark"


class TestSystemPromptBuilding:
    """Tests for system prompt building with language detection."""

    def test_build_default_system_prompt(self, orchestrator: QueryOrchestrator) -> None:
        """Test building default personality prompt includes language detection."""
        prompt = orchestrator._build_system_prompt()
        assert "expert in warhammer 40,000 lore" in prompt.lower()
        assert "## Language Detection" in prompt
        assert '"language": "HU" or "EN"' in prompt

    def test_build_grimdark_system_prompt(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
        mock_llm_router,
        mock_response_formatter,
    ) -> None:
        """Test building grimdark persona prompt includes language detection."""
        with patch.dict("os.environ", {"BOT_PERSONALITY": "grimdark"}):
            orchestrator = QueryOrchestrator(
                embedding_generator=mock_embedding_generator,
                hybrid_retrieval=mock_hybrid_retrieval,
                context_expander=mock_context_expander,
                llm_router=mock_llm_router,
                response_formatter=mock_response_formatter,
            )
            prompt = orchestrator._build_system_prompt()
            assert "grimdark narrator" in prompt.lower()
            assert "## Language Detection" in prompt

    def test_system_prompt_contains_hungarian_detection(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test system prompt instructs Hungarian detection."""
        prompt = orchestrator._build_system_prompt()
        assert "Hungarian" in prompt
        assert 'language="HU"' in prompt

    def test_system_prompt_contains_english_default(self, orchestrator: QueryOrchestrator) -> None:
        """Test system prompt sets English as default for non-Hungarian."""
        prompt = orchestrator._build_system_prompt()
        assert 'language="EN"' in prompt
        assert "other languages" in prompt.lower()


class TestContextBuilding:
    """Tests for context building from chunks."""

    def test_build_context_basic(
        self,
        orchestrator: QueryOrchestrator,
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test building context from chunks."""
        context = orchestrator._build_context(sample_chunks)
        assert "Roboute Guilliman" in context
        assert "Biography" in context
        assert "Primarch of the Ultramarines" in context
        assert "Ultramarines" in context

    def test_build_context_empty_chunks(self, orchestrator: QueryOrchestrator) -> None:
        """Test building context with empty list."""
        context = orchestrator._build_context([])
        assert context == ""

    def test_build_context_infobox_section(self, orchestrator: QueryOrchestrator) -> None:
        """Test building context hides Infobox section path."""
        chunks: list[ChunkData] = [
            {
                "id": "chunk1",
                "wiki_page_id": "page1",
                "article_title": "Test Article",
                "section_path": "Infobox",
                "chunk_text": "Infobox content here.",
                "chunk_index": 0,
                "metadata": {},
            }
        ]
        context = orchestrator._build_context(chunks)
        assert "[Test Article]" in context
        assert "Infobox" not in context.split("\n")[0]


class TestProcessHappyPath:
    """Tests for happy path processing."""

    async def test_process_lore_question_success(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_lore_response: LLMStructuredResponse,
    ) -> None:
        """Test successful processing of a lore question."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_lore_response

        # Process
        request = QueryRequest(query_text="Who is Guilliman?", user_id="user1")
        response = await orchestrator.process(request)

        # Verify
        assert response.error is None
        assert response.answer == mock_lore_response.answer
        assert response.personality_reply == mock_lore_response.personality_reply
        assert response.smalltalk is False
        assert len(response.sources) == 1
        assert "latency_ms" in response.metadata

    async def test_process_smalltalk_success(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_smalltalk_response: LLMStructuredResponse,
    ) -> None:
        """Test successful processing of smalltalk."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_smalltalk_response

        # Process
        request = QueryRequest(query_text="Hello there!")
        response = await orchestrator.process(request)

        # Verify
        assert response.error is None
        assert response.answer == ""
        assert response.personality_reply == mock_smalltalk_response.personality_reply
        assert response.smalltalk is True
        assert len(response.sources) == 0

    async def test_process_metadata_populated(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_lore_response: LLMStructuredResponse,
    ) -> None:
        """Test that metadata is correctly populated."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_lore_response

        # Process
        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        # Verify metadata
        assert "latency_ms" in response.metadata
        assert "embedding_ms" in response.metadata
        assert "retrieval_ms" in response.metadata
        assert "expansion_ms" in response.metadata
        assert "llm_ms" in response.metadata
        assert "chunks_retrieved" in response.metadata
        assert "chunks_expanded" in response.metadata
        assert response.metadata["chunks_retrieved"] == 2
        assert response.metadata["chunks_expanded"] == 2


class TestProcessErrors:
    """Tests for error handling in process method."""

    async def test_process_embedding_error(
        self,
        orchestrator: QueryOrchestrator,
    ) -> None:
        """Test handling of embedding generation error."""
        # Mock embedding failure
        orchestrator.embedding_generator.generate_embeddings.return_value = [None]

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "retrieve" in response.error.lower() or "error" in response.error.lower()

    async def test_process_retrieval_error(
        self,
        orchestrator: QueryOrchestrator,
    ) -> None:
        """Test handling of retrieval error."""
        # Mock retrieval failure
        orchestrator.hybrid_retrieval.retrieve.side_effect = RetrievalError("Search failed")

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "retrieve" in response.error.lower()

    async def test_process_llm_provider_error(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test handling of LLM provider error."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.side_effect = LLMProviderError("API error")

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "generate" in response.error.lower() or "response" in response.error.lower()

    async def test_process_llm_validation_error(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test handling of LLM validation error (invalid JSON schema)."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        validation_error = PydanticValidationError.from_exception_data(
            "LLMStructuredResponse",
            [{"type": "missing", "loc": ("answer",), "msg": "Field required"}],
        )
        orchestrator.llm_router.generate_structured.side_effect = validation_error

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "invalid" in response.error.lower() or "format" in response.error.lower()

    async def test_process_unexpected_error(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
    ) -> None:
        """Test handling of unexpected error."""
        # Mock unexpected error
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.side_effect = RuntimeError(
            "Unexpected failure"
        )

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "unexpected" in response.error.lower()

    async def test_process_timeout_error(
        self,
        orchestrator: QueryOrchestrator,
    ) -> None:
        """Test handling of timeout."""
        # Mock timeout
        orchestrator.hybrid_retrieval.retrieve.side_effect = TimeoutError()

        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        assert response.error is not None
        assert "timed out" in response.error.lower()


class TestProcessWithLanguageDetection:
    """Tests for language detection from LLM response."""

    async def test_process_returns_answer_in_detected_english(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_lore_response: LLMStructuredResponse,
    ) -> None:
        """Test processing returns answer with LLM-detected English language."""
        # Setup mocks - mock_lore_response has language="EN"
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_lore_response

        # Process
        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        # Verify response contains the English answer
        assert response.answer == mock_lore_response.answer
        assert response.error is None

    async def test_process_returns_answer_in_detected_hungarian(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test processing returns answer with LLM-detected Hungarian language."""
        # Create Hungarian response
        hungarian_response = LLMStructuredResponse(
            answer="Roboute Guilliman az Ultramarines Legion primarcha.",
            personality_reply="A Csaszar ved.",
            sources=["https://warhammer40k.fandom.com/wiki/Roboute_Guilliman"],
            smalltalk=False,
            language="HU",
        )

        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = hungarian_response

        # Process
        request = QueryRequest(query_text="Ki az Guilliman?")
        response = await orchestrator.process(request)

        # Verify response contains the Hungarian answer
        assert response.answer == hungarian_response.answer
        assert response.error is None


class TestErrorResponse:
    """Tests for error response creation."""

    def test_error_response_creation(self, orchestrator: QueryOrchestrator) -> None:
        """Test error response has correct structure."""
        start_time = time.perf_counter()
        response = orchestrator._error_response("Test error message", start_time)

        assert response.answer == ""
        assert response.personality_reply == ""
        assert response.sources == []
        assert response.smalltalk is False
        assert response.error == "Test error message"
        assert "latency_ms" in response.metadata


class TestSourceUrlConversion:
    """Tests for source URL conversion."""

    async def test_http_url_to_string_conversion(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test Pydantic HttpUrl is converted to string."""
        # Create response with multiple sources
        mock_response = LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test reply",
            sources=[
                "https://warhammer40k.fandom.com/wiki/Guilliman",
                "https://warhammer40k.fandom.com/wiki/Ultramarines",
            ],
            smalltalk=False,
            language="EN",
        )

        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_response

        # Process
        request = QueryRequest(query_text="Who is Guilliman?")
        response = await orchestrator.process(request)

        # Verify sources are strings
        assert all(isinstance(url, str) for url in response.sources)
        assert len(response.sources) == 2


class TestPipelineExecution:
    """Tests for pipeline execution order."""

    async def test_retrieval_always_executes(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_smalltalk_response: LLMStructuredResponse,
    ) -> None:
        """Test that retrieval executes even for potential smalltalk."""
        # Setup mocks
        orchestrator.hybrid_retrieval.retrieve.return_value = sample_retrieval_results
        orchestrator.context_expander.expand_context.return_value = sample_chunks
        orchestrator.llm_router.generate_structured.return_value = mock_smalltalk_response

        # Process a greeting (potential smalltalk)
        request = QueryRequest(query_text="Hello there!")
        await orchestrator.process(request)

        # Verify retrieval was still called
        orchestrator.hybrid_retrieval.retrieve.assert_called_once()
        orchestrator.context_expander.expand_context.assert_called_once()

    async def test_pipeline_calls_in_order(
        self,
        orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
        mock_lore_response: LLMStructuredResponse,
    ) -> None:
        """Test that pipeline components are called in correct order."""
        # Track call order
        call_order = []

        def track_embedding(*args, **kwargs):
            call_order.append("embedding")
            return [np.array([0.1] * 1536, dtype=np.float32)]

        async def track_retrieval(*args, **kwargs):
            call_order.append("retrieval")
            return sample_retrieval_results

        async def track_expansion(*args, **kwargs):
            call_order.append("expansion")
            return sample_chunks

        async def track_llm(*args, **kwargs):
            call_order.append("llm")
            return mock_lore_response

        # Setup mocks
        orchestrator.embedding_generator.generate_embeddings = track_embedding
        orchestrator.hybrid_retrieval.retrieve = track_retrieval
        orchestrator.context_expander.expand_context = track_expansion
        orchestrator.llm_router.generate_structured = track_llm

        # Process
        request = QueryRequest(query_text="Who is Guilliman?")
        await orchestrator.process(request)

        # Verify order
        assert call_order == ["embedding", "retrieval", "expansion", "llm"]


class TestRetrievalDataclasses:
    """Tests for RetrievalResult and RetrievalMetadata dataclasses."""

    def test_retrieval_metadata_creation(self) -> None:
        """Test RetrievalMetadata creation."""
        metadata = RetrievalMetadata(
            latency_ms=150,
            embedding_ms=20,
            retrieval_ms=100,
            expansion_ms=30,
            initial_count=10,
            expanded_count=15,
        )
        assert metadata.latency_ms == 150
        assert metadata.embedding_ms == 20
        assert metadata.retrieval_ms == 100
        assert metadata.expansion_ms == 30
        assert metadata.initial_count == 10
        assert metadata.expanded_count == 15

    def test_retrieval_result_creation(
        self,
        sample_retrieval_results: list[tuple[ChunkData, float]],
    ) -> None:
        """Test RetrievalResult creation."""
        metadata = RetrievalMetadata(
            latency_ms=150,
            embedding_ms=20,
            retrieval_ms=100,
            expansion_ms=30,
            initial_count=2,
            expanded_count=2,
        )
        result = RetrievalResult(chunks=sample_retrieval_results, metadata=metadata)
        assert len(result.chunks) == 2
        assert result.metadata.latency_ms == 150


class TestRetrieveOnly:
    """Tests for retrieve_only method."""

    @pytest.fixture
    def retrieval_only_orchestrator(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
    ) -> QueryOrchestrator:
        """Create QueryOrchestrator without LLM services."""
        return QueryOrchestrator(
            embedding_generator=mock_embedding_generator,
            hybrid_retrieval=mock_hybrid_retrieval,
            context_expander=mock_context_expander,
            llm_router=None,
            response_formatter=None,
        )

    async def test_retrieve_only_success(
        self,
        retrieval_only_orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test successful retrieval-only operation."""
        # Setup mocks
        retrieval_only_orchestrator.hybrid_retrieval.retrieve.return_value = (
            sample_retrieval_results
        )
        retrieval_only_orchestrator.context_expander.expand_context.return_value = sample_chunks

        # Execute
        result = await retrieval_only_orchestrator.retrieve_only("Who is Guilliman?")

        # Verify
        assert len(result.chunks) == 2
        assert result.metadata.latency_ms >= 0  # May be 0 in fast tests
        assert result.metadata.initial_count == 2
        assert result.metadata.expanded_count == 2

    async def test_retrieve_only_with_expansion(
        self,
        retrieval_only_orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test retrieve_only with context expansion adding chunks."""
        # Create expanded chunk
        expanded_chunk: ChunkData = {
            "id": "expanded_chunk",
            "wiki_page_id": "page3",
            "article_title": "Expanded Article",
            "section_path": "Overview",
            "chunk_text": "Expanded content.",
            "chunk_index": 0,
            "metadata": {},
        }
        expanded_chunks = sample_chunks + [expanded_chunk]

        # Setup mocks
        retrieval_only_orchestrator.hybrid_retrieval.retrieve.return_value = (
            sample_retrieval_results
        )
        retrieval_only_orchestrator.context_expander.expand_context.return_value = expanded_chunks

        # Execute
        result = await retrieval_only_orchestrator.retrieve_only("Who is Guilliman?")

        # Verify expansion
        assert result.metadata.initial_count == 2
        assert result.metadata.expanded_count == 3
        assert result.chunks[2][1] == 0.0  # Expanded chunk has score 0.0

    async def test_retrieve_only_with_top_k(
        self,
        retrieval_only_orchestrator: QueryOrchestrator,
        sample_retrieval_results: list[tuple[ChunkData, float]],
        sample_chunks: list[ChunkData],
    ) -> None:
        """Test retrieve_only respects top_k parameter."""
        # Setup mocks
        retrieval_only_orchestrator.hybrid_retrieval.retrieve.return_value = (
            sample_retrieval_results
        )
        retrieval_only_orchestrator.context_expander.expand_context.return_value = sample_chunks

        # Execute with explicit top_k
        await retrieval_only_orchestrator.retrieve_only("Who is Guilliman?", top_k=5)

        # Verify top_k passed to retrieval
        retrieval_only_orchestrator.hybrid_retrieval.retrieve.assert_called_once()
        call_kwargs = retrieval_only_orchestrator.hybrid_retrieval.retrieve.call_args.kwargs
        assert call_kwargs.get("top_k") == 5

    async def test_retrieve_only_embedding_error(
        self,
        retrieval_only_orchestrator: QueryOrchestrator,
    ) -> None:
        """Test retrieve_only raises RetrievalError on embedding failure."""
        # Mock embedding failure
        retrieval_only_orchestrator.embedding_generator.generate_embeddings.return_value = [None]

        with pytest.raises(RetrievalError, match="Failed to generate query embedding"):
            await retrieval_only_orchestrator.retrieve_only("Who is Guilliman?")


class TestProcessWithoutLLM:
    """Tests for process() without LLM services."""

    async def test_process_raises_without_llm(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
    ) -> None:
        """Test process() raises ConfigurationError without LLM services."""
        orchestrator = QueryOrchestrator(
            embedding_generator=mock_embedding_generator,
            hybrid_retrieval=mock_hybrid_retrieval,
            context_expander=mock_context_expander,
            llm_router=None,
            response_formatter=None,
        )

        request = QueryRequest(query_text="Who is Guilliman?")

        with pytest.raises(ConfigurationError, match="LLM services not configured"):
            await orchestrator.process(request)

    def test_orchestrator_logs_llm_disabled(
        self,
        mock_embedding_generator,
        mock_hybrid_retrieval,
        mock_context_expander,
    ) -> None:
        """Test orchestrator indicates LLM is disabled on init."""
        # This test verifies the orchestrator can be created without LLM services
        orchestrator = QueryOrchestrator(
            embedding_generator=mock_embedding_generator,
            hybrid_retrieval=mock_hybrid_retrieval,
            context_expander=mock_context_expander,
            llm_router=None,
            response_formatter=None,
        )
        assert orchestrator.llm_router is None
        assert orchestrator.response_formatter is None
