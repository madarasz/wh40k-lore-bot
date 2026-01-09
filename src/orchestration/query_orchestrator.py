"""Query orchestrator for coordinating the RAG pipeline with structured LLM output."""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from pydantic import ValidationError as PydanticValidationError

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.llm.base_provider import GenerationOptions
from src.llm.llm_router import MultiLLMRouter
from src.llm.response_formatter import ResponseFormatter
from src.llm.structured_output import LLMStructuredResponse
from src.rag.context_expander import ContextExpander
from src.rag.hybrid_retrieval import HybridRetrievalService
from src.rag.vector_store import ChunkData
from src.utils.exceptions import (
    ConfigurationError,
    LLMProviderError,
    RetrievalError,
)

logger = structlog.get_logger(__name__)


# Environment configuration defaults
DEFAULT_QUERY_TIMEOUT_SECONDS = 10
DEFAULT_BOT_PERSONALITY = "default"

# Prompt file paths
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
SYSTEM_PROMPT_FILE = "system.md"
USER_PROMPT_FILE = "user.md"


@dataclass
class QueryRequest:
    """Request object for query processing.

    Attributes:
        query_text: The user's question or query
        user_id: Optional user identifier for logging
        server_id: Optional server identifier for logging
    """

    query_text: str
    user_id: str | None = None
    server_id: str | None = None


@dataclass
class QueryResponse:
    """Response object from query processing.

    Attributes:
        answer: The LLM-generated answer (empty if smalltalk)
        personality_reply: Thematic closing statement (always present)
        sources: List of wiki URLs as strings
        smalltalk: True if this was a smalltalk response
        language: Detected language code ("HU" or "EN")
        metadata: Performance and cost metadata
        error: Error message if processing failed
    """

    answer: str
    personality_reply: str
    sources: list[str]
    smalltalk: bool
    language: str
    metadata: dict[str, int | float | str]
    error: str | None = None


@dataclass
class StepTimings:
    """Timing breakdown for pipeline steps."""

    embedding_ms: int = 0
    retrieval_ms: int = 0
    expansion_ms: int = 0
    llm_ms: int = 0


@dataclass
class RetrievalMetadata:
    """Performance metadata for retrieval operations.

    Attributes:
        latency_ms: Total retrieval time in milliseconds
        embedding_ms: Time spent generating query embedding
        retrieval_ms: Time spent in hybrid retrieval
        expansion_ms: Time spent in context expansion
        initial_count: Chunks before expansion
        expanded_count: Chunks after expansion
    """

    latency_ms: int
    embedding_ms: int
    retrieval_ms: int
    expansion_ms: int
    initial_count: int
    expanded_count: int


@dataclass
class RetrievalResult:
    """Result from retrieval-only pipeline (no LLM).

    Attributes:
        chunks: List of retrieved ChunkData with scores
        metadata: Performance metadata
    """

    chunks: list[tuple[ChunkData, float]]
    metadata: RetrievalMetadata


class QueryOrchestrator:
    """Central orchestrator for the RAG query pipeline.

    Coordinates embedding generation, hybrid retrieval, context expansion,
    LLM generation with structured output, and response formatting.

    Attributes:
        embedding_generator: Service for generating query embeddings
        hybrid_retrieval: Service for hybrid vector + BM25 retrieval
        context_expander: Service for expanding context with cross-references
        llm_router: Router for multi-provider LLM access
        response_formatter: Formatter for structured responses
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        hybrid_retrieval: HybridRetrievalService,
        context_expander: ContextExpander,
        llm_router: MultiLLMRouter | None = None,
        response_formatter: ResponseFormatter | None = None,
    ) -> None:
        """Initialize QueryOrchestrator with required retrieval services.

        Args:
            embedding_generator: EmbeddingGenerator instance (required)
            hybrid_retrieval: HybridRetrievalService instance (required)
            context_expander: ContextExpander instance (required)
            llm_router: MultiLLMRouter instance (optional, required for process())
            response_formatter: ResponseFormatter instance (optional, required for process())
        """
        self.embedding_generator = embedding_generator
        self.hybrid_retrieval = hybrid_retrieval
        self.context_expander = context_expander
        self.llm_router = llm_router
        self.response_formatter = response_formatter

        # Load configuration from environment
        self._load_config()

        logger.info(
            "query_orchestrator_initialized",
            timeout_seconds=self.timeout_seconds,
            personality=self.personality,
            llm_enabled=llm_router is not None,
        )

    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        self.timeout_seconds = int(
            os.getenv("QUERY_TIMEOUT_SECONDS", str(DEFAULT_QUERY_TIMEOUT_SECONDS))
        )
        self.personality = os.getenv("BOT_PERSONALITY", DEFAULT_BOT_PERSONALITY)

    async def retrieve_only(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute retrieval pipeline without LLM generation.

        Performs embedding generation, hybrid retrieval, and context expansion,
        returning raw chunks with scores. Useful for testing and debugging.

        Args:
            query_text: The query text to search for
            top_k: Number of results to retrieve (defaults to hybrid_retrieval.top_k)

        Returns:
            RetrievalResult with chunks, scores, and performance metadata

        Raises:
            RetrievalError: If embedding generation or retrieval fails
        """
        start_time = time.perf_counter()
        timings = StepTimings()

        logger.info("retrieval_only_started", query_text=query_text[:100])

        # Step 1: Embedding generation
        step_start = time.perf_counter()
        query_embedding = await self._generate_embedding(query_text)
        timings.embedding_ms = int((time.perf_counter() - step_start) * 1000)

        # Step 2: Hybrid retrieval
        step_start = time.perf_counter()
        chunks_with_scores = await self.hybrid_retrieval.retrieve(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
        )
        timings.retrieval_ms = int((time.perf_counter() - step_start) * 1000)
        initial_count = len(chunks_with_scores)

        # Step 3: Context expansion
        step_start = time.perf_counter()
        chunk_data_list = [chunk for chunk, score in chunks_with_scores]
        expanded_chunks = await self.context_expander.expand_context(chunk_data_list)
        timings.expansion_ms = int((time.perf_counter() - step_start) * 1000)

        # Reconstruct results with expanded chunks
        # Keep original scores for initial chunks, assign 0.0 for expanded chunks
        result_chunks: list[tuple[ChunkData, float]] = list(chunks_with_scores)
        for expanded_chunk in expanded_chunks[initial_count:]:
            result_chunks.append((expanded_chunk, 0.0))

        # Build metadata
        total_latency = int((time.perf_counter() - start_time) * 1000)
        metadata = RetrievalMetadata(
            latency_ms=total_latency,
            embedding_ms=timings.embedding_ms,
            retrieval_ms=timings.retrieval_ms,
            expansion_ms=timings.expansion_ms,
            initial_count=initial_count,
            expanded_count=len(result_chunks),
        )

        logger.info(
            "retrieval_only_completed",
            initial_count=initial_count,
            expanded_count=len(result_chunks),
            latency_ms=total_latency,
        )

        return RetrievalResult(chunks=result_chunks, metadata=metadata)

    async def process(self, request: QueryRequest) -> QueryResponse:  # noqa: PLR0911
        """Process a query through the full RAG pipeline including LLM.

        Args:
            request: QueryRequest with query text and optional metadata

        Returns:
            QueryResponse with answer, sources, and metadata

        Raises:
            ConfigurationError: If LLM services not configured
        """
        # Validate LLM services are available
        if self.llm_router is None or self.response_formatter is None:
            raise ConfigurationError(
                "LLM services not configured. Use retrieve_only() for retrieval-only mode."
            )

        query_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        timings = StepTimings()

        logger.info(
            "query_started",
            query_id=query_id,
            user_id=request.user_id,
            server_id=request.server_id,
        )

        try:
            # Step 1: Embedding generation
            step_start = time.perf_counter()
            query_embedding = await self._generate_embedding(request.query_text)
            timings.embedding_ms = int((time.perf_counter() - step_start) * 1000)
            logger.info(
                "embedding_generated",
                query_id=query_id,
                latency_ms=timings.embedding_ms,
            )

            # Step 2: Hybrid retrieval (always executes)
            step_start = time.perf_counter()
            chunks = await self.hybrid_retrieval.retrieve(
                query_embedding=query_embedding,
                query_text=request.query_text,
            )
            timings.retrieval_ms = int((time.perf_counter() - step_start) * 1000)
            logger.info(
                "retrieval_completed",
                query_id=query_id,
                chunks_retrieved=len(chunks),
                latency_ms=timings.retrieval_ms,
            )

            # Step 3: Context expansion
            step_start = time.perf_counter()
            # Extract ChunkData from tuples
            chunk_data_list = [chunk for chunk, score in chunks]
            expanded_chunks = await self.context_expander.expand_context(chunk_data_list)
            timings.expansion_ms = int((time.perf_counter() - step_start) * 1000)
            logger.info(
                "context_expanded",
                query_id=query_id,
                initial_chunks=len(chunk_data_list),
                expanded_chunks=len(expanded_chunks),
                latency_ms=timings.expansion_ms,
            )

            # Step 4: LLM structured generation (includes language detection)
            step_start = time.perf_counter()
            llm_response = await self._generate_llm_response(
                query_text=request.query_text,
                chunks=expanded_chunks,
            )
            timings.llm_ms = int((time.perf_counter() - step_start) * 1000)
            logger.info(
                "llm_response_generated",
                query_id=query_id,
                smalltalk=llm_response.smalltalk,
                detected_language=llm_response.language,
                latency_ms=timings.llm_ms,
            )

            # Step 5: Build QueryResponse
            total_latency = int((time.perf_counter() - start_time) * 1000)

            response = QueryResponse(
                answer=llm_response.answer or "",
                personality_reply=llm_response.personality_reply,
                sources=[str(url) for url in (llm_response.sources or [])],
                smalltalk=llm_response.smalltalk,
                language=llm_response.language,
                metadata={
                    "latency_ms": total_latency,
                    "embedding_ms": timings.embedding_ms,
                    "retrieval_ms": timings.retrieval_ms,
                    "expansion_ms": timings.expansion_ms,
                    "llm_ms": timings.llm_ms,
                    "chunks_retrieved": len(chunks),
                    "chunks_expanded": len(expanded_chunks),
                },
            )

            logger.info(
                "query_completed",
                query_id=query_id,
                success=True,
                smalltalk=llm_response.smalltalk,
                total_latency_ms=total_latency,
            )

            return response

        except RetrievalError as e:
            logger.error(
                "query_retrieval_failed",
                query_id=query_id,
                user_id=request.user_id,
                server_id=request.server_id,
                error=str(e),
                exc_info=True,
            )
            return self._error_response(
                "Failed to retrieve relevant information. Please try again.", start_time
            )

        except LLMProviderError as e:
            logger.error(
                "query_llm_failed",
                query_id=query_id,
                user_id=request.user_id,
                server_id=request.server_id,
                error=str(e),
                exc_info=True,
            )
            return self._error_response(
                "Failed to generate response. Please try again.", start_time
            )

        except PydanticValidationError as e:
            logger.error(
                "query_llm_validation_failed",
                query_id=query_id,
                user_id=request.user_id,
                server_id=request.server_id,
                error=str(e),
                exc_info=True,
            )
            return self._error_response(
                "Invalid response format from LLM. Please try again.", start_time
            )

        except TimeoutError:
            logger.error(
                "query_timeout",
                query_id=query_id,
                user_id=request.user_id,
                server_id=request.server_id,
                timeout_seconds=self.timeout_seconds,
            )
            return self._error_response(
                f"Request timed out after {self.timeout_seconds} seconds.", start_time
            )

        except Exception as e:
            logger.error(
                "query_unexpected_error",
                query_id=query_id,
                user_id=request.user_id,
                server_id=request.server_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return self._error_response("An unexpected error occurred.", start_time)

    async def _generate_embedding(self, query_text: str) -> np.ndarray:
        """Generate embedding for query text.

        Args:
            query_text: The query to embed

        Returns:
            Numpy array with 1536-dim embedding

        Raises:
            RetrievalError: If embedding generation fails
        """
        # Run synchronous embedding generator in thread pool
        embeddings = await asyncio.to_thread(
            self.embedding_generator.generate_embeddings, [query_text]
        )

        if not embeddings or embeddings[0] is None:
            raise RetrievalError("Failed to generate query embedding")

        return embeddings[0]

    async def _generate_llm_response(
        self,
        query_text: str,
        chunks: list[ChunkData],
    ) -> LLMStructuredResponse:
        """Generate structured LLM response with automatic language detection.

        Args:
            query_text: User's query
            chunks: Retrieved and expanded chunks

        Returns:
            Validated LLMStructuredResponse with detected language

        Note:
            This method should only be called from process() which validates
            that llm_router is not None.
        """
        # Assert for type narrowing - process() validates this before calling
        assert self.llm_router is not None

        # Build system prompt (persona + instructions)
        system_prompt = self._build_system_prompt()

        # Build user prompt (context + question)
        user_prompt = self._build_user_prompt(query_text, chunks)

        # Generate structured response with separate prompts
        options = GenerationOptions(
            model=self.llm_router.default_model,
            temperature=0.7,
            max_tokens=800,
            system_prompt=system_prompt,
        )

        return await self.llm_router.generate_structured(user_prompt, options)

    def _build_system_prompt(self) -> str:
        """Build system prompt based on personality mode.

        Loads the system prompt template and persona from /prompts/ directory files.
        Uses .format() to properly handle escaped braces in JSON examples.

        Returns:
            System prompt string with persona and language detection instructions
        """
        # Load system prompt template
        system_template = (PROMPTS_DIR / SYSTEM_PROMPT_FILE).read_text()

        # Select persona file based on personality setting
        persona_file = f"persona-{self.personality}.md"
        persona_content = (PROMPTS_DIR / persona_file).read_text()

        # Substitute {persona} placeholder using .format() to handle escaped braces
        return system_template.format(persona=persona_content.strip())

    def _build_user_prompt(self, query_text: str, chunks: list[ChunkData]) -> str:
        """Build user prompt from template with context and question.

        Args:
            query_text: User's question
            chunks: Retrieved and expanded chunks

        Returns:
            User prompt string with context and question
        """
        # Load user prompt template
        user_template = (PROMPTS_DIR / USER_PROMPT_FILE).read_text()

        # Build context from chunks
        context = self._build_context(chunks)

        # Substitute placeholders
        return user_template.format(chunks=context, question=query_text)

    def _build_context(self, chunks: list[ChunkData]) -> str:
        """Build context string from chunks.

        Args:
            chunks: List of ChunkData to include in context

        Returns:
            Formatted context string with article titles and text
        """
        context_parts: list[str] = []

        for chunk in chunks:
            article_title = chunk.get("article_title", "Unknown")
            section_path = chunk.get("section_path", "")
            chunk_text = chunk.get("chunk_text", "")

            # Format: [Article Title > Section] Content
            if section_path and section_path != "Infobox":
                header = f"[{article_title} > {section_path}]"
            else:
                header = f"[{article_title}]"

            context_parts.append(f"{header}\n{chunk_text}")

        return "\n\n".join(context_parts)

    def _error_response(self, error_message: str, start_time: float) -> QueryResponse:
        """Create an error QueryResponse.

        Args:
            error_message: User-friendly error message
            start_time: Request start time for latency calculation

        Returns:
            QueryResponse with error field populated
        """
        total_latency = int((time.perf_counter() - start_time) * 1000)

        return QueryResponse(
            answer="",
            personality_reply="",
            sources=[],
            smalltalk=False,
            language="EN",
            metadata={"latency_ms": total_latency},
            error=error_message,
        )
