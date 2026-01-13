"""Query orchestrator for coordinating the RAG pipeline with structured LLM output."""

import asyncio
import os
import time
import uuid

import numpy as np
import structlog
from pydantic import ValidationError as PydanticValidationError

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.llm.base_provider import GenerationOptions
from src.llm.llm_router import MultiLLMRouter
from src.llm.prompt_builder import PromptBuilder
from src.llm.response_formatter import ResponseFormatter
from src.llm.structured_output import LLMStructuredResponse
from src.orchestration.models import (
    QueryRequest,
    QueryResponse,
    RetrievalMetadata,
    RetrievalPipelineResult,
    RetrievalResult,
    StepTimings,
)
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

        # Initialize prompt builder for template loading
        self.prompt_builder = PromptBuilder()

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

        logger.info("retrieval_only_started", query_text=query_text[:100])

        # Execute common retrieval pipeline
        pipeline_result = await self._execute_retrieval_pipeline(query_text, top_k)

        # Reconstruct results with expanded chunks (assign 0.0 score to expanded)
        initial_count = len(pipeline_result.initial_chunks_with_scores)
        result_chunks: list[tuple[ChunkData, float]] = list(
            pipeline_result.initial_chunks_with_scores
        )
        for expanded_chunk in pipeline_result.expanded_chunks[initial_count:]:
            result_chunks.append((expanded_chunk, 0.0))

        # Build metadata
        total_latency = int((time.perf_counter() - start_time) * 1000)
        metadata = RetrievalMetadata(
            latency_ms=total_latency,
            embedding_ms=pipeline_result.timings.embedding_ms,
            retrieval_ms=pipeline_result.timings.retrieval_ms,
            expansion_ms=pipeline_result.timings.expansion_ms,
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

        logger.info(
            "query_started",
            query_id=query_id,
            user_id=request.user_id,
            server_id=request.server_id,
        )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                # Steps 1-3: Execute common retrieval pipeline
                pipeline_result = await self._execute_retrieval_pipeline(request.query_text)

                logger.info(
                    "retrieval_pipeline_completed",
                    query_id=query_id,
                    chunks_retrieved=len(pipeline_result.initial_chunks_with_scores),
                    chunks_expanded=len(pipeline_result.expanded_chunks),
                    embedding_ms=pipeline_result.timings.embedding_ms,
                    retrieval_ms=pipeline_result.timings.retrieval_ms,
                    expansion_ms=pipeline_result.timings.expansion_ms,
                )

                # Step 4: LLM structured generation (includes language detection)
                step_start = time.perf_counter()
                llm_response = await self._generate_llm_response(
                    query_text=request.query_text,
                    chunks=pipeline_result.expanded_chunks,
                )
                pipeline_result.timings.llm_ms = int((time.perf_counter() - step_start) * 1000)
                logger.info(
                    "llm_response_generated",
                    query_id=query_id,
                    smalltalk=llm_response.smalltalk,
                    detected_language=llm_response.language,
                    latency_ms=pipeline_result.timings.llm_ms,
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
                        "embedding_ms": pipeline_result.timings.embedding_ms,
                        "retrieval_ms": pipeline_result.timings.retrieval_ms,
                        "expansion_ms": pipeline_result.timings.expansion_ms,
                        "llm_ms": pipeline_result.timings.llm_ms,
                        "chunks_retrieved": len(pipeline_result.initial_chunks_with_scores),
                        "chunks_expanded": len(pipeline_result.expanded_chunks),
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

    async def _execute_retrieval_pipeline(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> RetrievalPipelineResult:
        """Execute embedding, retrieval, and expansion steps.

        Args:
            query_text: The query text to search for
            top_k: Number of results to retrieve (defaults to hybrid_retrieval.top_k)

        Returns:
            RetrievalPipelineResult with chunks and timings

        Raises:
            RetrievalError: If embedding generation or retrieval fails
        """
        timings = StepTimings()

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

        # Step 3: Context expansion
        step_start = time.perf_counter()
        chunk_data_list = [chunk for chunk, score in chunks_with_scores]
        expanded_chunks = await self.context_expander.expand_context(chunk_data_list)
        timings.expansion_ms = int((time.perf_counter() - step_start) * 1000)

        return RetrievalPipelineResult(
            expanded_chunks=expanded_chunks,
            initial_chunks_with_scores=chunks_with_scores,
            timings=timings,
        )

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

        Raises:
            ConfigurationError: If llm_router is not configured
        """
        # Explicit check for type narrowing - process() validates this before calling
        if self.llm_router is None:
            raise ConfigurationError("LLM router must be configured for generation")

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

        Uses PromptBuilder to load system prompt template and persona files.

        Returns:
            System prompt string with persona and language detection instructions
        """
        return self.prompt_builder.build_system_prompt(self.personality)

    def _build_user_prompt(self, query_text: str, chunks: list[ChunkData]) -> str:
        """Build user prompt from template with context and question.

        Args:
            query_text: User's question
            chunks: Retrieved and expanded chunks

        Returns:
            User prompt string with context and question
        """
        # Build context from chunks
        context = self._build_context(chunks)

        return self.prompt_builder.build_user_prompt(context, query_text)

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
