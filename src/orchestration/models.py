"""Data models for the orchestration module."""

from dataclasses import dataclass

from src.rag.vector_store import ChunkData


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


@dataclass
class RetrievalPipelineResult:
    """Result from the retrieval pipeline (embedding + retrieval + expansion).

    Attributes:
        expanded_chunks: Chunks after context expansion
        initial_chunks_with_scores: Original retrieved chunks with scores
        timings: Timing breakdown for each step
    """

    expanded_chunks: list[ChunkData]
    initial_chunks_with_scores: list[tuple[ChunkData, float]]
    timings: StepTimings
