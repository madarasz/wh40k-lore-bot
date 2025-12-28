"""OpenAI embedding generation service with batching, retry, and rate limiting."""

import os
import time
from collections import deque

import numpy as np
import structlog
from openai import AuthenticationError, OpenAI, OpenAIError, RateLimitError

from src.utils.exceptions import ConfigurationError, EmbeddingGenerationError

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Rate limiter for API requests using sliding window algorithm."""

    def __init__(self, max_rpm: int = 3000):
        """Initialize rate limiter.

        Args:
            max_rpm: Maximum requests per minute allowed
        """
        self.max_rpm = max_rpm
        self.requests: deque[float] = deque()

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()

        # If at limit, sleep until oldest request expires
        if len(self.requests) >= self.max_rpm:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                logger.info("rate_limit_wait", sleep_seconds=sleep_time)
                time.sleep(sleep_time)
                # Clean up again after sleep
                now = time.time()
                while self.requests and self.requests[0] < now - 60:
                    self.requests.popleft()

        self.requests.append(time.time())


class EmbeddingGenerator:
    """Generate vector embeddings using OpenAI's text-embedding-3-small model."""

    # Model configuration
    MODEL_NAME = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536
    COST_PER_MILLION_TOKENS = 0.02  # $0.02 per 1M tokens

    # Batch configuration
    MAX_BATCH_SIZE = 100  # OpenAI limit
    MAX_TOKENS_PER_CHUNK = 8192  # OpenAI limit

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2  # Base for exponential backoff (2^retry seconds)

    def __init__(self, api_key: str | None = None, max_rpm: int = 3000):
        """Initialize embedding generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_rpm: Maximum requests per minute (default 3000)

        Raises:
            ConfigurationError: If API key is not provided or not found in environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY not set. Please set it in your environment or .env file."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.rate_limiter = RateLimiter(max_rpm=max_rpm)
        self.logger = structlog.get_logger(__name__)

        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    def generate_embeddings(self, chunks: list[str]) -> list[np.ndarray | None]:
        """Generate embeddings for text chunks with batching and retry logic.

        Args:
            chunks: List of text strings to embed

        Returns:
            List of numpy arrays (1536-dim, float32), one per chunk.
            Returns None for chunks that failed to embed after retries.

        Raises:
            EmbeddingGenerationError: If all chunks fail or critical error occurs
        """
        if not chunks:
            self.logger.warning("empty_chunks_list")
            return []

        # Filter out invalid chunks
        valid_indices = []
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                self.logger.warning("invalid_chunk_empty", chunk_index=i)
            else:
                valid_indices.append(i)
                valid_chunks.append(chunk)

        if not valid_chunks:
            self.logger.error("all_chunks_invalid")
            raise EmbeddingGenerationError("All chunks are empty or invalid")

        # Process in batches
        all_embeddings: list[np.ndarray | None] = [None] * len(chunks)
        total_batches = (len(valid_chunks) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        self.logger.info(
            "embedding_generation_started",
            total_chunks=len(valid_chunks),
            total_batches=total_batches,
        )

        for batch_idx in range(0, len(valid_chunks), self.MAX_BATCH_SIZE):
            batch_end = min(batch_idx + self.MAX_BATCH_SIZE, len(valid_chunks))
            batch_chunks = valid_chunks[batch_idx:batch_end]
            batch_indices = valid_indices[batch_idx:batch_end]

            self.logger.info(
                "processing_batch",
                batch_number=batch_idx // self.MAX_BATCH_SIZE + 1,
                batch_size=len(batch_chunks),
            )

            # Generate embeddings for batch with retry
            batch_embeddings = self._generate_batch_with_retry(batch_chunks)

            # Store embeddings at correct indices
            for local_idx, embedding in enumerate(batch_embeddings):
                original_idx = batch_indices[local_idx]
                all_embeddings[original_idx] = embedding

        # Log final statistics
        successful_count = sum(1 for e in all_embeddings if e is not None)
        failed_count = len(chunks) - successful_count

        self.logger.info(
            "embedding_generation_completed",
            total_chunks=len(chunks),
            successful=successful_count,
            failed=failed_count,
            total_tokens=self.total_tokens,
            total_cost_usd=round(self.total_cost, 4),
        )

        return all_embeddings

    def _generate_batch_with_retry(self, batch_chunks: list[str]) -> list[np.ndarray | None]:
        """Generate embeddings for a batch with retry logic.

        Args:
            batch_chunks: Batch of text chunks to embed

        Returns:
            List of embeddings (or None for failures)
        """
        for retry in range(self.MAX_RETRIES):
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()

                # Call OpenAI API
                response = self.client.embeddings.create(model=self.MODEL_NAME, input=batch_chunks)

                # Extract embeddings
                embeddings: list[np.ndarray] = [
                    np.array(item.embedding, dtype=np.float32) for item in response.data
                ]

                # Track cost
                tokens_used = response.usage.total_tokens
                cost = tokens_used * (self.COST_PER_MILLION_TOKENS / 1_000_000)
                self.total_tokens += tokens_used
                self.total_cost += cost

                self.logger.info(
                    "batch_embedded_successfully",
                    batch_size=len(batch_chunks),
                    tokens=tokens_used,
                    cost_usd=round(cost, 6),
                )

                # Return embeddings (cast to satisfy mypy - all are non-None in success case)
                return embeddings  # type: ignore[return-value]

            except RateLimitError as e:
                wait_time = self.RETRY_BASE_DELAY**retry
                self.logger.warning(
                    "rate_limit_error_retry",
                    retry_attempt=retry + 1,
                    max_retries=self.MAX_RETRIES,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                if retry < self.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error("rate_limit_max_retries_exceeded", error=str(e))
                    return [None] * len(batch_chunks)

            except AuthenticationError as e:
                # Don't retry auth errors
                self.logger.error("authentication_error", error=str(e))
                raise ConfigurationError(f"Invalid OpenAI API key: {e}") from e

            except OpenAIError as e:
                wait_time = self.RETRY_BASE_DELAY**retry
                self.logger.warning(
                    "openai_error_retry",
                    retry_attempt=retry + 1,
                    max_retries=self.MAX_RETRIES,
                    wait_seconds=wait_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if retry < self.MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        "openai_error_max_retries_exceeded",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    return [None] * len(batch_chunks)

            except Exception as e:
                # Unexpected errors
                self.logger.error(
                    "unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                return [None] * len(batch_chunks)

        # Shouldn't reach here, but return None for all if we do
        return [None] * len(batch_chunks)

    def get_cost_summary(self) -> dict[str, int | float]:
        """Get cost tracking summary.

        Returns:
            Dictionary with total_tokens, total_cost_usd, and cost_per_1k_tokens
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "cost_per_1k_tokens": round((self.COST_PER_MILLION_TOKENS / 1000), 6),  # $0.00002
        }
