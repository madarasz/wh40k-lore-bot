"""Repository for managing BM25 keyword search index."""

import os
import pickle  # nosec B403  # Internal use only - we generate and control this file
import time
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.ingestion.models import Chunk
from src.utils.chunk_id import generate_chunk_id

logger = structlog.get_logger(__name__)


class BM25Repository:
    """Repository for BM25 keyword-based search.

    Manages BM25 index for sparse retrieval, providing keyword matching
    alongside vector similarity search in hybrid retrieval.

    Attributes:
        index_path: Path to persisted BM25 index file
        tokenize_lowercase: Whether to lowercase tokens during tokenization
        bm25: BM25Okapi index instance (None until built)
        chunk_ids: Ordered list of chunk IDs corresponding to BM25 index positions
    """

    def __init__(
        self,
        index_path: Path | None = None,
        tokenize_lowercase: bool | None = None,
    ) -> None:
        """Initialize BM25Repository with configuration.

        Args:
            index_path: Path to BM25 index file (defaults to env BM25_INDEX_PATH)
            tokenize_lowercase: Whether to lowercase tokens (defaults to env config)
        """
        # Load configuration from environment
        default_index_path = os.getenv("BM25_INDEX_PATH", "data/bm25-index/bm25_index.pkl")
        self.index_path = index_path or Path(default_index_path)

        # Parse boolean from env (default: False)
        default_lowercase = os.getenv("BM25_TOKENIZE_LOWERCASE", "false").lower() == "true"
        self.tokenize_lowercase = (
            tokenize_lowercase if tokenize_lowercase is not None else default_lowercase
        )

        # Index state (initialized when build_index is called)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.unique_tokens: int = 0

        logger.info(
            "bm25_repository_initialized",
            index_path=str(self.index_path),
            tokenize_lowercase=self.tokenize_lowercase,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 index.

        Uses simple whitespace-based tokenization with optional lowercasing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (empty list if text is empty or None)
        """
        # Handle empty or None input
        if not text:
            logger.debug("tokenize_empty_input", text_value=text)
            return []

        # Lowercase if configured
        processed_text = text.lower() if self.tokenize_lowercase else text

        # Split on whitespace
        tokens = processed_text.split()

        logger.debug(
            "tokenized_text",
            original_length=len(text),
            token_count=len(tokens),
            lowercase=self.tokenize_lowercase,
        )

        return tokens

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks.

        Tokenizes all chunk texts and builds BM25Okapi index for keyword search.
        Stores only chunk IDs (not full Chunk objects) to minimize duplication.

        Args:
            chunks: List of Chunk objects to index

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")

        start_time = time.time()

        # Build ordered list of chunk IDs (matches BM25 index positions)
        # Using wiki_page_id_chunk_index format (consistent with vector store)
        self.chunk_ids = []
        for chunk in chunks:
            # Use wiki_page_id if available, otherwise fall back to article_title
            if chunk.wiki_page_id is not None:
                chunk_id = generate_chunk_id(chunk.wiki_page_id, chunk.chunk_index)
            else:
                # Fallback for chunks without wiki_page_id (backwards compatibility)
                logger.warning(
                    "chunk_missing_wiki_page_id",
                    article_title=chunk.article_title,
                    chunk_index=chunk.chunk_index,
                )
                chunk_id = f"{chunk.article_title}_{chunk.chunk_index}"
            self.chunk_ids.append(chunk_id)

        # Tokenize all chunk texts
        tokenized_corpus = [self._tokenize(chunk.chunk_text) for chunk in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Calculate build time
        build_time_ms = (time.time() - start_time) * 1000

        # Collect token statistics
        total_tokens = sum(len(tokens) for tokens in tokenized_corpus)
        self.unique_tokens = len({token for tokens in tokenized_corpus for token in tokens})

        logger.info(
            "bm25_index_built",
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            unique_tokens=self.unique_tokens,
            build_time_ms=round(build_time_ms, 2),
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for chunks matching query using BM25.

        Args:
            query: Search query text
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending (empty if no index)

        Raises:
            ValueError: If query is empty or index not built
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Check if index is built
        if self.bm25 is None or not self.chunk_ids:
            raise ValueError("Index not built. Call build_index() first")

        start_time = time.time()

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Create list of (chunk_id, score) tuples with indices
        chunk_scores = [(self.chunk_ids[i], float(scores[i])) for i in range(len(self.chunk_ids))]

        # Sort by score descending
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # Limit to top_k results
        results = chunk_scores[:top_k]

        # Calculate search latency
        search_latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "bm25_search_completed",
            query_text=query[:100],  # Truncate long queries for logging
            results_count=len(results),
            top_k=top_k,
            search_latency_ms=round(search_latency_ms, 2),
        )

        return results

    def save_index(self, filepath: Path | None = None) -> None:
        """Save BM25 index to disk.

        Serializes BM25 index and chunk IDs list to disk using pickle.
        Only stores chunk IDs (not full Chunk objects) to minimize duplication
        with ChromaDB (5x file size reduction).

        Args:
            filepath: Path to save index (defaults to self.index_path)

        Raises:
            ValueError: If index not built
            OSError: If file write fails
        """
        if self.bm25 is None or not self.chunk_ids:
            raise ValueError("Index not built. Call build_index() first")

        save_path = filepath or self.index_path

        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization (optimized format)
            index_data = {
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids,
                "tokenize_lowercase": self.tokenize_lowercase,  # Save config for consistency
                "unique_tokens": self.unique_tokens,
            }

            # Write to file
            with open(save_path, "wb") as f:
                pickle.dump(index_data, f)

            logger.info(
                "bm25_index_saved",
                filepath=str(save_path),
                chunk_count=len(self.chunk_ids),
            )

        except OSError as e:
            logger.error(
                "bm25_index_save_failed",
                filepath=str(save_path),
                error=str(e),
            )
            raise

    def load_index(self, filepath: Path | None = None) -> None:  # noqa: PLR0912
        """Load BM25 index from disk.

        Deserializes BM25 index and chunk IDs from disk. Supports both old format
        (with full Chunk objects) and new optimized format (chunk IDs only).

        Args:
            filepath: Path to load index from (defaults to self.index_path)

        Raises:
            FileNotFoundError: If index file doesn't exist
            OSError: If file read fails
            ValueError: If loaded data is invalid
        """
        load_path = filepath or self.index_path

        if not load_path.exists():
            raise FileNotFoundError(f"Index file not found: {load_path}")

        try:
            # Read from file
            with open(load_path, "rb") as f:
                index_data = pickle.load(f)  # nosec B301  # Safe: internally generated file

            # Validate data structure
            if not isinstance(index_data, dict):
                raise ValueError("Invalid index data: expected dictionary")

            if "bm25" not in index_data:
                raise ValueError("Invalid index data: missing 'bm25' key")

            # Load BM25 index
            self.bm25 = index_data["bm25"]

            # Handle both old and new formats
            if "chunk_ids" in index_data:
                # New optimized format
                self.chunk_ids = index_data["chunk_ids"]
                # Restore tokenization config if available
                if "tokenize_lowercase" in index_data:
                    self.tokenize_lowercase = index_data["tokenize_lowercase"]
                # Restore unique_tokens if available
                if "unique_tokens" in index_data:
                    self.unique_tokens = index_data["unique_tokens"]
                else:
                    self.unique_tokens = 0  # Unknown for older index files
                logger.info(
                    "bm25_index_loaded",
                    filepath=str(load_path),
                    chunk_count=len(self.chunk_ids),
                    format="optimized",
                    tokenize_lowercase=self.tokenize_lowercase,
                )
            elif "chunks" in index_data and "chunk_mapping" in index_data:
                # Old format - migrate to chunk_ids
                chunks = index_data["chunks"]
                # Extract chunk IDs from chunks (rebuild from Chunk objects)
                self.chunk_ids = []
                for chunk in chunks:
                    if chunk.wiki_page_id is not None:
                        chunk_id = generate_chunk_id(chunk.wiki_page_id, chunk.chunk_index)
                    else:
                        chunk_id = f"{chunk.article_title}_{chunk.chunk_index}"
                    self.chunk_ids.append(chunk_id)

                # Old format doesn't have unique_tokens stored
                self.unique_tokens = 0

                logger.warning(
                    "bm25_index_loaded_old_format",
                    filepath=str(load_path),
                    chunk_count=len(self.chunk_ids),
                    message="Consider rebuilding index to use optimized format",
                )
            else:
                raise ValueError("Invalid index data: missing both 'chunk_ids' and 'chunks' keys")

        except (OSError, pickle.UnpicklingError) as e:
            logger.error(
                "bm25_index_load_failed",
                filepath=str(load_path),
                error=str(e),
            )
            raise

    def update_index(self, chunks: list[Chunk]) -> None:
        """Update BM25 index with new chunks.

        Note: This performs a full rebuild. Incremental updates not supported in MVP.

        Args:
            chunks: List of Chunk objects to index

        Raises:
            ValueError: If chunks list is empty
        """
        logger.info("bm25_index_update_started", chunk_count=len(chunks))

        # Perform full rebuild
        self.build_index(chunks)

        logger.info("bm25_index_update_completed")

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the current index.

        Returns:
            Dictionary with index statistics:
                - total_chunks: Number of chunks in index
                - unique_tokens: Number of unique tokens in index
                - index_built: Whether index is ready for search
                - index_path: Configured index file path
        """
        stats: dict[str, Any] = {
            "total_chunks": len(self.chunk_ids),
            "unique_tokens": self.unique_tokens,
            "index_built": self.is_index_built(),
            "index_path": str(self.index_path),
        }

        return stats

    def is_index_built(self) -> bool:
        """Check if index has been built.

        Returns:
            True if index is built and ready for search, False otherwise
        """
        return self.bm25 is not None and len(self.chunk_ids) > 0
