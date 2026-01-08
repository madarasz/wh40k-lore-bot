"""Context expander for enriching retrieval results with cross-referenced content."""

import asyncio
import os

import structlog

from src.rag.vector_store import ChromaVectorStore, ChunkData
from src.utils.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class ContextExpander:
    """Expands context by following cross-references in chunk metadata.

    Uses the metadata.links field to fetch related chunks from linked articles,
    enriching the context for LLM generation.

    Attributes:
        vector_store: ChromaVectorStore instance for fetching linked chunks
        enabled: Whether context expansion is enabled
        expansion_depth: How many levels to expand (0-2)
        max_chunks: Maximum total chunks after expansion
    """

    DEFAULT_ENABLED = True
    DEFAULT_EXPANSION_DEPTH = 1
    DEFAULT_MAX_CHUNKS = 30
    MIN_EXPANSION_DEPTH = 0
    MAX_EXPANSION_DEPTH = 2
    MAX_CHUNKS_PER_LINK_DEPTH1 = 2
    MAX_CHUNKS_PER_LINK_DEPTH2 = 1

    @staticmethod
    def _parse_env_bool(key: str, default: bool) -> bool:
        """Parse boolean from environment with validation."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _parse_env_int(key: str, default: int) -> int:
        """Parse integer from environment with validation."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ValidationError(f"Invalid integer value for {key}: {value}") from None

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        enabled: bool | None = None,
        expansion_depth: int | None = None,
        max_chunks: int | None = None,
    ) -> None:
        """Initialize ContextExpander.

        Args:
            vector_store: ChromaVectorStore instance for fetching chunks
            enabled: Enable/disable expansion (defaults to env CONTEXT_EXPANSION_ENABLED or True)
            expansion_depth: Expansion depth 0-2 (defaults to env CONTEXT_EXPANSION_DEPTH or 1)
            max_chunks: Max total chunks (defaults to env CONTEXT_EXPANSION_MAX_CHUNKS or 30)

        Raises:
            ValidationError: If expansion_depth is out of range or max_chunks is negative
        """
        self.vector_store = vector_store

        # Load configuration from environment with fallbacks
        self.enabled = (
            enabled
            if enabled is not None
            else self._parse_env_bool("CONTEXT_EXPANSION_ENABLED", self.DEFAULT_ENABLED)
        )
        self.expansion_depth = (
            expansion_depth
            if expansion_depth is not None
            else self._parse_env_int("CONTEXT_EXPANSION_DEPTH", self.DEFAULT_EXPANSION_DEPTH)
        )
        self.max_chunks = (
            max_chunks
            if max_chunks is not None
            else self._parse_env_int("CONTEXT_EXPANSION_MAX_CHUNKS", self.DEFAULT_MAX_CHUNKS)
        )

        # Validate configuration
        if not (self.MIN_EXPANSION_DEPTH <= self.expansion_depth <= self.MAX_EXPANSION_DEPTH):
            raise ValidationError(f"Expansion depth must be 0-2 (got {self.expansion_depth})")

        if self.max_chunks < 0:
            raise ValidationError(f"Max chunks cannot be negative (got {self.max_chunks})")

        logger.info(
            "context_expander_initialized",
            enabled=self.enabled,
            expansion_depth=self.expansion_depth,
            max_chunks=self.max_chunks,
        )

    async def expand_context(
        self,
        chunks: list[ChunkData],
        expansion_depth: int | None = None,
    ) -> list[ChunkData]:
        """Expand context by following cross-references.

        Args:
            chunks: Initial retrieved chunks
            expansion_depth: Override depth (0-2, defaults to self.expansion_depth)

        Returns:
            Expanded list of chunks with related content

        Raises:
            ValidationError: If expansion_depth is out of range
        """
        # Use provided depth or fall back to instance default
        depth = expansion_depth if expansion_depth is not None else self.expansion_depth

        # Validate depth
        if not (self.MIN_EXPANSION_DEPTH <= depth <= self.MAX_EXPANSION_DEPTH):
            raise ValidationError(f"Expansion depth must be 0-2 (got {depth})")

        # Return original chunks if expansion disabled or depth is 0
        if not self.enabled or depth == 0:
            logger.info(
                "context_expansion_skipped",
                enabled=self.enabled,
                expansion_depth=depth,
                initial_chunks=len(chunks),
            )
            return chunks

        initial_count = len(chunks)

        logger.info(
            "context_expansion_started",
            initial_chunks=initial_count,
            expansion_depth=depth,
        )

        # Track seen chunk IDs to prevent duplicates
        expanded = chunks.copy()
        seen_ids = {chunk["id"] for chunk in chunks}

        # Depth 1: Expand from initial chunks
        if depth >= 1:
            depth1_added = await self._expand_depth1(expanded, seen_ids)
            logger.info("depth1_expansion_completed", chunks_added=depth1_added)

        # Depth 2: Expand from depth-1 chunks
        if depth >= self.MAX_EXPANSION_DEPTH:
            depth2_added = await self._expand_depth2(expanded, seen_ids, initial_count)
            logger.info("depth2_expansion_completed", chunks_added=depth2_added)

        # Enforce max chunks limit
        if len(expanded) > self.max_chunks:
            expanded = expanded[: self.max_chunks]

        final_count = len(expanded)

        logger.info(
            "context_expansion_completed",
            initial_chunks=initial_count,
            final_chunks=final_count,
            chunks_added=final_count - initial_count,
            expansion_depth=depth,
        )

        return expanded

    async def _expand_depth1(
        self,
        expanded: list[ChunkData],
        seen_ids: set[str],
    ) -> int:
        """Perform depth-1 expansion from initial chunks.

        Args:
            expanded: List to append expanded chunks to (modified in place)
            seen_ids: Set of seen chunk IDs (modified in place)

        Returns:
            Number of chunks added
        """
        # Extract links from initial chunks (all chunks in expanded at this point)
        links = self._extract_links(expanded)
        chunks_added = 0

        logger.info("depth1_links_extracted", unique_links=len(links))

        # Fetch chunks for each link (limit to max_chunks to prevent runaway expansion)
        for link in links[: self.max_chunks]:
            related_chunks = await self._fetch_chunks_by_article(
                link, limit=self.MAX_CHUNKS_PER_LINK_DEPTH1
            )

            for chunk in related_chunks:
                if chunk["id"] not in seen_ids and len(expanded) < self.max_chunks:
                    expanded.append(chunk)
                    seen_ids.add(chunk["id"])
                    chunks_added += 1

        return chunks_added

    async def _expand_depth2(
        self,
        expanded: list[ChunkData],
        seen_ids: set[str],
        initial_count: int,
    ) -> int:
        """Perform depth-2 expansion from depth-1 chunks.

        Args:
            expanded: List to append expanded chunks to (modified in place)
            seen_ids: Set of seen chunk IDs (modified in place)
            initial_count: Number of initial chunks (to identify depth-1 chunks)

        Returns:
            Number of chunks added
        """
        # Extract links from depth-1 chunks only
        depth1_chunks = expanded[initial_count:]
        secondary_links = self._extract_links(depth1_chunks)
        chunks_added = 0

        logger.info("depth2_links_extracted", unique_links=len(secondary_links))

        # Fetch chunks for each secondary link (1 chunk per link)
        for link in secondary_links:
            if len(expanded) >= self.max_chunks:
                break

            related_chunks = await self._fetch_chunks_by_article(
                link, limit=self.MAX_CHUNKS_PER_LINK_DEPTH2
            )

            for chunk in related_chunks:
                if chunk["id"] not in seen_ids and len(expanded) < self.max_chunks:
                    expanded.append(chunk)
                    seen_ids.add(chunk["id"])
                    chunks_added += 1

        return chunks_added

    def _extract_links(self, chunks: list[ChunkData]) -> list[str]:
        """Extract unique article titles from chunk metadata links.

        Args:
            chunks: List of chunks to extract links from

        Returns:
            List of unique article titles (preserving order)
        """
        links: list[str] = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_links = metadata.get("links", [])

            if isinstance(chunk_links, list):
                links.extend(chunk_links)

        # Return unique links preserving order
        unique_links = list(dict.fromkeys(links))

        return unique_links

    async def _fetch_chunks_by_article(
        self,
        article_title: str,
        limit: int,
    ) -> list[ChunkData]:
        """Fetch chunks for a specific article title.

        Args:
            article_title: Article title to fetch chunks for
            limit: Maximum number of chunks to fetch

        Returns:
            List of ChunkData for the article
        """
        try:
            # Query vector store with article_title filter
            results = await asyncio.to_thread(
                self.vector_store.collection.get,
                where={"article_title": article_title},
                limit=limit,
                include=["metadatas", "documents"],
            )

            chunks: list[ChunkData] = []

            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    document = results["documents"][i] if results["documents"] else ""

                    # Use vector_store's helper to convert metadata to ChunkData
                    chunk = self.vector_store._metadata_to_chunk(
                        chunk_id=chunk_id,
                        metadata=metadata,  # type: ignore[arg-type]
                        document=document,
                    )
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(
                "fetch_chunks_by_article_failed",
                article_title=article_title,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Return empty list on error rather than failing the entire expansion
            return []
