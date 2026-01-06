"""Hybrid retrieval service combining vector similarity and BM25 keyword search."""

import asyncio
import os
import time
from typing import Any

import numpy as np
import structlog

from src.rag.vector_store import ChromaVectorStore, ChunkData
from src.repositories.bm25_repository import BM25Repository

logger = structlog.get_logger(__name__)


class HybridRetrievalService:
    """Service for hybrid retrieval combining vector similarity and BM25 search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from vector similarity
    search (ChromaDB) and BM25 keyword search for improved retrieval quality.

    Attributes:
        vector_store: ChromaVectorStore instance for vector similarity search
        bm25_repository: BM25Repository instance for keyword search
        top_k: Number of results to retrieve (default from env or 20)
        vector_weight: Weight for vector search scores (default: 0.5)
        bm25_weight: Weight for BM25 search scores (default: 0.5)
        rrf_k: RRF constant for fusion algorithm (default: 60)
    """

    DEFAULT_TOP_K = 20
    DEFAULT_VECTOR_WEIGHT = 0.5
    DEFAULT_BM25_WEIGHT = 0.5
    DEFAULT_RRF_K = 60
    WEIGHT_SUM_TOLERANCE_MIN = 0.99
    WEIGHT_SUM_TOLERANCE_MAX = 1.01

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_repository: BM25Repository,
        top_k: int | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> None:
        """Initialize HybridRetrievalService.

        Args:
            vector_store: ChromaVectorStore instance for vector search
            bm25_repository: BM25Repository instance for BM25 search
            top_k: Number of results to retrieve (defaults to env RETRIEVAL_TOP_K or 20)
            vector_weight: Weight for vector scores (defaults to env RETRIEVAL_VECTOR_WEIGHT or 0.5)
            bm25_weight: Weight for BM25 scores (defaults to env RETRIEVAL_BM25_WEIGHT or 0.5)

        Raises:
            ValueError: If weights don't sum to 1.0 or are negative
        """
        self.vector_store = vector_store
        self.bm25_repository = bm25_repository

        # Load configuration from environment with fallbacks
        self.top_k = (
            top_k
            if top_k is not None
            else int(os.getenv("RETRIEVAL_TOP_K", str(self.DEFAULT_TOP_K)))
        )
        self.vector_weight = (
            vector_weight
            if vector_weight is not None
            else float(os.getenv("RETRIEVAL_VECTOR_WEIGHT", str(self.DEFAULT_VECTOR_WEIGHT)))
        )
        self.bm25_weight = (
            bm25_weight
            if bm25_weight is not None
            else float(os.getenv("RETRIEVAL_BM25_WEIGHT", str(self.DEFAULT_BM25_WEIGHT)))
        )
        self.rrf_k = self.DEFAULT_RRF_K

        # Validate weights
        if self.vector_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Weights cannot be negative")

        weight_sum = self.vector_weight + self.bm25_weight
        if not (self.WEIGHT_SUM_TOLERANCE_MIN <= weight_sum <= self.WEIGHT_SUM_TOLERANCE_MAX):
            raise ValueError(f"Weights must sum to 1.0 (got {weight_sum})")

        logger.info(
            "hybrid_retrieval_service_initialized",
            top_k=self.top_k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            rrf_k=self.rrf_k,
        )

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ChunkData, float]]:
        """Execute hybrid retrieval with parallel execution and RRF fusion.

        Runs vector similarity search and BM25 keyword search in parallel,
        then combines results using Reciprocal Rank Fusion.

        Args:
            query_embedding: Query embedding vector (1536-dim numpy array)
            query_text: Query text string for BM25 search
            top_k: Number of results to return (defaults to self.top_k)
            filters: Optional metadata filters for vector search

        Returns:
            List of tuples (ChunkData, fused_score) sorted by score descending

        Raises:
            ValueError: If query_text is empty or query_embedding is invalid
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if query_embedding.size == 0:
            raise ValueError("Query embedding cannot be empty")

        k = top_k or self.top_k

        start_time = time.time()

        # Run both searches in parallel
        vector_results, bm25_results = await asyncio.gather(
            self._vector_search(query_embedding, k, filters),
            self._bm25_search(query_text, k),
        )

        # Fuse results using RRF
        fused_results = self._fuse_results(vector_results, bm25_results, k)

        # Calculate total retrieval time
        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "hybrid_retrieval_completed",
            vector_results_count=len(vector_results),
            bm25_results_count=len(bm25_results),
            fused_results_count=len(fused_results),
            total_time_ms=round(total_time_ms, 2),
            top_k=k,
        )

        return fused_results

    async def _vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[ChunkData, float]]:
        """Execute vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve
            filters: Optional metadata filters

        Returns:
            List of tuples (ChunkData, distance) sorted by similarity

        Raises:
            Exception: If vector search fails
        """
        start_time = time.time()

        try:
            # Chroma query is synchronous, run in thread pool
            results = await asyncio.to_thread(
                self.vector_store.query,
                query_embedding=query_embedding,
                n_results=top_k,
                filters=filters,
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "vector_search_completed",
                results_count=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return results

        except Exception as e:
            logger.error(
                "vector_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    async def _bm25_search(
        self,
        query_text: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Execute BM25 keyword search.

        Args:
            query_text: Query text string
            top_k: Number of results to retrieve

        Returns:
            List of tuples (chunk_id, bm25_score) sorted by score descending

        Raises:
            Exception: If BM25 search fails
        """
        start_time = time.time()

        try:
            # BM25 search is synchronous, run in thread pool
            results = await asyncio.to_thread(
                self.bm25_repository.search,
                query=query_text,
                top_k=top_k,
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "bm25_search_completed",
                results_count=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return results

        except Exception as e:
            logger.error(
                "bm25_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    def _fuse_results(
        self,
        vector_results: list[tuple[ChunkData, float]],
        bm25_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[ChunkData, float]]:
        """Fuse vector and BM25 results using Reciprocal Rank Fusion.

        Args:
            vector_results: List of (ChunkData, distance) from vector search
            bm25_results: List of (chunk_id, score) from BM25 search
            top_k: Number of final results to return

        Returns:
            List of (ChunkData, fused_score) sorted by fused score descending
        """
        start_time = time.time()

        # Build chunk_id to ChunkData mapping from vector results
        chunk_map: dict[str, ChunkData] = {chunk["id"]: chunk for chunk, _ in vector_results}

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}

        # Add vector search RRF scores
        for rank, (chunk, _) in enumerate(vector_results, start=1):
            chunk_id = chunk["id"]
            rrf_score = self.vector_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        # Add BM25 search RRF scores and collect chunk data
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            # If chunk not in vector results, need to fetch from vector store
            if chunk_id not in chunk_map:
                chunk_data = self.vector_store.get_by_id(chunk_id)
                if chunk_data:
                    chunk_map[chunk_id] = chunk_data

        # Sort by fused score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # Build final results (only include chunks we have data for)
        fused_results: list[tuple[ChunkData, float]] = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            if chunk_id in chunk_map:
                fused_results.append((chunk_map[chunk_id], rrf_scores[chunk_id]))

        fusion_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "rrf_fusion_completed",
            unique_chunks=len(rrf_scores),
            final_results=len(fused_results),
            fusion_time_ms=round(fusion_time_ms, 2),
        )

        return fused_results
