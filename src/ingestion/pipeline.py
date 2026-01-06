"""Complete ingestion pipeline orchestration for markdown-based wiki data processing."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from tqdm import tqdm

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import Chunk, WikiArticle, build_chunk_metadata
from src.ingestion.text_chunker import MarkdownChunker
from src.rag.vector_store import ChromaVectorStore, ChunkData
from src.repositories.bm25_repository import BM25Repository
from src.utils.chunk_id import generate_chunk_id
from src.utils.exceptions import IngestionError

logger = structlog.get_logger(__name__)


@dataclass
class IngestionStatistics:
    """Statistics for the ingestion pipeline run.

    Attributes:
        articles_processed: Number of articles successfully processed
        articles_skipped: Number of unchanged articles skipped
        articles_failed: Number of articles that failed processing
        chunks_created: Total number of chunks created
        chunks_deleted: Number of chunks deleted for re-ingestion
        embeddings_generated: Number of embeddings generated
        tokens_used: Total tokens used for embedding generation
        total_cost: Total cost in USD for API calls
        start_time: Start time as unix timestamp
        end_time: End time as unix timestamp
        duration_seconds: Total processing duration
    """

    articles_processed: int = 0
    articles_skipped: int = 0
    articles_failed: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0
    embeddings_generated: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "articles_processed": self.articles_processed,
            "articles_skipped": self.articles_skipped,
            "articles_failed": self.articles_failed,
            "chunks_created": self.chunks_created,
            "chunks_deleted": self.chunks_deleted,
            "embeddings_generated": self.embeddings_generated,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": round(self.total_cost, 4),
            "duration_seconds": int(self.duration_seconds),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
        }


class IngestionPipeline:
    """Orchestrates end-to-end wiki ingestion pipeline from markdown files.

    Processes markdown files through the complete pipeline:
    1. Load markdown files from archive
    2. Check for changes (skip unchanged articles)
    3. Delete old chunks if article changed
    4. Chunk markdown articles
    5. Extract metadata from chunks
    6. Generate embeddings (batched)
    7. Store chunks + embeddings + metadata in Chroma

    Supports batch processing, change detection, and resumable execution.
    """

    def __init__(
        self,
        archive_path: str | Path | None = None,
        chroma_path: str | None = None,
    ) -> None:
        """Initialize ingestion pipeline components.

        Args:
            archive_path: Path to markdown archive directory
            chroma_path: Path to Chroma vector database
        """
        self.logger = structlog.get_logger(__name__)

        # Configure paths
        self.archive_path = Path(archive_path) if archive_path else Path("data/markdown-archive")
        self.logs_path = Path("logs")
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = MarkdownLoader(self.archive_path)
        self.chunker = MarkdownChunker()
        self.metadata_extractor = MetadataExtractor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore(
            storage_path=chroma_path or ChromaVectorStore.DEFAULT_STORAGE_PATH
        )

        # Statistics tracking
        self.stats = IngestionStatistics()

        # Track processed wiki_page_ids in current run (for change detection without DB)
        self._processed_wiki_ids: set[str] = set()

        # Track all processed chunks for BM25 index building
        self._all_chunks: list[Chunk] = []

        self.logger.info(
            "ingestion_pipeline_initialized",
            archive_path=str(self.archive_path),
            chroma_path=chroma_path or ChromaVectorStore.DEFAULT_STORAGE_PATH,
        )

    def run(
        self,
        wiki_ids: list[str] | None = None,
        batch_size: int = 100,
        dry_run: bool = False,
        force: bool = False,
    ) -> IngestionStatistics:
        """Run complete ingestion pipeline on markdown archive.

        Args:
            wiki_ids: Optional list of wiki IDs to process (None = all files)
            batch_size: Number of articles to process per batch
            dry_run: If True, skip embedding generation and vector storage
            force: If True, re-ingest all articles regardless of last_updated

        Returns:
            IngestionStatistics object with processing metrics

        Raises:
            IngestionError: If pipeline fails fatally
            FileNotFoundError: If archive directory doesn't exist
        """
        self.logger.info(
            "pipeline_run_started",
            archive_path=str(self.archive_path),
            wiki_ids_count=len(wiki_ids) if wiki_ids else None,
            batch_size=batch_size,
            dry_run=dry_run,
            force=force,
        )

        self.stats = IngestionStatistics()
        self.stats.start_time = time.time()
        self._processed_wiki_ids = set()
        self._all_chunks = []  # Clear chunks from previous runs

        try:
            # Load articles from markdown archive
            articles = self.loader.load_all(wiki_ids=wiki_ids)

            # Process articles in batches
            batch: list[WikiArticle] = []
            total_articles = len(wiki_ids) if wiki_ids else None

            # Use tqdm for progress tracking
            with tqdm(
                desc="Processing articles",
                unit="article",
                total=total_articles,
            ) as pbar:
                for article in articles:
                    batch.append(article)

                    # Process batch when size is reached
                    if len(batch) >= batch_size:
                        self._process_batch(batch, dry_run=dry_run, force=force)
                        pbar.update(len(batch))
                        batch = []

                # Process remaining articles
                if batch:
                    self._process_batch(batch, dry_run=dry_run, force=force)
                    pbar.update(len(batch))

            # Calculate final statistics
            self.stats.end_time = time.time()
            self.stats.duration_seconds = self.stats.end_time - self.stats.start_time

            # Get cost summary from embedding generator
            if not dry_run:
                cost_summary = self.embedding_generator.get_cost_summary()
                self.stats.tokens_used = int(cost_summary["total_tokens"])
                self.stats.total_cost = float(cost_summary["total_cost_usd"])

            # Save summary report
            self._save_summary_report()

            # Build BM25 index if not dry run and chunks were processed
            if not dry_run and self._all_chunks:
                self._build_bm25_index()

            # Log completion
            self.logger.info(
                "pipeline_run_completed",
                articles_processed=self.stats.articles_processed,
                articles_skipped=self.stats.articles_skipped,
                articles_failed=self.stats.articles_failed,
                chunks_created=self.stats.chunks_created,
                chunks_deleted=self.stats.chunks_deleted,
                embeddings_generated=self.stats.embeddings_generated,
                tokens_used=self.stats.tokens_used,
                total_cost_usd=round(self.stats.total_cost, 4),
                duration_seconds=int(self.stats.duration_seconds),
            )

            return self.stats

        except Exception as e:
            self.logger.error(
                "pipeline_run_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise IngestionError(f"Pipeline execution failed: {e}") from e

    def _should_process_article(self, article: WikiArticle, force: bool) -> bool:
        """Check if article should be processed based on change detection.

        Compares the article's last_updated timestamp with what's stored in
        ChromaDB to skip unchanged articles.

        Args:
            article: WikiArticle to check
            force: If True, always process

        Returns:
            True if article should be processed (new or changed)
        """
        if force:
            return True

        # Check if already processed in this run
        if article.wiki_id in self._processed_wiki_ids:
            return False

        # Check ChromaDB for existing article with same last_updated
        try:
            stored_last_updated = self.vector_store.get_article_last_updated(article.wiki_id)

            if stored_last_updated is None:
                # Article not in vector store - needs processing
                return True

            # Compare timestamps - skip if unchanged
            if stored_last_updated == article.last_updated:
                self.logger.debug(
                    "skipping_unchanged_article",
                    wiki_id=article.wiki_id,
                    last_updated=article.last_updated,
                )
                return False

            # Article changed - needs re-processing
            self.logger.info(
                "article_changed_needs_reprocessing",
                wiki_id=article.wiki_id,
                old_last_updated=stored_last_updated,
                new_last_updated=article.last_updated,
            )
            return True

        except Exception as e:
            # On error, process to be safe
            self.logger.warning(
                "change_detection_failed_will_process",
                wiki_id=article.wiki_id,
                error=str(e),
            )
            return True

    def _process_batch(  # noqa: PLR0912, PLR0915
        self, batch: list[WikiArticle], dry_run: bool = False, force: bool = False
    ) -> None:
        """Process a batch of articles through the complete pipeline.

        Args:
            batch: List of WikiArticle objects to process
            dry_run: If True, skip embedding generation and storage
            force: If True, re-ingest all articles

        Raises:
            IngestionError: If batch processing fails fatally
        """
        try:
            self.logger.info("processing_batch", batch_size=len(batch))

            # Step 1: Filter articles based on change detection
            articles_to_process: list[WikiArticle] = []
            for article in batch:
                if not self._should_process_article(article, force):
                    self.logger.debug(
                        "skipping_unchanged_article",
                        article_title=article.title,
                        wiki_id=article.wiki_id,
                    )
                    self.stats.articles_skipped += 1
                    continue
                articles_to_process.append(article)

            if not articles_to_process:
                self.logger.info("no_articles_to_process_in_batch")
                return

            # Step 2: Delete old chunks for changed articles
            if not dry_run:
                for article in articles_to_process:
                    try:
                        deleted_count = self.vector_store.delete_by_wiki_page_id(article.wiki_id)
                        if deleted_count > 0:
                            self.logger.info(
                                "deleted_old_chunks",
                                article_title=article.title,
                                wiki_id=article.wiki_id,
                                chunks_deleted=deleted_count,
                            )
                            self.stats.chunks_deleted += deleted_count
                    except Exception as e:
                        self.logger.warning(
                            "delete_old_chunks_failed",
                            article_title=article.title,
                            wiki_id=article.wiki_id,
                            error=str(e),
                        )

            # Step 3: Chunk all articles in batch
            all_chunks: list[tuple[WikiArticle, Any]] = []
            for article in articles_to_process:
                try:
                    chunks = self.chunker.chunk_markdown(
                        article.content,
                        article.title,
                        infobox=article.infobox,
                        infobox_links=article.infobox_links,
                    )
                    for chunk in chunks:
                        all_chunks.append((article, chunk))
                except Exception as e:
                    self.logger.error(
                        "chunking_failed",
                        article_title=article.title,
                        error=str(e),
                    )
                    self.stats.articles_failed += 1
                    continue

            if not all_chunks:
                self.logger.warning("no_chunks_created_in_batch")
                return

            # Step 4: Extract metadata for all chunks
            for article, chunk in all_chunks:
                try:
                    metadata = self.metadata_extractor.extract_metadata(chunk)
                    chunk.metadata = metadata
                except Exception as e:
                    self.logger.error(
                        "metadata_extraction_failed",
                        article_title=article.title,
                        chunk_index=chunk.chunk_index,
                        error=str(e),
                    )

            self.stats.chunks_created += len(all_chunks)

            if dry_run:
                self.logger.info(
                    "dry_run_batch_complete",
                    chunks_created=len(all_chunks),
                )
                self.stats.articles_processed += len(articles_to_process)
                return

            # Step 5: Generate embeddings for all chunks
            chunk_texts = [chunk.chunk_text for _, chunk in all_chunks]
            try:
                embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            except Exception as e:
                self.logger.error(
                    "embedding_generation_failed_batch",
                    error=str(e),
                )
                raise IngestionError(f"Embedding generation failed: {e}") from e

            # Filter out failed embeddings
            valid_chunks_with_embeddings: list[tuple[WikiArticle, Any, np.ndarray]] = []
            for (article, chunk), embedding in zip(all_chunks, embeddings, strict=True):
                if embedding is not None:
                    valid_chunks_with_embeddings.append((article, chunk, embedding))
                else:
                    self.logger.warning(
                        "embedding_failed_for_chunk",
                        article_title=article.title,
                        chunk_index=chunk.chunk_index,
                    )

            if not valid_chunks_with_embeddings:
                self.logger.error("no_valid_embeddings_in_batch")
                raise IngestionError("All embeddings failed in batch")

            self.stats.embeddings_generated += len(valid_chunks_with_embeddings)

            # Step 6: Convert to ChunkData dicts and store in vector DB
            chunk_data_list = []
            valid_embeddings = []
            for article, chunk, embedding in valid_chunks_with_embeddings:
                # Build complete metadata (shared with chunk.py CLI)
                extracted_metadata = getattr(chunk, "metadata", {})
                chunk_metadata = build_chunk_metadata(
                    extracted_metadata=extracted_metadata,
                    article_last_updated=article.last_updated,
                    links=chunk.links,
                )

                chunk_data: ChunkData = {
                    "id": generate_chunk_id(article.wiki_id, chunk.chunk_index),
                    "wiki_page_id": article.wiki_id,
                    "article_title": article.title,
                    "section_path": chunk.section_path,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk_metadata,
                }
                chunk_data_list.append(chunk_data)
                valid_embeddings.append(embedding)

            # Store in Chroma
            try:
                self.vector_store.add_chunks(chunk_data_list, valid_embeddings)
            except Exception as e:
                self.logger.error(
                    "vector_store_add_failed",
                    error=str(e),
                )
                raise IngestionError(f"Failed to add chunks to vector store: {e}") from e

            # Collect chunks for BM25 index building (only successful ones)
            for article, chunk, _ in valid_chunks_with_embeddings:
                # Set wiki_page_id for proper chunk ID generation
                chunk.wiki_page_id = article.wiki_id
                self._all_chunks.append(chunk)

            # Update statistics
            self.stats.articles_processed += len(articles_to_process)

            # Mark articles as processed only after successful storage
            for article in articles_to_process:
                self._processed_wiki_ids.add(article.wiki_id)

            self.logger.info(
                "batch_processed_successfully",
                articles=len(articles_to_process),
                chunks=len(all_chunks),
                valid_embeddings=len(valid_chunks_with_embeddings),
            )

        except IngestionError:
            raise
        except Exception as e:
            self.logger.error(
                "batch_processing_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise IngestionError(f"Batch processing failed: {e}") from e

    def _save_summary_report(self) -> None:
        """Save ingestion summary report to JSON file."""
        summary_path = self.logs_path / "ingestion-summary.json"

        summary = {
            **self.stats.to_dict(),
            "batches_completed": (self.stats.articles_processed // 100),
        }

        try:
            with summary_path.open("w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info("summary_report_saved", path=str(summary_path))
        except Exception as e:
            self.logger.error(
                "summary_report_save_failed",
                error=str(e),
            )

    def _build_bm25_index(self) -> None:
        """Build BM25 keyword search index from processed chunks.

        Builds and saves BM25 index using all chunks collected during the pipeline run.
        Errors are logged but don't fail the pipeline.
        """
        try:
            self.logger.info(
                "bm25_index_building_started",
                total_chunks=len(self._all_chunks),
            )

            # Initialize BM25 repository
            bm25_repo = BM25Repository()

            # Build index
            bm25_repo.build_index(self._all_chunks)

            # Get index stats
            stats = bm25_repo.get_index_stats()
            self.logger.info(
                "bm25_index_built",
                total_chunks=stats["total_chunks"],
                unique_tokens=stats["unique_tokens"],
                index_built=stats["index_built"],
            )

            # Save index to disk
            bm25_index_path_str = os.getenv("BM25_INDEX_PATH", "data/bm25-index/bm25_index.pkl")
            bm25_index_path = Path(bm25_index_path_str)
            bm25_repo.save_index(bm25_index_path)

            file_size_kb = bm25_index_path.stat().st_size / 1024
            self.logger.info(
                "bm25_index_saved",
                index_path=str(bm25_index_path),
                file_size_kb=round(file_size_kb, 1),
            )

        except Exception as e:
            # Don't fail the whole pipeline if BM25 indexing fails
            self.logger.error(
                "bm25_index_build_failed",
                error=str(e),
                exc_info=True,
            )
