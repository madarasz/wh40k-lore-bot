"""Unit tests for IngestionPipeline change detection logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.models import WikiArticle


class TestShouldProcessArticle:
    """Test _should_process_article change detection method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a pipeline with mocked dependencies."""
        with (
            patch("src.ingestion.pipeline.MarkdownLoader"),
            patch("src.ingestion.pipeline.MarkdownChunker"),
            patch("src.ingestion.pipeline.MetadataExtractor"),
            patch("src.ingestion.pipeline.EmbeddingGenerator"),
            patch("src.ingestion.pipeline.ChromaVectorStore") as mock_store_class,
        ):
            # Create mock vector store
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            # Import after patching
            from src.ingestion.pipeline import IngestionPipeline

            # Create pipeline with mocked dependencies
            pipeline = IngestionPipeline(
                archive_path=Path("test-data/markdown-archive"),
                chroma_path="test-data/chroma-db",
            )

            yield pipeline, mock_store

    @pytest.fixture
    def sample_article(self):
        """Create a sample WikiArticle for testing."""
        return WikiArticle(
            title="Blood Angels",
            wiki_id="page-123",
            last_updated="2024-01-15T10:30:00Z",
            content="Test content about Blood Angels...",
            word_count=100,
        )

    def test_should_process_new_article(self, mock_pipeline, sample_article):
        """Test that new articles (not in vector store) are processed."""
        pipeline, mock_store = mock_pipeline
        mock_store.get_article_last_updated.return_value = None  # Not found

        result = pipeline._should_process_article(sample_article, force=False)

        assert result is True
        mock_store.get_article_last_updated.assert_called_once_with("page-123")

    def test_should_skip_unchanged_article(self, mock_pipeline, sample_article):
        """Test that unchanged articles (same last_updated) are skipped."""
        pipeline, mock_store = mock_pipeline
        # Return same timestamp as article
        mock_store.get_article_last_updated.return_value = "2024-01-15T10:30:00Z"

        result = pipeline._should_process_article(sample_article, force=False)

        assert result is False

    def test_should_process_changed_article(self, mock_pipeline, sample_article):
        """Test that changed articles (different last_updated) are processed."""
        pipeline, mock_store = mock_pipeline
        # Return older timestamp than article
        mock_store.get_article_last_updated.return_value = "2023-06-01T00:00:00Z"

        result = pipeline._should_process_article(sample_article, force=False)

        assert result is True

    def test_force_flag_bypasses_change_detection(self, mock_pipeline, sample_article):
        """Test that force=True processes article regardless of change detection."""
        pipeline, mock_store = mock_pipeline
        # Return same timestamp (would normally be skipped)
        mock_store.get_article_last_updated.return_value = "2024-01-15T10:30:00Z"

        result = pipeline._should_process_article(sample_article, force=True)

        assert result is True
        # Should not even check vector store when force=True
        mock_store.get_article_last_updated.assert_not_called()

    def test_already_processed_in_current_run_skipped(self, mock_pipeline, sample_article):
        """Test that articles processed in current run are skipped.

        This prevents duplicate processing within a single pipeline run.
        """
        pipeline, mock_store = mock_pipeline
        # Simulate article already processed in this run
        pipeline._processed_wiki_ids.add("page-123")

        result = pipeline._should_process_article(sample_article, force=False)

        assert result is False
        # Should not query vector store for already-processed articles
        mock_store.get_article_last_updated.assert_not_called()

    def test_vector_store_error_processes_article(self, mock_pipeline, sample_article):
        """Test that vector store errors result in processing the article.

        On error, we process to be safe rather than potentially missing updates.
        """
        pipeline, mock_store = mock_pipeline
        mock_store.get_article_last_updated.side_effect = Exception("DB error")

        result = pipeline._should_process_article(sample_article, force=False)

        # Should process on error (fail-safe)
        assert result is True

    def test_articles_without_last_updated_in_store(self, mock_pipeline, sample_article):
        """Test handling articles that exist but have no last_updated metadata.

        This can happen for articles ingested before change detection was added.
        They should be re-processed to add the last_updated metadata.
        """
        pipeline, mock_store = mock_pipeline
        # Article exists but no last_updated (old format)
        mock_store.get_article_last_updated.return_value = None

        result = pipeline._should_process_article(sample_article, force=False)

        assert result is True


class TestChangeDetectionIntegration:
    """Integration tests for change detection behavior."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a pipeline with mocked dependencies for batch testing."""
        with (
            patch("src.ingestion.pipeline.MarkdownLoader"),
            patch("src.ingestion.pipeline.MarkdownChunker") as mock_chunker_class,
            patch("src.ingestion.pipeline.MetadataExtractor"),
            patch("src.ingestion.pipeline.EmbeddingGenerator"),
            patch("src.ingestion.pipeline.ChromaVectorStore") as mock_store_class,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            mock_chunker = MagicMock()
            mock_chunker_class.return_value = mock_chunker

            from src.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline(
                archive_path=Path("test-data/markdown-archive"),
                chroma_path="test-data/chroma-db",
            )

            yield pipeline, mock_store

    def test_batch_skips_all_unchanged_articles(self, mock_pipeline):
        """Test that a batch of unchanged articles is entirely skipped."""
        pipeline, mock_store = mock_pipeline

        # Create batch of articles
        articles = [
            WikiArticle(
                title=f"Article {i}",
                wiki_id=f"page-{i}",
                last_updated="2024-01-15T10:30:00Z",
                content=f"Content {i}",
                word_count=100,
            )
            for i in range(5)
        ]

        # All articles are unchanged
        mock_store.get_article_last_updated.return_value = "2024-01-15T10:30:00Z"

        # Process batch
        pipeline._process_batch(articles, dry_run=False, force=False)

        # All articles should be skipped
        assert pipeline.stats.articles_skipped == 5
        assert pipeline.stats.articles_processed == 0
        assert pipeline.stats.chunks_created == 0
