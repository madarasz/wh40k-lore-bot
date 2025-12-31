"""Unit tests for ingest CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestIngestCLI:
    """Test ingest CLI command."""

    def test_ingest_dry_run_succeeds_with_mocked_pipeline(self):
        """Test that ingest --dry-run works when pipeline is mocked."""
        runner = CliRunner()

        with patch("src.cli.ingest.IngestionPipeline") as mock_pipeline_class:
            # Mock the pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            # Mock run to return valid stats
            mock_stats = MagicMock()
            mock_stats.articles_processed = 0
            mock_stats.articles_skipped = 0
            mock_stats.articles_failed = 0
            mock_stats.chunks_created = 0
            mock_stats.chunks_deleted = 0
            mock_stats.embeddings_generated = 0
            mock_stats.total_cost = 0.0
            mock_stats.tokens_used = 0
            mock_stats.duration_seconds = 0
            mock_pipeline.run.return_value = mock_stats

            from src.cli.ingest import ingest

            result = runner.invoke(
                ingest,
                ["--dry-run", "--archive-path", "data/markdown-archive"],
            )

            # Should complete without error
            assert result.exit_code == 0 or "Ingestion Complete" in result.output


class TestIngestCLIOptions:
    """Test ingest CLI command options."""

    def test_ingest_help_shows_options(self):
        """Test that --help shows all expected options."""
        runner = CliRunner()

        from src.cli.ingest import ingest

        result = runner.invoke(ingest, ["--help"])

        assert result.exit_code == 0
        assert "--archive-path" in result.output
        assert "--wiki-ids-file" in result.output
        assert "--batch-size" in result.output
        assert "--dry-run" in result.output
        assert "--force" in result.output
        assert "--chroma-path" in result.output

    def test_wiki_ids_file_validation(self):
        """Test that non-existent wiki-ids-file is rejected."""
        runner = CliRunner()

        from src.cli.ingest import ingest

        result = runner.invoke(
            ingest,
            ["--wiki-ids-file", "nonexistent-file.txt"],
        )

        # Should fail because file doesn't exist
        assert result.exit_code != 0
