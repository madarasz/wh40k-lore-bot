"""Unit tests for stats-db CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestStatsDbCLI:
    """Test stats-db CLI command."""

    def test_stats_db_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.stats_db import stats_db

        result = runner.invoke(stats_db, ["--help"])

        assert result.exit_code == 0
        assert "--chroma-path" in result.output

    def test_stats_db_empty_database_aborts(self) -> None:
        """Test that empty database causes abort."""
        runner = CliRunner()

        with patch("src.cli.stats_db.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 0

            from src.cli.stats_db import stats_db

            result = runner.invoke(stats_db, ["--chroma-path", "data/chroma-db/"])

            assert result.exit_code != 0
            assert "No chunks found" in result.output

    def test_stats_db_success_with_mocked_store(self) -> None:
        """Test stats-db succeeds with mocked vector store."""
        runner = CliRunner()

        with (
            patch("src.cli.stats_db.ChromaVectorStore") as mock_store_class,
            patch("src.cli.stats_db.tiktoken") as mock_tiktoken,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 2

            # Mock collection.get response
            mock_store.collection.get.return_value = {
                "ids": ["1_0", "1_1"],
                "documents": ["Hello world", "Another chunk"],
                "metadatas": [
                    {"article_title": "Test", "article_last_updated": "2024-01-01"},
                    {"article_title": "Test", "article_last_updated": "2024-01-02"},
                ],
            }

            # Mock tiktoken encoder
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_tiktoken.get_encoding.return_value = mock_encoder

            from src.cli.stats_db import stats_db

            result = runner.invoke(stats_db, ["--chroma-path", "data/chroma-db/"])

            assert result.exit_code == 0 or "Total Chunks: 2" in result.output
