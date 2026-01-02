"""Unit tests for db-health CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestDbHealthCLI:
    """Test db-health CLI command."""

    def test_db_health_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.db_health import db_health

        result = runner.invoke(db_health, ["--help"])

        assert result.exit_code == 0
        assert "--chroma-path" in result.output

    def test_db_health_success_chroma_ok(self) -> None:
        """Test db-health succeeds when ChromaDB is OK."""
        runner = CliRunner()

        with patch("src.cli.db_health.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 100
            mock_store.collection_name = "wh40k-lore"

            from src.cli.db_health import db_health

            result = runner.invoke(db_health, [])

            assert result.exit_code == 0
            assert "ChromaDB: OK" in result.output
            assert "Total Chunks: 100" in result.output

    def test_db_health_chroma_error(self) -> None:
        """Test db-health handles ChromaDB errors."""
        runner = CliRunner()

        with patch("src.cli.db_health.ChromaVectorStore") as mock_store_class:
            mock_store_class.side_effect = Exception("Connection failed")

            from src.cli.db_health import db_health

            result = runner.invoke(db_health, [])

            assert result.exit_code == 1
            assert "ChromaDB: FAILED" in result.output
