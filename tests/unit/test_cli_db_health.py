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

    def test_db_health_success_both_ok(self) -> None:
        """Test db-health succeeds when both stores are OK."""
        runner = CliRunner()

        with (
            patch("src.cli.db_health.os.getenv") as mock_getenv,
            patch("src.cli.db_health.create_engine") as mock_engine,
            patch("src.cli.db_health.ChromaVectorStore") as mock_store_class,
        ):
            mock_getenv.return_value = "sqlite:///test.db"

            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 100
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 100
            mock_store.collection_name = "wh40k-lore"

            from src.cli.db_health import db_health

            result = runner.invoke(db_health, [])

            assert result.exit_code == 0
            assert "SQLite: OK" in result.output
            assert "Chroma: OK" in result.output
            assert "CONSISTENT" in result.output

    def test_db_health_inconsistent_counts(self) -> None:
        """Test db-health reports inconsistent when counts differ."""
        runner = CliRunner()

        with (
            patch("src.cli.db_health.os.getenv") as mock_getenv,
            patch("src.cli.db_health.create_engine") as mock_engine,
            patch("src.cli.db_health.ChromaVectorStore") as mock_store_class,
        ):
            mock_getenv.return_value = "sqlite:///test.db"

            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 50
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 100
            mock_store.collection_name = "wh40k-lore"

            from src.cli.db_health import db_health

            result = runner.invoke(db_health, [])

            assert result.exit_code == 0
            assert "INCONSISTENT" in result.output

    def test_db_health_missing_database_url(self) -> None:
        """Test db-health handles missing DATABASE_URL."""
        runner = CliRunner()

        with (
            patch("src.cli.db_health.os.getenv") as mock_getenv,
            patch("src.cli.db_health.ChromaVectorStore") as mock_store_class,
        ):
            mock_getenv.return_value = None

            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 100
            mock_store.collection_name = "wh40k-lore"

            from src.cli.db_health import db_health

            result = runner.invoke(db_health, [])

            assert result.exit_code == 1
            assert "DATABASE_URL environment variable is not set" in result.output
            assert "SQLite: FAILED" in result.output
