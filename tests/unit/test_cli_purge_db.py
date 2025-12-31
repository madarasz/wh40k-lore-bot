"""Unit tests for purge-db CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestPurgeDbCLI:
    """Test purge-db CLI command."""

    def test_purge_db_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.purge_db import purge_db

        result = runner.invoke(purge_db, ["--help"])

        assert result.exit_code == 0
        assert "--chroma-path" in result.output
        assert "--force" in result.output

    def test_purge_db_empty_stores_exits_early(self) -> None:
        """Test that empty stores exits without confirmation."""
        runner = CliRunner()

        with (
            patch("src.cli.purge_db.ChromaVectorStore") as mock_store_class,
            patch("src.cli.purge_db.os.getenv") as mock_getenv,
            patch("src.cli.purge_db.create_engine") as mock_engine,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 0

            mock_getenv.return_value = "sqlite:///test.db"
            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 0
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

            from src.cli.purge_db import purge_db

            result = runner.invoke(purge_db, [])

            assert result.exit_code == 0
            assert "No data to delete" in result.output

    def test_purge_db_cancelled_without_correct_confirmation(self) -> None:
        """Test that purge is cancelled when confirmation text doesn't match."""
        runner = CliRunner()

        with (
            patch("src.cli.purge_db.ChromaVectorStore") as mock_store_class,
            patch("src.cli.purge_db.os.getenv") as mock_getenv,
            patch("src.cli.purge_db.create_engine") as mock_engine,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 100

            mock_getenv.return_value = "sqlite:///test.db"
            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 100
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

            from src.cli.purge_db import purge_db

            # Input wrong confirmation text
            result = runner.invoke(purge_db, [], input="wrong text\n")

            assert "Deletion cancelled" in result.output

    def test_purge_db_success_with_force(self) -> None:
        """Test purge-db succeeds with --force flag."""
        runner = CliRunner()

        with (
            patch("src.cli.purge_db.ChromaVectorStore") as mock_store_class,
            patch("src.cli.purge_db.os.getenv") as mock_getenv,
            patch("src.cli.purge_db.create_engine") as mock_engine,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 50
            mock_store.collection.get.return_value = {"ids": ["1_0", "1_1"]}

            mock_getenv.return_value = "sqlite:///test.db"
            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 50
            mock_conn.execute.return_value.rowcount = 50
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

            from src.cli.purge_db import purge_db

            result = runner.invoke(purge_db, ["--force"])

            assert result.exit_code == 0
            assert "Purge Complete" in result.output

    def test_purge_db_success_with_delete_all_confirmation(self) -> None:
        """Test purge-db succeeds when user types DELETE ALL."""
        runner = CliRunner()

        with (
            patch("src.cli.purge_db.ChromaVectorStore") as mock_store_class,
            patch("src.cli.purge_db.os.getenv") as mock_getenv,
            patch("src.cli.purge_db.create_engine") as mock_engine,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.count.return_value = 50
            mock_store.collection.get.return_value = {"ids": ["1_0"]}

            mock_getenv.return_value = "sqlite:///test.db"
            mock_conn = MagicMock()
            mock_conn.execute.return_value.scalar.return_value = 50
            mock_conn.execute.return_value.rowcount = 50
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

            from src.cli.purge_db import purge_db

            result = runner.invoke(purge_db, [], input="DELETE ALL\n")

            assert result.exit_code == 0
            assert "Purge Complete" in result.output
