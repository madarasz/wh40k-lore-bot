"""Unit tests for delete-chunk CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestDeleteChunkCLI:
    """Test delete-chunk CLI command."""

    def test_delete_chunk_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.delete_chunk import delete_chunk

        result = runner.invoke(delete_chunk, ["--help"])

        assert result.exit_code == 0
        assert "--chroma-path" in result.output
        assert "--force" in result.output
        assert "CHUNK_ID" in result.output

    def test_delete_chunk_not_found_aborts(self) -> None:
        """Test that non-existent chunk causes abort."""
        runner = CliRunner()

        with patch("src.cli.delete_chunk.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

            from src.cli.delete_chunk import delete_chunk

            result = runner.invoke(delete_chunk, ["nonexistent_0", "--force"])

            assert result.exit_code != 0
            assert "not found" in result.output

    def test_delete_chunk_cancelled_without_force(self) -> None:
        """Test that deletion is cancelled when user declines confirmation."""
        runner = CliRunner()

        with patch("src.cli.delete_chunk.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.collection.get.return_value = {
                "ids": ["58_0"],
                "documents": ["Content"],
                "metadatas": [{"article_title": "Test"}],
            }

            from src.cli.delete_chunk import delete_chunk

            # Input 'n' to decline confirmation
            result = runner.invoke(delete_chunk, ["58_0"], input="n\n")

            assert "Deletion cancelled" in result.output

    def test_delete_chunk_success_with_force(self) -> None:
        """Test delete-chunk succeeds with --force flag."""
        runner = CliRunner()

        with patch("src.cli.delete_chunk.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.collection.get.return_value = {
                "ids": ["58_0"],
                "documents": ["Content"],
                "metadatas": [{"article_title": "Test"}],
            }

            from src.cli.delete_chunk import delete_chunk

            result = runner.invoke(delete_chunk, ["58_0", "--force"])

            assert result.exit_code == 0
            assert "Deletion Complete" in result.output
