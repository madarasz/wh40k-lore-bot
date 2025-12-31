"""Unit tests for show-chunk CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestShowChunkCLI:
    """Test show-chunk CLI command."""

    def test_show_chunk_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.show_chunk import show_chunk

        result = runner.invoke(show_chunk, ["--help"])

        assert result.exit_code == 0
        assert "--chroma-path" in result.output
        assert "CHUNK_ID" in result.output

    def test_show_chunk_not_found_aborts(self) -> None:
        """Test that non-existent chunk causes abort."""
        runner = CliRunner()

        with patch("src.cli.show_chunk.ChromaVectorStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

            from src.cli.show_chunk import show_chunk

            result = runner.invoke(show_chunk, ["nonexistent_0"])

            assert result.exit_code != 0
            assert "not found" in result.output

    def test_show_chunk_success_with_mocked_store(self) -> None:
        """Test show-chunk succeeds with mocked vector store."""
        runner = CliRunner()

        with (
            patch("src.cli.show_chunk.ChromaVectorStore") as mock_store_class,
            patch("src.cli.show_chunk.tiktoken") as mock_tiktoken,
        ):
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            # Mock collection.get response
            mock_store.collection.get.return_value = {
                "ids": ["58_0"],
                "documents": ["This is the chunk text content."],
                "metadatas": [
                    {
                        "article_title": "Test Article",
                        "section_path": "Root > Section",
                        "wiki_page_id": "58",
                        "chunk_index": 0,
                        "faction": "Space Marines",
                    }
                ],
            }

            # Mock tiktoken encoder
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tiktoken.get_encoding.return_value = mock_encoder

            from src.cli.show_chunk import show_chunk

            result = runner.invoke(show_chunk, ["58_0"])

            assert result.exit_code == 0
            assert "Chunk ID: 58_0" in result.output
            assert "Test Article" in result.output
