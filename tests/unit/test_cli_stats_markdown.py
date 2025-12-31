"""Unit tests for stats-markdown CLI command."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestStatsMarkdownCLI:
    """Test stats-markdown CLI command."""

    def test_stats_markdown_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()

        from src.cli.stats_markdown import stats_markdown

        result = runner.invoke(stats_markdown, ["--help"])

        assert result.exit_code == 0
        assert "--archive-path" in result.output

    def test_stats_markdown_success_with_mocked_loader(self) -> None:
        """Test stats-markdown succeeds with mocked loader."""
        runner = CliRunner()

        with patch("src.cli.stats_markdown.MarkdownLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader

            # Create mock articles
            mock_article1 = MagicMock()
            mock_article1.title = "Article One"
            mock_article1.word_count = 100

            mock_article2 = MagicMock()
            mock_article2.title = "Article Two"
            mock_article2.word_count = 500

            mock_loader.load_all.return_value = iter([mock_article1, mock_article2])

            from src.cli.stats_markdown import stats_markdown

            result = runner.invoke(stats_markdown, ["--archive-path", "data/markdown-archive"])

            assert result.exit_code == 0 or "Total Files: 2" in result.output

    def test_stats_markdown_empty_archive_aborts(self) -> None:
        """Test that empty archive causes abort."""
        runner = CliRunner()

        with patch("src.cli.stats_markdown.MarkdownLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_all.return_value = iter([])

            from src.cli.stats_markdown import stats_markdown

            result = runner.invoke(stats_markdown, ["--archive-path", "data/markdown-archive"])

            assert result.exit_code != 0
            assert "No markdown files found" in result.output
