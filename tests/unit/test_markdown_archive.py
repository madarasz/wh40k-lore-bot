"""Unit tests for markdown archive utilities."""

from pathlib import Path

import pytest

from src.ingestion.markdown_archive import save_markdown_file
from src.ingestion.models import WikiArticle


class TestSaveMarkdownFile:
    """Test cases for save_markdown_file function."""

    @pytest.fixture
    def sample_article(self) -> WikiArticle:
        """Create a sample WikiArticle for testing."""
        return WikiArticle(
            title="Blood Angels",
            wiki_id="12345",
            last_updated="2024-09-20T14:18:31Z",
            content="The Blood Angels are a Space Marine chapter...",
            word_count=1500,
        )

    @pytest.fixture
    def temp_archive_path(self, tmp_path: Path) -> Path:
        """Create a temporary archive directory for testing."""
        archive_path = tmp_path / "markdown-archive"
        archive_path.mkdir(parents=True, exist_ok=True)
        return archive_path

    def test_creates_file_with_correct_name(
        self, sample_article: WikiArticle, temp_archive_path: Path
    ) -> None:
        """Test that file is created with sanitized filename."""
        result_path = save_markdown_file(sample_article, temp_archive_path)

        assert result_path.exists()
        assert result_path.name == "Blood_Angels.md"
        assert result_path.parent == temp_archive_path

    def test_frontmatter_properly_formatted(
        self, sample_article: WikiArticle, temp_archive_path: Path
    ) -> None:
        """Test that YAML frontmatter is properly formatted."""
        result_path = save_markdown_file(sample_article, temp_archive_path)
        content = result_path.read_text(encoding="utf-8")

        # Check frontmatter delimiters
        assert content.startswith("---\n")
        assert "---\n\n" in content

        # Check frontmatter fields (YAML safe_dump may use single quotes or no quotes)
        assert "title: Blood Angels" in content
        assert "wiki_id: '12345'" in content
        assert "last_updated: '2024-09-20T14:18:31Z'" in content
        assert "word_count: 1500" in content

    def test_content_preserved_exactly(
        self, sample_article: WikiArticle, temp_archive_path: Path
    ) -> None:
        """Test that article content is preserved exactly."""
        result_path = save_markdown_file(sample_article, temp_archive_path)
        content = result_path.read_text(encoding="utf-8")

        # Extract content after frontmatter
        parts = content.split("---\n", 2)
        assert len(parts) == 3
        actual_content = parts[2].lstrip("\n")

        assert actual_content == sample_article.content

    def test_file_path_returned_correctly(
        self, sample_article: WikiArticle, temp_archive_path: Path
    ) -> None:
        """Test that correct Path object is returned."""
        result_path = save_markdown_file(sample_article, temp_archive_path)

        assert isinstance(result_path, Path)
        assert result_path == temp_archive_path / "Blood_Angels.md"

    def test_overwrite_existing_file(
        self, sample_article: WikiArticle, temp_archive_path: Path
    ) -> None:
        """Test that existing files are overwritten."""
        # Create file first time
        first_path = save_markdown_file(sample_article, temp_archive_path)

        # Modify article and save again
        sample_article.content = "Updated content for Blood Angels chapter"
        sample_article.word_count = 500

        second_path = save_markdown_file(sample_article, temp_archive_path)
        second_content = second_path.read_text(encoding="utf-8")

        # Verify overwrite
        assert first_path == second_path
        assert "Updated content for Blood Angels chapter" in second_content
        assert "word_count: 500" in second_content

    def test_uses_default_archive_path(
        self, sample_article: WikiArticle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default archive path is used when not specified."""
        # Change to temp directory so default path is created there
        monkeypatch.chdir(tmp_path)

        result_path = save_markdown_file(sample_article)

        expected_path = tmp_path / "data" / "markdown-archive" / "Blood_Angels.md"
        assert result_path.exists()
        assert result_path.resolve() == expected_path

    def test_creates_archive_directory_if_missing(
        self, sample_article: WikiArticle, tmp_path: Path
    ) -> None:
        """Test that archive directory is created if it doesn't exist."""
        nonexistent_path = tmp_path / "new" / "archive" / "path"
        assert not nonexistent_path.exists()

        result_path = save_markdown_file(sample_article, nonexistent_path)

        assert nonexistent_path.exists()
        assert result_path.exists()

    def test_special_characters_in_title(self, temp_archive_path: Path) -> None:
        """Test handling of special characters in article title."""
        article = WikiArticle(
            title="Chapter Master: Gabriel Angelos (Blood Ravens)",
            wiki_id="67890",
            last_updated="2024-09-21T10:00:00Z",
            content="Gabriel Angelos is the Chapter Master...",
            word_count=2000,
        )

        result_path = save_markdown_file(article, temp_archive_path)

        assert result_path.exists()
        assert result_path.name == "Chapter_Master-_Gabriel_Angelos_Blood_Ravens.md"

    def test_unicode_in_title(self, temp_archive_path: Path) -> None:
        """Test handling of unicode characters in title."""
        article = WikiArticle(
            title="Château de Señor",
            wiki_id="11111",
            last_updated="2024-09-22T12:00:00Z",
            content="A fictional location...",
            word_count=100,
        )

        result_path = save_markdown_file(article, temp_archive_path)

        assert result_path.exists()
        assert result_path.name == "Chateau_de_Senor.md"

    def test_multiline_content(self, temp_archive_path: Path) -> None:
        """Test that multiline content is preserved correctly."""
        multiline_content = """# Blood Angels

## Overview
The Blood Angels are a Space Marine chapter.

## History
Founded during the Great Crusade.

## Notable Members
- Commander Dante
- Sanguinius (Primarch)"""

        article = WikiArticle(
            title="Blood Angels",
            wiki_id="99999",
            last_updated="2024-09-23T15:00:00Z",
            content=multiline_content,
            word_count=50,
        )

        result_path = save_markdown_file(article, temp_archive_path)
        saved_content = result_path.read_text(encoding="utf-8")

        # Extract content after frontmatter
        parts = saved_content.split("---\n", 2)
        actual_content = parts[2].lstrip("\n")

        assert actual_content == multiline_content

    def test_double_quotes_in_title(self, temp_archive_path: Path) -> None:
        """Test that double quotes in title are properly escaped in YAML."""
        article = WikiArticle(
            title='The "Fallen" Angels',
            wiki_id="22222",
            last_updated="2024-09-24T16:00:00Z",
            content="The Fallen are renegade Dark Angels...",
            word_count=300,
        )

        result_path = save_markdown_file(article, temp_archive_path)
        saved_content = result_path.read_text(encoding="utf-8")

        # Verify the file was created and YAML is valid (doesn't break on quotes)
        assert result_path.exists()
        assert saved_content.startswith("---\n")
        assert "---\n\n" in saved_content

        # YAML should properly escape or quote the title with double quotes
        # yaml.safe_dump will use single quotes or escape sequences
        assert "title:" in saved_content
        # Verify the content is preserved correctly
        assert "The Fallen are renegade Dark Angels..." in saved_content
