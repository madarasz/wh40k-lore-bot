"""Markdown archive utilities for saving wiki articles."""

from pathlib import Path

import yaml

from src.ingestion.filename_utils import sanitize_filename
from src.ingestion.models import WikiArticle


def save_markdown_file(article: WikiArticle, archive_path: Path | None = None) -> Path:
    """Save wiki article as markdown file with YAML frontmatter.

    Args:
        article: WikiArticle instance to save
        archive_path: Directory to save file in (default: data/markdown-archive)

    Returns:
        Path object of the saved file

    Raises:
        OSError: If file write fails
        ValueError: If article data is invalid
    """
    # Use default archive path if not specified
    if archive_path is None:
        archive_path = Path("data") / "markdown-archive"

    # Ensure archive directory exists
    archive_path.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    sanitized_name = sanitize_filename(article.title)
    file_path = archive_path / f"{sanitized_name}.md"

    # Generate YAML frontmatter
    frontmatter_data = {
        "title": article.title,
        "wiki_id": article.wiki_id,
        "last_updated": str(article.last_updated),
        "word_count": int(article.word_count),
    }
    frontmatter_body = yaml.safe_dump(
        frontmatter_data, default_flow_style=False, allow_unicode=True
    )
    frontmatter = f"---\n{frontmatter_body}---\n"

    # Combine frontmatter and content
    full_content = frontmatter + "\n" + article.content

    # Write to file
    try:
        file_path.write_text(full_content, encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to write file {file_path}: {e}") from e

    return file_path
