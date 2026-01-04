"""Data models for wiki ingestion."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Represents a text chunk from a wiki article.

    Attributes:
        chunk_text: The actual text content of the chunk
        article_title: Title of the source article
        section_path: Hierarchical section path (e.g., "History > The Great Crusade")
        chunk_index: 0-based index of chunk within the article
        links: Internal wiki links found in this chunk's text
    """

    chunk_text: str
    article_title: str
    section_path: str
    chunk_index: int
    links: list[str] = field(default_factory=list)


def build_chunk_metadata(
    extracted_metadata: dict[str, Any],
    article_last_updated: str,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """Build complete chunk metadata for storage.

    Merges extracted metadata with article timestamp and chunk links.
    This function is shared between the unified pipeline and step-by-step CLI commands
    to ensure consistent metadata structure.

    Args:
        extracted_metadata: Metadata from MetadataExtractor (character_names, source_books)
        article_last_updated: ISO 8601 timestamp of article's last update
        links: Optional list of internal wiki links from the chunk

    Returns:
        Complete metadata dictionary ready for storage
    """
    metadata = dict(extracted_metadata)
    metadata["article_last_updated"] = article_last_updated
    if links:
        metadata["links"] = links
    return metadata


@dataclass
class WikiArticle:
    """Represents a wiki article with metadata.

    Attributes:
        title: Article title
        wiki_id: Unique identifier from wiki XML
        last_updated: ISO 8601 timestamp of last update
        content: Full markdown content of the article
        word_count: Number of words in the content
        infobox: Formatted infobox text (None if no infobox)
        infobox_links: Internal wiki links found in the infobox
    """

    title: str
    wiki_id: str
    last_updated: str
    content: str
    word_count: int
    infobox: str | None = None
    infobox_links: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate article data after initialization.

        Raises:
            ValueError: If any required field is invalid
        """
        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.wiki_id or not self.wiki_id.strip():
            raise ValueError("Wiki ID cannot be empty")
        if not self.content:
            raise ValueError("Content cannot be empty")
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")
