"""Data models for wiki ingestion."""

from dataclasses import dataclass


@dataclass
class WikiArticle:
    """Represents a wiki article with metadata.

    Attributes:
        title: Article title
        wiki_id: Unique identifier from wiki XML
        last_updated: ISO 8601 timestamp of last update
        content: Full markdown content of the article
        word_count: Number of words in the content
    """

    title: str
    wiki_id: str
    last_updated: str
    content: str
    word_count: int

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
