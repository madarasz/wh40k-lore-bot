"""WikiChunk model for storing article chunks with embeddings and metadata."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class WikiChunk(Base):
    """Represents a chunk of wiki article content with metadata.

    This model stores individual chunks of wiki articles along with their
    metadata for efficient retrieval and filtering in the RAG system.

    Attributes:
        id: Unique identifier for the chunk (UUID)
        wiki_page_id: Wiki page ID from XML export (for traceability)
        article_title: Title of the source article
        section_path: Hierarchical section path (e.g., "History > Horus Heresy")
        chunk_text: The actual text content of the chunk
        chunk_index: Zero-based index of chunk within the article
        metadata_json: Flexible JSON metadata structure
        created_at: Timestamp when chunk was created
        updated_at: Timestamp when chunk was last updated
    """

    __tablename__ = "wiki_chunk"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Core fields
    wiki_page_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    article_title: Mapped[str] = mapped_column(String(500), index=True, nullable=False)
    section_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Metadata as JSON
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize WikiChunk with defaults for auto-generated fields."""
        # Set defaults for fields not provided
        if "id" not in kwargs:
            kwargs["id"] = str(uuid.uuid4())
        if "metadata_json" not in kwargs:
            kwargs["metadata_json"] = {}
        if "created_at" not in kwargs:
            kwargs["created_at"] = datetime.now(UTC)
        if "updated_at" not in kwargs:
            kwargs["updated_at"] = datetime.now(UTC)

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of WikiChunk.

        Returns:
            String representation showing id and article title
        """
        return f"<WikiChunk(id={self.id}, article_title='{self.article_title}')>"
