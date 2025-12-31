"""WikiChunk model for storing article chunks with embeddings and metadata."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


def generate_chunk_id(wiki_page_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID from wiki_page_id and chunk_index.

    Format: {wiki_page_id}_{chunk_index}
    Example: "58_0", "58_1", "100_5"

    This enables reliable upsert/delete operations for re-ingestion.

    Args:
        wiki_page_id: Wiki page ID from markdown frontmatter
        chunk_index: Zero-based index of chunk within article

    Returns:
        Deterministic ID string
    """
    return f"{wiki_page_id}_{chunk_index}"


class WikiChunk(Base):
    """Represents a chunk of wiki article content with metadata.

    This model stores individual chunks of wiki articles along with their
    metadata for efficient retrieval and filtering in the RAG system.

    Chunk IDs are deterministic (wiki_page_id_chunk_index) to enable
    reliable updates during re-ingestion.

    Attributes:
        id: Deterministic identifier (wiki_page_id_chunk_index format)
        wiki_page_id: Wiki page ID from markdown frontmatter (for deduplication)
        article_title: Title of the source article
        section_path: Hierarchical section path (e.g., "History > Horus Heresy")
        chunk_text: The actual text content of the chunk
        chunk_index: Zero-based index of chunk within the article
        metadata_json: Flexible JSON metadata structure
        created_at: Timestamp when chunk was created
        updated_at: Timestamp when chunk was last updated
    """

    __tablename__ = "wiki_chunk"

    # Primary key - deterministic based on wiki_page_id and chunk_index
    id: Mapped[str] = mapped_column(
        String(50),  # Increased from 36 (UUID) to accommodate new format
        primary_key=True,
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
        """Initialize WikiChunk with defaults for auto-generated fields.

        The chunk ID is automatically generated from wiki_page_id and chunk_index
        if not explicitly provided.
        """
        # Generate deterministic ID if not provided
        if "id" not in kwargs:
            if "wiki_page_id" in kwargs and "chunk_index" in kwargs:
                kwargs["id"] = generate_chunk_id(kwargs["wiki_page_id"], kwargs["chunk_index"])
            else:
                raise ValueError(
                    "Either 'id' or both 'wiki_page_id' and 'chunk_index' must be provided"
                )

        # Set defaults for optional fields
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
