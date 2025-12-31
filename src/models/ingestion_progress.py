"""IngestionProgress model for tracking article processing status."""

import uuid
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class IngestionStatus(str, Enum):
    """Status values for ingestion progress tracking."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionProgress(Base):
    """Tracks processing status of wiki articles during ingestion.

    Used for progress tracking, resumable pipeline execution, and change detection.
    Records which articles have been successfully processed and which failed.
    Stores the article's last_updated timestamp to detect changes on re-ingestion.

    Attributes:
        id: Unique identifier (UUID)
        article_id: Wiki page ID from markdown frontmatter
        article_last_updated: ISO timestamp of article's last update (for change detection)
        status: Current processing status (pending, processing, completed, failed)
        batch_number: Batch number this article was processed in
        processed_at: Timestamp when processing completed (None if pending/processing)
        error_message: Error message if processing failed (None if successful)
        created_at: Timestamp when record was created
    """

    __tablename__ = "ingestion_progress"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Article identification
    article_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    # Article version tracking (for change detection)
    article_last_updated: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Processing status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=IngestionStatus.PENDING.value,
        index=True,
    )

    # Batch tracking
    batch_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timestamps
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of IngestionProgress.

        Returns:
            String representation showing article_id and status
        """
        return f"<IngestionProgress(article_id='{self.article_id}', status='{self.status}')>"
