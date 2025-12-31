"""Repository for managing ingestion progress tracking."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.ingestion_progress import IngestionProgress, IngestionStatus


class IngestionProgressRepository:
    """Repository for CRUD operations on IngestionProgress.

    Provides methods for tracking article processing status during ingestion pipeline.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def get_article_last_updated(self, article_id: str) -> str | None:
        """Get the stored last_updated timestamp for an article.

        Args:
            article_id: Wiki page ID

        Returns:
            ISO timestamp string if article exists, None otherwise
        """
        stmt = select(IngestionProgress.article_last_updated).where(
            IngestionProgress.article_id == article_id,
            IngestionProgress.status == IngestionStatus.COMPLETED.value,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def should_process_article(self, article_id: str, last_updated: str) -> bool:
        """Check if an article needs to be processed based on last_updated.

        Args:
            article_id: Wiki page ID
            last_updated: Current last_updated timestamp from markdown frontmatter

        Returns:
            True if article should be processed (new or changed), False if unchanged
        """
        stored_last_updated = await self.get_article_last_updated(article_id)

        if stored_last_updated is None:
            # Article not found or not completed - needs processing
            return True

        # Compare timestamps - process if different
        return stored_last_updated != last_updated

    async def upsert_article_progress(
        self,
        article_id: str,
        last_updated: str,
        batch_number: int,
        status: IngestionStatus = IngestionStatus.COMPLETED,
    ) -> IngestionProgress:
        """Insert or update article progress with last_updated tracking.

        If article exists, updates its last_updated and status.
        If article doesn't exist, creates a new record.

        Args:
            article_id: Wiki page ID
            last_updated: ISO timestamp from markdown frontmatter
            batch_number: Current batch number
            status: Processing status (default: COMPLETED)

        Returns:
            IngestionProgress record
        """
        # Check if record exists
        stmt = select(IngestionProgress).where(IngestionProgress.article_id == article_id)
        result = await self.session.execute(stmt)
        progress = result.scalar_one_or_none()

        if progress:
            # Update existing record
            progress.article_last_updated = last_updated
            progress.status = status.value
            progress.batch_number = batch_number
            progress.processed_at = datetime.now(UTC)
            progress.error_message = None
        else:
            # Create new record
            progress = IngestionProgress(
                id=str(uuid.uuid4()),
                article_id=article_id,
                article_last_updated=last_updated,
                status=status.value,
                batch_number=batch_number,
                processed_at=datetime.now(UTC),
            )
            self.session.add(progress)

        await self.session.commit()
        return progress

    async def mark_as_processing(
        self, article_id: str, batch_number: int, last_updated: str | None = None
    ) -> IngestionProgress:
        """Mark an article as currently being processed.

        Args:
            article_id: Wiki page ID
            batch_number: Current batch number
            last_updated: Optional last_updated timestamp from frontmatter

        Returns:
            IngestionProgress record
        """
        progress = IngestionProgress(
            id=str(uuid.uuid4()),
            article_id=article_id,
            article_last_updated=last_updated,
            status=IngestionStatus.PROCESSING.value,
            batch_number=batch_number,
        )
        self.session.add(progress)
        await self.session.commit()
        return progress

    async def mark_as_completed(self, article_id: str) -> None:
        """Mark an article as successfully processed.

        Args:
            article_id: Wiki page ID
        """
        stmt = select(IngestionProgress).where(IngestionProgress.article_id == article_id)
        result = await self.session.execute(stmt)
        progress = result.scalar_one_or_none()

        if progress:
            progress.status = IngestionStatus.COMPLETED.value
            progress.processed_at = datetime.now(UTC)
            await self.session.commit()

    async def mark_as_failed(self, article_id: str, error_message: str) -> None:
        """Mark an article as failed with error message.

        Args:
            article_id: Wiki page ID
            error_message: Error description
        """
        stmt = select(IngestionProgress).where(IngestionProgress.article_id == article_id)
        result = await self.session.execute(stmt)
        progress = result.scalar_one_or_none()

        if progress:
            progress.status = IngestionStatus.FAILED.value
            progress.processed_at = datetime.now(UTC)
            progress.error_message = error_message
            await self.session.commit()

    async def is_completed(self, article_id: str) -> bool:
        """Check if an article has already been successfully processed.

        Args:
            article_id: Wiki page ID

        Returns:
            True if article is marked as completed
        """
        stmt = select(IngestionProgress).where(
            IngestionProgress.article_id == article_id,
            IngestionProgress.status == IngestionStatus.COMPLETED.value,
        )
        result = await self.session.execute(stmt)
        progress = result.scalar_one_or_none()
        return progress is not None

    async def get_last_completed_batch(self) -> int | None:
        """Get the last successfully completed batch number.

        Returns:
            Batch number of last completed batch, or None if no batches completed
        """
        stmt = (
            select(IngestionProgress.batch_number)
            .where(IngestionProgress.status == IngestionStatus.COMPLETED.value)
            .order_by(IngestionProgress.batch_number.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        batch_number = result.scalar_one_or_none()
        return batch_number

    async def rollback_incomplete_batches(self) -> int:
        """Rollback articles marked as processing (incomplete batches).

        Returns:
            Number of articles rolled back
        """
        stmt = select(IngestionProgress).where(
            IngestionProgress.status == IngestionStatus.PROCESSING.value
        )
        result = await self.session.execute(stmt)
        incomplete_records = result.scalars().all()

        count = 0
        for record in incomplete_records:
            record.status = IngestionStatus.PENDING.value
            count += 1

        await self.session.commit()
        return count
