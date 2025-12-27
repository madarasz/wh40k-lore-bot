"""Repository for WikiChunk database operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import attributes

from src.models.wiki_chunk import WikiChunk


class ChunkRepository:
    """Repository for WikiChunk CRUD operations.

    This repository implements the repository pattern for database access,
    providing a clean interface for WikiChunk operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the chunk repository.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(self, chunk: WikiChunk) -> WikiChunk:
        """Create a new wiki chunk in the database.

        Args:
            chunk: WikiChunk instance to create

        Returns:
            Created WikiChunk instance with database-generated fields populated
        """
        self.session.add(chunk)
        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk

    async def get_by_id(self, chunk_id: str) -> WikiChunk | None:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: UUID string of the chunk

        Returns:
            WikiChunk instance if found, None otherwise
        """
        stmt = select(WikiChunk).where(WikiChunk.id == chunk_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_article_title(self, title: str) -> list[WikiChunk]:
        """Retrieve all chunks for a specific article.

        Args:
            title: Article title to search for

        Returns:
            List of WikiChunk instances for the article, ordered by chunk_index
        """
        stmt = (
            select(WikiChunk)
            .where(WikiChunk.article_title == title)
            .order_by(WikiChunk.chunk_index)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_wiki_page_id(self, page_id: str) -> list[WikiChunk]:
        """Retrieve all chunks for a specific wiki page ID.

        Args:
            page_id: Wiki page ID from XML export

        Returns:
            List of WikiChunk instances for the page, ordered by chunk_index
        """
        stmt = (
            select(WikiChunk)
            .where(WikiChunk.wiki_page_id == page_id)
            .order_by(WikiChunk.chunk_index)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, chunk: WikiChunk) -> WikiChunk:
        """Update an existing wiki chunk.

        Args:
            chunk: WikiChunk instance with updated fields

        Returns:
            Updated WikiChunk instance
        """
        # Mark metadata_json as modified to detect changes in mutable dict
        attributes.flag_modified(chunk, "metadata_json")

        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk

    async def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by its ID.

        Args:
            chunk_id: UUID string of the chunk to delete

        Returns:
            True if chunk was deleted, False if not found
        """
        chunk = await self.get_by_id(chunk_id)
        if chunk is None:
            return False

        await self.session.delete(chunk)
        await self.session.flush()
        return True

    async def bulk_create(self, chunks: list[WikiChunk]) -> list[WikiChunk]:
        """Create multiple chunks in a single batch.

        Args:
            chunks: List of WikiChunk instances to create

        Returns:
            List of created WikiChunk instances
        """
        self.session.add_all(chunks)
        await self.session.flush()
        for chunk in chunks:
            await self.session.refresh(chunk)
        return chunks
