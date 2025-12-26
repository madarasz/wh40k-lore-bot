"""Unit tests for ChunkRepository."""

import pytest

from src.models.wiki_chunk import WikiChunk
from src.repositories.chunk_repository import ChunkRepository


@pytest.mark.asyncio
class TestChunkRepository:
    """Test suite for ChunkRepository."""

    async def test_create_chunk(self, async_session) -> None:
        """Test creating a chunk in the database."""
        repo = ChunkRepository(async_session)

        chunk = WikiChunk(
            wiki_page_id="123",
            article_title="Test Article",
            section_path="Introduction",
            chunk_text="Test content",
            chunk_index=0,
            metadata_json={"faction": "Space Marines"},
        )

        created_chunk = await repo.create(chunk)
        await async_session.commit()

        assert created_chunk.id is not None
        assert created_chunk.wiki_page_id == "123"
        assert created_chunk.article_title == "Test Article"

    async def test_get_by_id(self, async_session) -> None:
        """Test retrieving a chunk by ID."""
        repo = ChunkRepository(async_session)

        chunk = WikiChunk(
            wiki_page_id="456",
            article_title="Get By ID Test",
            section_path="Test",
            chunk_text="Content",
            chunk_index=0,
            metadata_json={},
        )
        created = await repo.create(chunk)
        await async_session.commit()

        retrieved = await repo.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.article_title == "Get By ID Test"

    async def test_get_by_id_not_found(self, async_session) -> None:
        """Test retrieving non-existent chunk returns None."""
        repo = ChunkRepository(async_session)

        result = await repo.get_by_id("non-existent-id")

        assert result is None

    async def test_get_by_article_title(self, async_session) -> None:
        """Test retrieving all chunks for an article."""
        repo = ChunkRepository(async_session)

        # Create multiple chunks for same article
        for i in range(3):
            chunk = WikiChunk(
                wiki_page_id="789",
                article_title="Multi-Chunk Article",
                section_path=f"Section {i}",
                chunk_text=f"Content {i}",
                chunk_index=i,
                metadata_json={},
            )
            await repo.create(chunk)
        await async_session.commit()

        chunks = await repo.get_by_article_title("Multi-Chunk Article")

        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2

    async def test_get_by_wiki_page_id(self, async_session) -> None:
        """Test retrieving chunks by wiki page ID."""
        repo = ChunkRepository(async_session)

        # Create chunks for same wiki page
        for i in range(2):
            chunk = WikiChunk(
                wiki_page_id="wiki-999",
                article_title=f"Article {i}",
                section_path="Main",
                chunk_text="Content",
                chunk_index=i,
                metadata_json={},
            )
            await repo.create(chunk)
        await async_session.commit()

        chunks = await repo.get_by_wiki_page_id("wiki-999")

        assert len(chunks) == 2
        assert all(c.wiki_page_id == "wiki-999" for c in chunks)

    async def test_update_chunk(self, async_session) -> None:
        """Test updating a chunk."""
        repo = ChunkRepository(async_session)

        chunk = WikiChunk(
            wiki_page_id="101",
            article_title="Original Title",
            section_path="Test",
            chunk_text="Original content",
            chunk_index=0,
            metadata_json={},
        )
        created = await repo.create(chunk)
        await async_session.commit()

        # Modify the chunk
        created.article_title = "Updated Title"
        created.chunk_text = "Updated content"

        updated = await repo.update(created)
        await async_session.commit()

        # Retrieve and verify
        retrieved = await repo.get_by_id(updated.id)
        assert retrieved is not None
        assert retrieved.article_title == "Updated Title"
        assert retrieved.chunk_text == "Updated content"

    async def test_delete_chunk(self, async_session) -> None:
        """Test deleting a chunk."""
        repo = ChunkRepository(async_session)

        chunk = WikiChunk(
            wiki_page_id="202",
            article_title="To Delete",
            section_path="Test",
            chunk_text="Content",
            chunk_index=0,
            metadata_json={},
        )
        created = await repo.create(chunk)
        await async_session.commit()

        # Delete the chunk
        result = await repo.delete(created.id)
        await async_session.commit()

        assert result is True

        # Verify it's gone
        retrieved = await repo.get_by_id(created.id)
        assert retrieved is None

    async def test_delete_nonexistent_chunk(self, async_session) -> None:
        """Test deleting non-existent chunk returns False."""
        repo = ChunkRepository(async_session)

        result = await repo.delete("non-existent-id")

        assert result is False

    async def test_bulk_create(self, async_session) -> None:
        """Test creating multiple chunks in bulk."""
        repo = ChunkRepository(async_session)

        chunks = [
            WikiChunk(
                wiki_page_id=f"bulk-{i}",
                article_title=f"Bulk Article {i}",
                section_path="Test",
                chunk_text=f"Content {i}",
                chunk_index=i,
                metadata_json={},
            )
            for i in range(5)
        ]

        created_chunks = await repo.bulk_create(chunks)
        await async_session.commit()

        assert len(created_chunks) == 5
        assert all(c.id is not None for c in created_chunks)

        # Verify all were saved
        for i in range(5):
            chunks_for_page = await repo.get_by_wiki_page_id(f"bulk-{i}")
            assert len(chunks_for_page) == 1
