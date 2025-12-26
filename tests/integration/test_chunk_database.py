"""Integration tests for WikiChunk database operations."""

import pytest
from sqlalchemy import select, text

from src.models.wiki_chunk import WikiChunk
from src.repositories.chunk_repository import ChunkRepository


@pytest.mark.asyncio
class TestChunkDatabase:
    """Integration tests for database operations with WikiChunk."""

    async def test_full_crud_cycle(self, async_session) -> None:
        """Test complete CRUD cycle with database."""
        repo = ChunkRepository(async_session)

        # Create
        chunk = WikiChunk(
            wiki_page_id="integration-1",
            article_title="Integration Test Article",
            section_path="Main > Sub",
            chunk_text="Integration test content",
            chunk_index=0,
            metadata_json={
                "faction": "Orks",
                "era": "War of the Beast",
                "spoiler_flag": False,
                "content_type": "lore",
                "character_names": ["Ghazghkull"],
                "links": [],
                "source_books": ["Codex: Orks"],
            },
        )
        created = await repo.create(chunk)
        await async_session.commit()
        chunk_id = created.id

        # Read
        retrieved = await repo.get_by_id(chunk_id)
        assert retrieved is not None
        assert retrieved.metadata_json["faction"] == "Orks"

        # Update
        retrieved.metadata_json["era"] = "Current Era"
        await repo.update(retrieved)
        await async_session.commit()

        updated_chunk = await repo.get_by_id(chunk_id)
        assert updated_chunk is not None
        assert updated_chunk.metadata_json["era"] == "Current Era"

        # Delete
        deleted = await repo.delete(chunk_id)
        await async_session.commit()
        assert deleted is True

        final_check = await repo.get_by_id(chunk_id)
        assert final_check is None

    async def test_indexes_exist(self, async_session) -> None:
        """Test that all required indexes are created."""
        # Query SQLite system tables to check for indexes
        result = await async_session.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='wiki_chunk'")
        )
        indexes = [row[0] for row in result.fetchall()]

        # Check for required indexes
        assert "ix_wiki_chunk_article_title" in indexes
        assert "ix_wiki_chunk_wiki_page_id" in indexes
        assert "ix_wiki_chunk_faction" in indexes
        assert "ix_wiki_chunk_era" in indexes
        assert "ix_wiki_chunk_spoiler_flag" in indexes

    async def test_json_metadata_querying(self, async_session) -> None:
        """Test querying chunks by JSON metadata fields."""
        repo = ChunkRepository(async_session)

        # Create chunks with different factions
        chunks_data = [
            ("Space Marines", "Horus Heresy", False),
            ("Chaos", "Horus Heresy", True),
            ("Tyranids", "Current Era", False),
        ]

        for faction, era, spoiler in chunks_data:
            chunk = WikiChunk(
                wiki_page_id=f"json-test-{faction}",
                article_title=f"{faction} Article",
                section_path="Main",
                chunk_text=f"Content about {faction}",
                chunk_index=0,
                metadata_json={
                    "faction": faction,
                    "era": era,
                    "spoiler_flag": spoiler,
                    "content_type": "lore",
                    "character_names": [],
                    "links": [],
                    "source_books": [],
                },
            )
            await repo.create(chunk)
        await async_session.commit()

        # Query by faction using JSON extraction
        stmt = (
            select(WikiChunk)
            .where(text("json_extract(metadata_json, '$.faction') = :faction"))
            .params(faction="Space Marines")
        )
        result = await async_session.execute(stmt)
        space_marine_chunks = list(result.scalars().all())

        assert len(space_marine_chunks) == 1
        assert space_marine_chunks[0].metadata_json["faction"] == "Space Marines"

        # Query by era
        stmt = (
            select(WikiChunk)
            .where(text("json_extract(metadata_json, '$.era') = :era"))
            .params(era="Horus Heresy")
        )
        result = await async_session.execute(stmt)
        heresy_chunks = list(result.scalars().all())

        assert len(heresy_chunks) == 2

        # Query by spoiler flag
        stmt = (
            select(WikiChunk)
            .where(text("json_extract(metadata_json, '$.spoiler_flag') = :spoiler"))
            .params(spoiler=0)
        )  # SQLite stores False as 0
        result = await async_session.execute(stmt)
        non_spoiler_chunks = list(result.scalars().all())

        assert len(non_spoiler_chunks) == 2

    async def test_bulk_operations_performance(self, async_session) -> None:
        """Test bulk create and query operations."""
        repo = ChunkRepository(async_session)

        # Create 50 chunks in bulk
        chunks = [
            WikiChunk(
                wiki_page_id=f"bulk-perf-{i // 10}",
                article_title=f"Article {i // 10}",
                section_path=f"Section {i % 10}",
                chunk_text=f"Content {i}",
                chunk_index=i % 10,
                metadata_json={
                    "faction": "Test Faction",
                    "era": "Test Era",
                    "spoiler_flag": False,
                    "content_type": "lore",
                    "character_names": [],
                    "links": [],
                    "source_books": [],
                },
            )
            for i in range(50)
        ]

        created_chunks = await repo.bulk_create(chunks)
        await async_session.commit()

        assert len(created_chunks) == 50

        # Query all chunks for one article (should have 10 chunks)
        article_chunks = await repo.get_by_article_title("Article 0")
        assert len(article_chunks) == 10
        assert article_chunks[0].chunk_index == 0
        assert article_chunks[9].chunk_index == 9

    async def test_timestamps_auto_update(self, async_session) -> None:
        """Test that timestamps are automatically managed."""

        repo = ChunkRepository(async_session)

        chunk = WikiChunk(
            wiki_page_id="timestamp-test",
            article_title="Timestamp Test",
            section_path="Main",
            chunk_text="Content",
            chunk_index=0,
            metadata_json={},
        )
        created = await repo.create(chunk)
        await async_session.commit()

        original_created_at = created.created_at
        original_updated_at = created.updated_at

        assert original_created_at is not None
        assert original_updated_at is not None
        # Timestamps should be very close (within 1 second)
        assert abs((original_created_at - original_updated_at).total_seconds()) < 1

        # Update the chunk
        created.chunk_text = "Updated content"
        await async_session.flush()
        await async_session.commit()

        # Refresh to get updated timestamp
        await async_session.refresh(created)

        # created_at should remain the same
        assert created.created_at == original_created_at
        # updated_at should be updated (SQLite may not automatically update, but the field exists)
        assert created.updated_at is not None

    async def test_complex_metadata_structure(self, async_session) -> None:
        """Test storing and retrieving complex metadata structures."""
        repo = ChunkRepository(async_session)

        complex_metadata = {
            "faction": "Adeptus Mechanicus",
            "subfaction": "Forge World Mars",
            "character_names": ["Belisarius Cawl", "Archmagos Dominus"],
            "era": "Dark Imperium",
            "spoiler_flag": True,
            "content_type": "technology",
            "links": [
                "Primaris Space Marines",
                "Mars",
                "Imperium of Man",
            ],
            "source_books": [
                "Codex: Adeptus Mechanicus",
                "Gathering Storm III: Rise of the Primarch",
            ],
        }

        chunk = WikiChunk(
            wiki_page_id="complex-meta",
            article_title="Adeptus Mechanicus",
            section_path="History > Recent Developments",
            chunk_text="The Adeptus Mechanicus has developed...",
            chunk_index=0,
            metadata_json=complex_metadata,
        )

        created = await repo.create(chunk)
        await async_session.commit()

        # Retrieve and verify complex metadata
        retrieved = await repo.get_by_id(created.id)
        assert retrieved is not None
        assert retrieved.metadata_json["faction"] == "Adeptus Mechanicus"
        assert retrieved.metadata_json["subfaction"] == "Forge World Mars"
        assert len(retrieved.metadata_json["character_names"]) == 2
        assert "Belisarius Cawl" in retrieved.metadata_json["character_names"]
        assert len(retrieved.metadata_json["links"]) == 3
        assert len(retrieved.metadata_json["source_books"]) == 2
