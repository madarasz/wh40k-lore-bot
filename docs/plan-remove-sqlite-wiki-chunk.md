# Plan: Remove Redundant SQLite wiki_chunk Storage

## Summary

The SQLite `wiki_chunk` table fully duplicates data stored in ChromaDB. This plan removes the redundant storage while keeping:
- **ChromaDB** - Primary storage for chunks (embeddings + text + metadata)
- **ingestion_progress** table - For future admin/state tracking

## Confirmed Redundancy

| Data | SQLite wiki_chunk | ChromaDB | Redundant? |
|------|-------------------|----------|-----------|
| chunk_text | Yes (Text column) | Yes (documents) | **YES** |
| wiki_page_id | Yes (indexed) | Yes (metadata) | **YES** |
| article_title | Yes (indexed) | Yes (metadata) | **YES** |
| section_path | Yes | Yes (metadata) | **YES** |
| chunk_index | Yes | Yes (metadata) | **YES** |
| metadata (faction, era, etc.) | Yes (JSON) | Yes (metadata) | **YES** |
| embeddings | No | Yes | No - ChromaDB only |

## Decisions

- **Migration approach**: Delete migration file (not create drop migration)
- **WikiChunk class**: Remove entirely (not keep as DTO)
- **CLI tools**: Update to query ChromaDB (not remove)

---

## Files Analysis

### Already ChromaDB-Only (No Changes Needed)
- `src/cli/show_chunk.py` ✓
- `src/cli/stats_db.py` ✓

### To Delete

| File | Reason |
|------|--------|
| `src/models/wiki_chunk.py` | SQLAlchemy ORM model - removing entirely |
| `src/repositories/chunk_repository.py` | SQLite repository - no longer needed |
| `src/migrations/versions/2025_12_26_2316-2ac2f41edf9c_create_wiki_chunk_table.py` | Creates redundant table |
| `tests/unit/test_wiki_chunk.py` | Tests for deleted model |
| `tests/unit/test_chunk_repository.py` | Tests for deleted repository |

### To Update

| File | Changes |
|------|---------|
| `src/models/__init__.py` | Remove WikiChunk export |
| `src/repositories/__init__.py` | Remove ChunkRepository export |
| `src/rag/vector_store.py` | Replace WikiChunk with TypedDict/dict |
| `src/ingestion/pipeline.py` | Use dict instead of WikiChunk |
| `src/cli/delete_chunk.py` | Remove SQLite logic, ChromaDB only |
| `src/cli/db_health.py` | Remove SQLite checks, ChromaDB only |
| `src/cli/purge_db.py` | Remove SQLite logic, ChromaDB only |
| `tests/unit/test_vector_store.py` | Update for new dict-based API |
| `tests/unit/test_cli_db_health.py` | Remove SQLite mock/assertions |
| `tests/unit/test_cli_delete_chunk.py` | Remove SQLite mock/assertions |
| `tests/unit/test_cli_purge_db.py` | Remove SQLite mock/assertions |
| `tests/integration/test_chunk_database.py` | Update or delete based on content |

### Documentation to Update
- `docs/architecture/source-tree.md` - Remove wiki_chunk references
- `docs/stories/1.2.database-schema.md` - Update database schema docs
- `docs/stories/1.8.chroma-vector-database.md` - Note as sole chunk storage

---

## Implementation Steps

### Step 1: Create ChunkData TypedDict
Create a simple TypedDict in `src/rag/vector_store.py` to replace WikiChunk:

```python
class ChunkData(TypedDict):
    id: str
    wiki_page_id: str
    article_title: str
    section_path: str
    chunk_text: str
    chunk_index: int
    metadata: dict[str, Any]  # Contains faction, era, spoiler_flag, etc.
```

### Step 2: Update vector_store.py
- Change `add_chunks(chunks: list[WikiChunk], ...)` → `add_chunks(chunks: list[ChunkData], ...)`
- Change `query(...) -> list[tuple[WikiChunk, float]]` → `list[tuple[ChunkData, float]]`
- Change `get_by_id(...) -> WikiChunk | None` → `ChunkData | None`
- Update `_chunk_to_metadata()` and `_metadata_to_chunk()` for ChunkData
- Remove `from src.models.wiki_chunk import WikiChunk`

### Step 3: Update ingestion/pipeline.py
- Create dict/ChunkData directly instead of WikiChunk objects
- Remove `from src.models.wiki_chunk import WikiChunk`

### Step 4: Update CLI Tools

**delete_chunk.py:**
- Remove SQLite imports (create_engine, delete, select, WikiChunk)
- Remove `_get_database_url()`, `_check_sqlite()` functions
- Update `_perform_deletion()` to only delete from ChromaDB
- Simplify output messages

**db_health.py:**
- Remove SQLite imports and WikiChunk
- Remove `_check_sqlite()` function
- Remove consistency check (no longer applicable)
- Rename to just check ChromaDB health

**purge_db.py:**
- Remove SQLite imports and WikiChunk
- Update `_get_counts()` to only return ChromaDB count
- Update `_perform_purge()` to only purge ChromaDB

### Step 5: Delete Files
- `src/models/wiki_chunk.py`
- `src/repositories/chunk_repository.py`
- `src/migrations/versions/2025_12_26_2316-2ac2f41edf9c_create_wiki_chunk_table.py`
- `tests/unit/test_wiki_chunk.py`
- `tests/unit/test_chunk_repository.py`

### Step 6: Update __init__.py Files
- `src/models/__init__.py`: Remove WikiChunk from exports
- `src/repositories/__init__.py`: Remove ChunkRepository from exports

### Step 7: Update Tests
- `tests/unit/test_vector_store.py`: Update to use ChunkData
- `tests/unit/test_cli_db_health.py`: Remove SQLite mocks
- `tests/unit/test_cli_delete_chunk.py`: Remove SQLite mocks
- `tests/unit/test_cli_purge_db.py`: Remove SQLite mocks
- `tests/integration/test_chunk_database.py`: Check if still relevant

### Step 8: Update Documentation
- `docs/architecture/source-tree.md`: Remove wiki_chunk model reference
- `docs/stories/1.2.database-schema.md`: Update schema docs
- Other docs as needed based on grep for "wiki_chunk"

### Step 9: Run Tests & Verify
- Run `poetry run pytest` to ensure all tests pass
- Run `poetry run mypy src` for type checking
- Manual verification of CLI tools

---

## Notes

- The `ingestion_progress` table is **kept** for future admin/state tracking
- ChromaDB becomes the **single source of truth** for chunk data
- This reduces storage requirements and eliminates sync concerns
- Existing database will need manual cleanup (drop wiki_chunk table) or recreate fresh
