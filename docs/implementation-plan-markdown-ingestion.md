# Implementation Plan: Markdown-Based Ingestion

## Status: In Progress

### Completed Tasks (Phase 1: Core Infrastructure)

1. **Created `src/ingestion/markdown_loader.py`**
   - `MarkdownLoader` class with YAML frontmatter parsing
   - `load_all()` method with wiki_ids filtering
   - `load_file()` method for single files
   - Statistics tracking (files_loaded, files_skipped)

2. **Updated `src/models/ingestion_progress.py`**
   - Added `article_last_updated` field for change detection

3. **Created Alembic migration**
   - `2025_12_31_1136-cdf0612a18b6_add_article_last_updated_to_ingestion_.py`
   - Adds `article_last_updated` column with index

4. **Updated `src/repositories/ingestion_progress_repository.py`**
   - Added `get_article_last_updated(article_id)` method
   - Added `should_process_article(article_id, last_updated)` method
   - Added `upsert_article_progress(article_id, last_updated, batch_number, status)` method

5. **Updated `src/rag/vector_store.py`**
   - Added `wiki_page_id` to `ChunkMetadata` TypedDict
   - Added `wiki_page_id` to `_chunk_to_metadata()` output
   - Added `delete_by_wiki_page_id(wiki_page_id)` method
   - Updated `_metadata_to_chunk()` to read `wiki_page_id`

6. **Updated `src/models/wiki_chunk.py`**
   - Added `generate_chunk_id(wiki_page_id, chunk_index)` function
   - Changed ID generation to deterministic format: `{wiki_page_id}_{chunk_index}`
   - Updated `__init__` to auto-generate ID from wiki_page_id + chunk_index

---

## Remaining Tasks

### Phase 2: Pipeline Refactor (Task 7)

**File: `src/ingestion/pipeline.py`**

Replace the entire file with markdown-based ingestion:

```python
"""Complete ingestion pipeline orchestration for markdown-based wiki data processing."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from tqdm import tqdm

from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.markdown_loader import MarkdownLoader
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import WikiArticle
from src.ingestion.text_chunker import MarkdownChunker
from src.models.wiki_chunk import WikiChunk
from src.rag.vector_store import ChromaVectorStore
from src.utils.exceptions import IngestionError

logger = structlog.get_logger(__name__)


@dataclass
class IngestionStatistics:
    """Statistics for the ingestion pipeline run."""
    articles_processed: int = 0
    articles_skipped: int = 0  # NEW: unchanged articles
    articles_failed: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0  # NEW: for re-ingestion
    embeddings_generated: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "articles_processed": self.articles_processed,
            "articles_skipped": self.articles_skipped,
            "articles_failed": self.articles_failed,
            "chunks_created": self.chunks_created,
            "chunks_deleted": self.chunks_deleted,
            "embeddings_generated": self.embeddings_generated,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": round(self.total_cost, 4),
            "duration_seconds": int(self.duration_seconds),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
        }


class IngestionPipeline:
    """Orchestrates end-to-end wiki ingestion pipeline from markdown files.

    Processes markdown files through the complete pipeline:
    1. Load markdown files from archive
    2. Check for changes (skip unchanged articles)
    3. Delete old chunks if article changed
    4. Chunk markdown articles
    5. Extract metadata from chunks
    6. Generate embeddings (batched)
    7. Store chunks + embeddings + metadata in Chroma

    Supports batch processing, change detection, and resumable execution.
    """

    def __init__(
        self,
        archive_path: str | Path | None = None,
        chroma_path: str | None = None,
    ) -> None:
        """Initialize ingestion pipeline components.

        Args:
            archive_path: Path to markdown archive directory
            chroma_path: Path to Chroma vector database
        """
        self.logger = structlog.get_logger(__name__)

        # Initialize components
        self.loader = MarkdownLoader(archive_path)
        self.chunker = MarkdownChunker()
        self.metadata_extractor = MetadataExtractor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = ChromaVectorStore(
            storage_path=chroma_path or ChromaVectorStore.DEFAULT_STORAGE_PATH
        )

        # Configure paths
        self.archive_path = Path(archive_path) if archive_path else Path("data/markdown-archive")
        self.logs_path = Path("logs")
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = IngestionStatistics()

        self.logger.info(
            "ingestion_pipeline_initialized",
            archive_path=str(self.archive_path),
            chroma_path=chroma_path or ChromaVectorStore.DEFAULT_STORAGE_PATH,
        )

    def run(
        self,
        wiki_ids: list[str] | None = None,
        batch_size: int = 100,
        dry_run: bool = False,
        force: bool = False,
    ) -> IngestionStatistics:
        """Run complete ingestion pipeline on markdown archive.

        Args:
            wiki_ids: Optional list of wiki IDs to process (None = all files)
            batch_size: Number of articles to process per batch
            dry_run: If True, skip embedding generation and vector storage
            force: If True, re-ingest all articles regardless of last_updated

        Returns:
            IngestionStatistics object with processing metrics
        """
        # Implementation here...
        pass
```

**Key Changes:**
- Replace `WikiXMLParser` with `MarkdownLoader`
- Remove `save_markdown_file` (articles already in markdown)
- Add `articles_skipped` and `chunks_deleted` to statistics
- Add `force` parameter to bypass change detection
- For each article:
  1. Check `should_process_article()` (skip if unchanged)
  2. If processing: delete old chunks first via `delete_by_wiki_page_id()`
  3. Then chunk, embed, store as before
  4. Update progress via `upsert_article_progress()`

---

### Phase 3: CLI Commands (Tasks 8-13)

#### Task 8: Create `src/cli/ingest.py`

```python
"""CLI command for running the markdown-based ingestion pipeline."""

from pathlib import Path

import click
import structlog

from src.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)


@click.command()
@click.option(
    "--archive-path",
    type=click.Path(exists=True, path_type=Path),
    default="data/markdown-archive",
    help="Path to markdown archive directory (default: data/markdown-archive)",
)
@click.option(
    "--wiki-ids-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to file containing wiki IDs to process (one per line)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of articles to process per batch (default: 100)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Process without generating embeddings or storing (useful for testing)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-ingest all articles regardless of last_updated timestamp",
)
@click.option(
    "--chroma-path",
    type=str,
    default=None,
    help="Path to Chroma vector database (default: data/chroma-db/)",
)
def ingest(
    archive_path: Path,
    wiki_ids_file: Path | None,
    batch_size: int,
    dry_run: bool,
    force: bool,
    chroma_path: str | None,
) -> None:
    """Ingest markdown articles into vector database.

    Reads markdown files from the archive directory and processes them
    through the full pipeline: chunk -> extract metadata -> embed -> store.

    Unchanged articles (same last_updated timestamp) are skipped unless --force is used.

    Examples:

        \b
        # Basic usage - process entire archive
        poetry run ingest

        \b
        # Process specific articles
        poetry run ingest --wiki-ids-file data/test-bed-pages.txt

        \b
        # Force re-ingest all articles
        poetry run ingest --force

        \b
        # Dry run (no embeddings, for testing)
        poetry run ingest --dry-run --batch-size 10
    """
    # Implementation...
```

#### Task 9: Create `src/cli/chunk.py`

```python
"""CLI command for chunking markdown articles."""

@click.command()
@click.option("--archive-path", ...)
@click.option("--wiki-ids-file", ...)
@click.option("--output", type=click.Path(path_type=Path), default="data/chunks.json")
def chunk(...):
    """Chunk markdown articles from archive.

    Outputs chunks as JSON for the embed step.
    """
```

#### Task 10: Create `src/cli/embed.py`

```python
"""CLI command for generating embeddings."""

@click.command()
@click.argument("chunks_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", default="data/embeddings.json")
@click.option("--batch-size", default=100)
def embed(...):
    """Generate embeddings for chunks.

    Reads chunks JSON and outputs embeddings JSON.
    """
```

#### Task 11: Create `src/cli/store.py`

```python
"""CLI command for storing embeddings in vector database."""

@click.command()
@click.argument("embeddings_file", type=click.Path(exists=True, path_type=Path))
@click.option("--chroma-path", default="data/chroma-db/")
def store(...):
    """Store embeddings in Chroma vector database."""
```

#### Task 12: Delete `src/cli/ingest_wiki.py`

Simply delete the file - no deprecation, clean removal.

#### Task 13: Update `pyproject.toml`

```toml
[tool.poetry.scripts]
# XML utilities (data preparation)
parse-wiki = "src.cli.parse_wiki:parse_wiki"
build-test-bed = "src.cli.build_test_bed:build_test_bed"

# Ingestion commands (markdown-based)
ingest = "src.cli.ingest:ingest"
chunk = "src.cli.chunk:chunk"
embed = "src.cli.embed:embed"
store = "src.cli.store:store"
```

Remove the `ingest-wiki` entry entirely.

---

### Phase 4: Testing (Tasks 14-17)

#### Task 14: Unit tests for MarkdownLoader

**File: `tests/unit/test_markdown_loader.py`**

```python
import pytest
from pathlib import Path
from src.ingestion.markdown_loader import MarkdownLoader

class TestMarkdownLoader:
    def test_load_file_with_valid_frontmatter(self, tmp_path):
        # Create test markdown file
        md_content = """---
title: Test Article
wiki_id: '123'
last_updated: '2025-01-01T00:00:00Z'
word_count: 100
---

# Test Content
This is test content.
"""
        test_file = tmp_path / "test.md"
        test_file.write_text(md_content)

        loader = MarkdownLoader(tmp_path)
        article = loader.load_file(test_file)

        assert article is not None
        assert article.title == "Test Article"
        assert article.wiki_id == "123"
        assert article.last_updated == "2025-01-01T00:00:00Z"

    def test_load_file_missing_frontmatter(self, tmp_path):
        # Should return None
        pass

    def test_load_all_with_filter(self, tmp_path):
        # Test wiki_ids filtering
        pass

    def test_load_all_empty_directory(self, tmp_path):
        # Should yield nothing
        pass
```

#### Task 15: Unit tests for change detection

**File: `tests/unit/test_change_detection.py`**

```python
import pytest
from src.repositories.ingestion_progress_repository import IngestionProgressRepository

class TestChangeDetection:
    async def test_should_process_new_article(self, db_session):
        repo = IngestionProgressRepository(db_session)
        # New article should be processed
        result = await repo.should_process_article("999", "2025-01-01T00:00:00Z")
        assert result is True

    async def test_should_skip_unchanged_article(self, db_session):
        repo = IngestionProgressRepository(db_session)
        # First, mark as completed with timestamp
        await repo.upsert_article_progress("123", "2025-01-01T00:00:00Z", 1)
        # Same timestamp should skip
        result = await repo.should_process_article("123", "2025-01-01T00:00:00Z")
        assert result is False

    async def test_should_process_changed_article(self, db_session):
        repo = IngestionProgressRepository(db_session)
        # Mark with old timestamp
        await repo.upsert_article_progress("123", "2025-01-01T00:00:00Z", 1)
        # New timestamp should process
        result = await repo.should_process_article("123", "2025-06-01T00:00:00Z")
        assert result is True
```

#### Tasks 16-17: Integration tests

Test the full flow from markdown to vector DB, including re-ingestion.

---

### Phase 5: Documentation (Tasks 18-20)

#### Task 18: Update CLI.md

- Document new commands: `ingest`, `chunk`, `embed`, `store`
- Remove `ingest-wiki` documentation
- Update examples

#### Task 19: Update docs/ingestion-guide.md

- Update architecture diagram (markdown-first)
- Update workflow description
- Add change detection explanation
- Update examples

#### Task 20: Update story files

- `docs/stories/1.10.ingestion-pipeline.md` - Mark as updated
- `docs/stories/1.11.data-quality-validation.md` - Update notes
- `docs/stories/1.12.ingestion-documentation.md` - Update references

---

## Intermediate Data Formats

### chunks.json (output of `chunk` command)

```json
{
  "version": "1.0",
  "created_at": "2025-12-31T10:00:00Z",
  "source_archive": "data/markdown-archive",
  "total_chunks": 523,
  "chunks": [
    {
      "wiki_page_id": "58",
      "article_title": "Blood Angels",
      "last_updated": "2025-06-26T00:04:10Z",
      "section_path": "History > Horus Heresy",
      "chunk_index": 0,
      "chunk_text": "The Blood Angels are one of the...",
      "metadata": {
        "faction": "Space Marines",
        "eras": ["Horus Heresy"],
        "content_type": "lore",
        "spoiler_flag": false
      }
    }
  ]
}
```

### embeddings.json (output of `embed` command)

```json
{
  "version": "1.0",
  "created_at": "2025-12-31T11:00:00Z",
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "total_embeddings": 523,
  "cost_usd": 0.05,
  "tokens_used": 125000,
  "embeddings": [
    {
      "chunk_id": "58_0",
      "wiki_page_id": "58",
      "article_title": "Blood Angels",
      "chunk_text": "The Blood Angels are one of the...",
      "metadata": {...},
      "embedding": [0.123, -0.456, ...]
    }
  ]
}
```

---

## Change Detection Flow (Detailed)

```
ingest command starts
│
├─ Initialize MarkdownLoader, Pipeline, VectorStore
│
├─ For each article from loader.load_all():
│   │
│   ├─ Check should_process_article(wiki_id, last_updated)
│   │   │
│   │   ├─ If False (unchanged):
│   │   │   ├─ Log: "Skipping unchanged article: {title}"
│   │   │   ├─ stats.articles_skipped += 1
│   │   │   └─ Continue to next article
│   │   │
│   │   └─ If True (new or changed):
│   │       │
│   │       ├─ Delete old chunks (if article exists):
│   │       │   ├─ deleted = vector_store.delete_by_wiki_page_id(wiki_id)
│   │       │   ├─ Log: "Deleted {deleted} old chunks for {title}"
│   │       │   └─ stats.chunks_deleted += deleted
│   │       │
│   │       ├─ Process article:
│   │       │   ├─ chunks = chunker.chunk_markdown(content, title)
│   │       │   ├─ For each chunk: extract_metadata()
│   │       │   ├─ embeddings = embedding_generator.generate_embeddings(chunks)
│   │       │   └─ vector_store.add_chunks(wiki_chunks, embeddings)
│   │       │
│   │       ├─ Update progress:
│   │       │   └─ upsert_article_progress(wiki_id, last_updated, batch_num)
│   │       │
│   │       └─ stats.articles_processed += 1
│
└─ Return stats
```

---

## Critical Files Summary

| File | Status | Description |
|------|--------|-------------|
| `src/ingestion/markdown_loader.py` | ✅ Done | Load markdown with frontmatter |
| `src/models/ingestion_progress.py` | ✅ Done | Added article_last_updated |
| `src/repositories/ingestion_progress_repository.py` | ✅ Done | Change detection methods |
| `src/rag/vector_store.py` | ✅ Done | delete_by_wiki_page_id |
| `src/models/wiki_chunk.py` | ✅ Done | Deterministic IDs |
| `src/ingestion/pipeline.py` | 🔄 In Progress | Major refactor needed |
| `src/cli/ingest.py` | ⏳ Pending | New main command |
| `src/cli/chunk.py` | ⏳ Pending | Step 1 command |
| `src/cli/embed.py` | ⏳ Pending | Step 2 command |
| `src/cli/store.py` | ⏳ Pending | Step 3 command |
| `src/cli/ingest_wiki.py` | ❌ To Delete | Old command |
| `pyproject.toml` | ⏳ Pending | Update entry points |
