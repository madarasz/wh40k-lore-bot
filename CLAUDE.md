# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WH40K Lore Bot is a Discord bot that answers Warhammer 40,000 lore questions using RAG (Retrieval-Augmented Generation). It parses the WH40K Fandom Wiki XML export, creates searchable embeddings, and uses hybrid retrieval (vector + BM25) with multi-LLM provider support.

## Development Commands

```bash
# Install dependencies
poetry install

# Run the bot
poetry run python -m src

# Run tests with coverage
poetry run pytest

# Run single test file
poetry run pytest tests/unit/test_logger.py -v

# Run integration tests (costs money - uses real APIs)
poetry run pytest -m integration

# Linting and formatting
poetry run ruff check . --fix
poetry run ruff format .

# Type checking
poetry run mypy src/

# All pre-commit hooks
poetry run pre-commit run --all-files

# Database migrations
poetry run alembic upgrade head
poetry run alembic revision --autogenerate -m "description"
poetry run alembic downgrade -1
```

## CLI Tools

### Data Preparation
```bash
poetry run parse-wiki <xml_path>              # Parse wiki XML to markdown
poetry run build-test-bed <xml_path>          # Build test subset via BFS
```

### Ingestion Pipeline
```bash
poetry run ingest                             # Full pipeline (chunk + embed + store)
poetry run ingest --wiki-ids-file data/test-bed-pages.txt  # Process subset
poetry run ingest --dry-run                   # Parse without embeddings
```

### Database Analysis
```bash
poetry run stats-db                           # Chroma statistics
poetry run stats-markdown                     # Archive statistics
poetry run show-chunk <chunk_id>              # View chunk details
poetry run db-health                          # Check DB consistency
poetry run purge-db --force                   # Delete all chunks
```

## Architecture

The project follows a **Central Orchestrator Pattern** with layered architecture:

```
Entry Points (adapters/) → Query Orchestrator → RAG Pipeline → Response
     ↓                           ↓                    ↓
  Discord/CLI            Validate/Rate Limit    Hybrid Retrieval
                                                Metadata Filter
                                                LLM Generation
```

### Key Directories
- `src/adapters/` - Entry points (Discord bot, CLI)
- `src/orchestration/` - Central query orchestrator
- `src/rag/` - RAG pipeline (hybrid retrieval, filtering)
- `src/llm/` - Multi-provider LLM integration
- `src/repositories/` - Data access layer (VectorRepository, BM25Repository)
- `src/ingestion/` - Wiki XML parsing, chunking, embedding
- `src/models/` - Domain models (WikiChunk, QueryLog)
- `src/cli/` - CLI command implementations

### Data Flow
1. Wiki XML → `parse-wiki` → Markdown Archive (`data/markdown-archive/`)
2. Markdown → `ingest` → Chunks + Embeddings → ChromaDB (`data/chroma-db/`)
3. Query → Hybrid Retrieval (Vector + BM25) → Context → LLM → Response

## Coding Standards

### Critical Rules
- **Never use print()** - use structlog logger
- **All database ops via repositories** - not direct session access
- **All LLM calls via router** - not direct OpenAI/Anthropic clients
- **Use pathlib** for file operations, not os.path
- **Use custom exceptions** from `src/utils/exceptions.py`
- **Await all async functions** - no blocking calls in async code
- **No fallback logic** - fail fast, let errors propagate
- **SQL changes via Alembic only** - never raw ALTER TABLE
- **Only top level imports** - no imports inside methods, no conditional imports

### Style
- Line length: 100 characters
- Type hints required for all function signatures
- Docstrings: Google style, required for public APIs
- Imports: stdlib → third-party → local (ruff handles ordering)

### Commits
```
<type>: <description>

Types: feat, fix, refactor, test, docs, chore
```

## Tech Stack

- **Python 3.11** with Poetry
- **discord.py** - Discord bot framework
- **ChromaDB** - Vector database (embedded mode)
- **rank-bm25** - BM25 keyword search
- **SQLAlchemy 2.0 + Alembic** - ORM and migrations
- **OpenAI/Anthropic/Gemini** - LLM providers
- **structlog** - Structured JSON logging
- **ruff** - Linting and formatting
- **mypy** - Type checking (strict mode)
- **pytest + pytest-asyncio** - Testing
