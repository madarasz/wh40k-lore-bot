# Technical Assumptions

## Repository Structure: Monorepo

All components (RAG engine, Discord bot, CLI tools, tests) reside in a single repository for simplified development and deployment.

## Service Architecture

**Monolithic Service with Central Orchestrator Pattern:**
- Single Python application containing all components
- Central RAG Orchestrator exposes unified query interface
- Discord Bot Adapter is a thin client calling the orchestrator
- CLI Adapter provides offline tools (ingestion, index rebuild)
- Traditional server-based deployment (single Debian VM process)
- Simpler operations, lower latency, persistent in-memory caching

**Rationale:** For MVP with <5 servers and <1000 queries/month, a monolith is simpler to develop, deploy, and maintain than microservices. The central orchestrator pattern ensures the RAG pipeline is reusable across Discord, CLI, and tests.

## Testing Requirements

**Testing Pyramid Approach:**
- **Unit Tests:** Core business logic (chunking, metadata extraction, retrieval, ranking) with >70% coverage target
- **Integration Tests:** Critical workflows end-to-end:
  - XML parsing → markdown → chunking → embedding → vector DB storage
  - Query → retrieval → ranking → LLM generation → response
  - Discord command handling → orchestrator → response
- **Manual Testing:** Discord bot interaction testing (conversational flows, error handling)
- **Testing Tools:** pytest, pytest-cov, pytest-mock, pytest-asyncio
- **End-to-End CLI Tests:** Complete workflow validation using minimal real data:
  - Separate test suite (`tests/e2e/`) with dedicated marker (`@pytest.mark.e2e`)
  - Not run by default (excluded from coverage and CI fast paths)
  - Run explicitly via `poetry run pytest -m e2e`
  - Uses minimal XML test file (~10 pages) stored in `./test-data/`
  - Validates all CLI commands: build-test-bed, parse-wiki, chunk, embed, store, build-bm25, ingest
  - Sequential test execution with shared artifacts (test-bed → markdown → chunks → embeddings → storage)
  - Skips dependent tests if prerequisites fail (graceful degradation)
  - Tests requiring external APIs (embed, ingest) conditionally skip if credentials unavailable
  - Outputs stored in `./test-data/e2e-outputs/` for post-test inspection
  - Full pipeline testing including real embedding generation and storage

**Rationale:** Unit tests catch logic bugs early, integration tests validate end-to-end workflows, manual testing ensures Discord UX quality. E2E tests catch integration issues between CLI commands and validate the full data pipeline with realistic (but minimal) data. Separating them from unit tests keeps CI fast while providing comprehensive validation before releases.

## Additional Technical Assumptions and Requests

**Programming Language:**
- **Python 3.11+** for all components (Discord.py, async/await support, modern type hints)

**Data Source:**
- **WH40K Fandom Wiki XML Export** (~173MB MediaWiki XML dump format)
- Downloaded manually from `https://warhammer40k.fandom.com`
- Legal: CC-BY-SA licensed, non-commercial use, attribution required
- **No web scraping required** - single offline XML file parsing

**Vector Database:**
- **MVP:** Chroma (embedded mode, persistent to disk, zero server setup)
- **Future:** Qdrant (if advanced metadata filtering or scaling needed)

**BM25 Search:**
- **rank-bm25** library (pure Python, no external dependencies)

**Embedding API:**
- **OpenAI text-embedding-3-small** (1536 dimensions, $0.02 per 1M tokens)
- Expected one-time cost: ~$1 for full wiki (~50K chunks)

**LLM Providers:**
- **MVP:** OpenAI API (GPT-3.5-turbo for cost efficiency, GPT-4 for quality)
- **Phase 2:** Add Gemini, Claude, Grok, Mistral via multi-provider strategy pattern
- Provider abstraction layer for easy switching

**XML Parsing:**
- **lxml** (fast C-based parser, XPath support)
- **defusedxml** (security protection against XML vulnerabilities)
- **mwparserfromhell** (MediaWiki markup → Markdown conversion)
- Streaming parser (iterparse) for memory efficiency (<500MB footprint)

**Database (Structured Data):**
- **SQLite** (embedded, zero-config, adequate for <5 servers)
- Stores: query logs, trivia attempts, server configs, leaderboards

**ORM & Migrations:**
- **SQLAlchemy 2.0+** (type-safe ORM, async support, repository pattern)
- **Alembic** (schema versioning and migrations)

**Discord Integration:**
- **Discord.py** (official Python library for Discord bots)

**Development Tools:**
- **Poetry** (dependency management, pinned versions)
- **ruff** (fast Python linter and formatter, replaces flake8+black+isort)
- **mypy** (static type checking)
- **structlog** (structured JSON logging with context injection)

**Deployment:**
- **Single Debian Linux VM** (traditional server-based deployment)
- **systemd** service for bot process management
- **Poetry** for environment management on server
- No containerization in MVP (can add later if needed)

**Cross-Cutting Concerns:**
- **Logging:** structlog with JSON formatting from start (integrated into all stories)
- **Error Handling:** Comprehensive error handling with retry logic for external APIs
- **Configuration:** All configuration via .env file (python-dotenv), including:
  - API keys (never in code/version control)
  - Bot settings (name, personality mode)
  - Rate limits (per-user, per-server)
  - Admin role names (for Discord role-based permissions)
  - LLM provider selection
  - No runtime configuration via Discord commands
- **Language Handling:** Automatic language detection and matching via LLM system prompt (Hungarian default, English when user queries in English)

---
