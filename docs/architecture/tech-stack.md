# Tech Stack

**⚠️ CRITICAL: This section is the DEFINITIVE technology selection. All other documents, code, and agents MUST reference these exact choices and versions.**

## Cloud Infrastructure

- **Provider:** Self-Hosted Debian VM
- **Key Services:**
  - Compute: Single Debian 12 (Bookworm) virtual machine
  - Storage: Local filesystem for vector DB, SQLite, markdown archives
  - Networking: Public IP for Discord webhook/gateway, optional SSH bastion
- **Deployment Regions:** Single region (your VM location)

## Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|--------------|
| **Language** | Python | 3.11.x | Primary development language | Stable LTS, excellent library support, your prior experience, wide RAG ecosystem |
| **Package Manager** | Poetry | 1.7+ | Dependency management and packaging | Modern lock files, reproducible builds, monorepo support, better than pip alone |
| **Discord Library** | discord.py | 2.3+ | Discord bot client and API wrapper | Most mature, extensive features, slash commands, your prior experience, active community |
| **Vector Database** | Chroma | 0.4.22+ | Embedded vector storage and similarity search | Simple embedded mode, zero-config for MVP, free, upgrade path to Qdrant if needed |
| **Vector Database (Future)** | Qdrant | 1.7+ (if needed) | Production vector DB with metadata filtering | Excellent metadata filtering, native hybrid search, self-hosted option, Python client |
| **BM25 Search** | rank-bm25 | 0.2.2 | Sparse keyword retrieval | Pure Python, simple, proven for hybrid search, no external dependencies |
| **Embedding API** | OpenAI Embeddings | text-embedding-3-small | Text-to-vector conversion | Best cost/quality ($0.00002/1K tokens), 1536-dim, well-supported, low latency |
| **LLM Provider (MVP)** | OpenAI API | Latest SDK | Primary LLM provider | GPT-4, GPT-3.5-turbo, proven quality, extensive API features |
| **LLM Providers (Phase 2)** | Google Gemini | Latest SDK | Cost-optimized LLM option | Competitive pricing, good Hungarian support, fast inference |
| | Anthropic Claude | Latest SDK | High-quality LLM option | Excellent reasoning, long context, good for complex lore queries |
| | xAI Grok | Latest SDK | Alternative LLM provider | Newer option, competitive features, diversification |
| | Mistral | Latest SDK | Open-weights LLM option | Cost-effective, European provider, good performance |
| **XML Parsing** | lxml | 5.0+ | XML parsing for wiki data export | High-performance, standards-compliant, XPath support for navigation |
| | defusedxml | 0.7+ | Secure XML parsing | Protection against XML vulnerabilities (billion laughs, entity expansion) |
| **Database (Logs/Trivia)** | SQLite | 3.40+ (Python stdlib) | Structured data persistence | Embedded, zero-config, perfect for single server, adequate for 300 users |
| **ORM** | SQLAlchemy | 2.0+ | Database abstraction layer | Type-safe, async support, migrations, repository pattern implementation |
| **Schema Migrations** | Alembic | 1.13+ | Database versioning and migrations | SQLAlchemy integration, version control for schema, team collaboration |
| **Testing Framework** | pytest | 8.0+ | Unit and integration testing | Industry standard, fixtures, parametrization, plugin ecosystem |
| **Test Coverage** | pytest-cov | 4.1+ | Code coverage measurement | Pytest integration, coverage reports, CI/CD friendly |
| **Mocking** | pytest-mock | 3.12+ | Test doubles and mocking | Clean mock API, pytest integration, simplifies testing LLM/DB calls |
| **Async Testing** | pytest-asyncio | 0.23+ | Testing async code | Essential for Discord bot and async orchestrator testing |
| **Linting** | ruff | 0.1+ | Fast Python linter and formatter | Extremely fast, replaces flake8+black+isort, modern rules |
| **Type Checking** | mypy | 1.8+ | Static type analysis | Catch bugs early, enforces type hints, improves IDE support |
| **Pre-commit** | pre-commit | 3.6+ | Git hook framework for linting | Automatic style/lint checks, prevents bad commits |
| **Environment Management** | python-dotenv | 1.0+ | Environment variable loading | Secure config management, separates secrets from code, 12-factor app |
| **Logging** | structlog | 24.1+ | Structured logging | JSON logs, context propagation, better than stdlib logging, analytics-ready |
| **Process Manager** | systemd | Native Debian | Service management and auto-restart | Built-in to Debian, reliable, auto-restart on failure, log management |
| **Evaluation Framework** | RAGAS | 0.1+ | RAG quality metrics | LLM-based evaluation, faithfulness/relevance metrics, no human labels needed |
| **Metrics Export** | prometheus-client | 0.19+ | Application metrics for monitoring | Standard metrics format, query latency tracking, operational visibility |
| **System Metrics** | node-exporter | 1.7+ | Server resource monitoring | CPU, memory, disk metrics, standard Prometheus exporter |
| **Admin Dashboard API (Phase 3)** | FastAPI | 0.109+ | REST API for admin dashboard | Modern async framework, auto-generated OpenAPI docs, fast, type-safe |
| **Admin Dashboard Frontend (Phase 3)** | htmx | 1.9+ | Dynamic frontend without heavy JS | Hypermedia-driven, minimal JS, simple server-side rendering, fast iteration |
| **Template Engine (Phase 3)** | Jinja2 | 3.1+ | HTML templating for admin dashboard | Industry standard, FastAPI integration, safe escaping, filters |

## Version Pinning Strategy

- **Exact pins for production:** Poetry lock file ensures exact versions across deployments
- **Minimum version notation (X.Y+):** Allows patch updates for security fixes
- **Major version locks:** Prevent breaking changes (e.g., discord.py 2.x, SQLAlchemy 2.x)
- **Review cadence:** Quarterly dependency updates, security patches immediate

## Backup Strategy

- **Markdown Archives:** Stored in `data/markdown-archive/` subdirectory (private sub-repo, version controlled, manually committed)
- **SQLite Databases:** Daily automated backup to `/opt/backups/`, manual file-based copy as needed
- **Vector DB:** Chroma embedded storage in `data/chroma-db/` (rebuildable from markdown archives)
