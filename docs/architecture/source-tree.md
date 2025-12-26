# Source Tree

```
wh40k-lore-bot/
├── src/                                    # Application source
│   ├── adapters/                          # Entry points
│   │   ├── discord_adapter.py
│   │   ├── cli_adapter.py
│   │   └── api_adapter.py                 # Phase 3
│   │
│   ├── orchestration/                     # Central orchestrator
│   │   ├── query_orchestrator.py          # ⭐ Main
│   │   ├── query_validator.py
│   │   ├── rate_limiter.py
│   │   └── models.py
│   │
│   ├── rag/                               # RAG pipeline
│   │   ├── rag_engine.py
│   │   ├── hybrid_retrieval.py
│   │   ├── metadata_filter.py
│   │   ├── context_expander.py
│   │   └── fusion.py
│   │
│   ├── llm/                               # LLM layer
│   │   ├── llm_router.py
│   │   ├── base_provider.py
│   │   ├── providers/
│   │   │   ├── openai_provider.py
│   │   │   ├── gemini_provider.py         # Phase 2
│   │   │   ├── claude_provider.py         # Phase 2
│   │   │   ├── grok_provider.py           # Phase 2
│   │   │   └── mistral_provider.py        # Phase 2
│   │   └── response_formatter.py
│   │
│   ├── repositories/                      # Data access
│   │   ├── vector_repository.py
│   │   ├── bm25_repository.py
│   │   ├── query_log_repository.py
│   │   ├── trivia_repository.py
│   │   └── server_config_repository.py
│   │
│   ├── services/                          # Supporting
│   │   ├── query_logger.py
│   │   └── trivia_system.py
│   │
│   ├── ingestion/                         # Data pipeline
│   │   ├── wiki_xml_parser.py
│   │   ├── chunking_service.py
│   │   ├── metadata_extractor.py
│   │   ├── embedding_service.py
│   │   └── index_builder.py
│   │
│   ├── models/                            # Domain models
│   │   ├── wiki_chunk.py
│   │   ├── query_log.py
│   │   ├── trivia.py
│   │   ├── server_config.py
│   │   └── user_feedback.py
│   │
│   ├── utils/                             # Shared utilities
│   │   ├── logger.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── metrics.py
│   │
│   ├── migrations/                        # DB migrations
│   │   └── versions/
│   │
│   ├── api/                               # Admin dashboard (Phase 3)
│   │   ├── main.py
│   │   ├── routes/
│   │   └── templates/
│   │
│   └── __main__.py                        # Entry point

├── tests/                                 # Test suite
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/

├── scripts/                               # Utility scripts
│   ├── parse_wiki_xml.sh
│   ├── rebuild_indexes.sh
│   ├── backup_databases.sh
│   └── migrate_database.sh

├── data/                                  # Data storage (gitignored)
│   ├── chroma-db/
│   ├── bm25-index/
│   ├── wh40k_lore_bot.db
│   ├── markdown-archive/                  # ⭐ Private sub-repo
│   ├── backups/
│   └── logs/

├── config/                                # Configuration
│   ├── .env.example
│   ├── wiki_urls.txt
│   └── personalities/

├── deployment/                            # Infrastructure
│   ├── systemd/
│   │   └── wh40k-lore-bot.service
│   ├── nginx/
│   └── monitoring/

├── docs/                                  # Documentation
│   ├── architecture.md                    # ⭐ This document
│   ├── brainstorming-session-results.md
│   └── rag-architecture-research-report.md

├── .bmad-core/                            # BMAD framework

├── pyproject.toml
├── poetry.lock
├── alembic.ini
├── .env
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── LICENSE
```

## Import Conventions

**Absolute imports only:**
```python
from src.orchestration.query_orchestrator import QueryOrchestrator
from src.repositories.vector_repository import VectorRepository
```

## Entry Points

**Discord Bot:**
```bash
poetry run python -m src
```

**CLI Commands:**
```bash
poetry run python -m src.adapters.cli_adapter query "Who is Guilliman?"
poetry run python -m src.adapters.cli_adapter parse-wiki-xml --source wiki-export.xml
poetry run python -m src.adapters.cli_adapter rebuild-indexes
```
