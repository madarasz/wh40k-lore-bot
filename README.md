# WH40K Lore Bot

A Discord bot that provides Warhammer 40,000 lore answers using RAG (Retrieval-Augmented Generation) technology. The bot parses the WH40K Fandom Wiki, creates searchable embeddings, and uses AI to provide accurate, contextual answers to lore questions.

## Features

- **RAG-powered Lore Queries**: Retrieve accurate WH40K lore using hybrid search (semantic + keyword)
- **Discord Integration**: Slash commands for easy querying
- **Trivia System**: Test your WH40K knowledge
- **Metadata Filtering**: Filter by faction, era, and spoiler preferences
- **Multi-LLM Support**: OpenAI, Google Gemini, Anthropic Claude, xAI Grok, Mistral

## Prerequisites

- **Python 3.11 or higher**
- **Poetry 1.7+** for dependency management

### Installing Poetry

If you don't have Poetry installed, install it using one of these methods:

**macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Verify installation:
```bash
poetry --version
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wh40k-lore-bot
```

### 2. Install Dependencies

```bash
poetry install
```

This will create a virtual environment and install all required dependencies.

### 3. Configure Environment Variables

Copy the environment template:

```bash
cp .env.template .env
```

Edit `.env` and fill in your credentials:

```bash
# Discord Bot Token
# Get from: https://discord.com/developers/applications
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# OpenAI API Key
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Database URL (SQLite)
DATABASE_URL=sqlite:///data/wh40k_lore_bot.db

# Logging Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
```

### 4. Initialize Database

```bash
poetry run alembic upgrade head
```

### 5. Install Pre-commit Hooks (Optional but Recommended)

```bash
poetry run pre-commit install
```

This will automatically run linting and type checking before each commit.

## Development Workflow

### Running the Bot

```bash
poetry run python -m src
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_logger.py

# Run with verbose output
poetry run pytest -v
```

```bash
# Run integration tests with real APIs (costs money)
poetry run pytest -m integration
```

```bash
# Create test data for e2e flows
poetry run python tests/e2e/fixtures/extract_minimal_xml.py

# Run e2e tests with real APIs (costs money)
poetry run pytest -m e2e
```

### Code Quality

```bash
# Run linter (with auto-fix)
poetry run ruff check . --fix

# Run formatter
poetry run ruff format .

# Run type checker
poetry run mypy src/

# Run all pre-commit hooks manually
poetry run pre-commit run --all-files
```

### Database Migrations

```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Rollback one migration
poetry run alembic downgrade -1

# View migration history
poetry run alembic history
```

## Project Structure

```
wh40k-lore-bot/
├── src/                       # Application source code
│   ├── adapters/             # Entry points (Discord, CLI, API)
│   ├── orchestration/        # Central query orchestrator
│   ├── rag/                  # RAG pipeline components
│   ├── llm/                  # LLM provider integrations
│   ├── repositories/         # Data access layer
│   ├── services/             # Business logic services
│   ├── ingestion/            # Wiki data ingestion pipeline
│   ├── models/               # Domain models
│   ├── utils/                # Shared utilities
│   └── migrations/           # Database migrations
│
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test fixtures
│
├── data/                     # Data storage (gitignored)
│   ├── markdown-archive/     # Processed wiki articles
│   ├── chroma-db/            # Vector database
│   └── wh40k_lore_bot.db    # SQLite database
│
├── config/                   # Configuration files
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

## Architecture

This project follows a **layered architecture** pattern:

- **Adapters Layer**: Entry points (Discord bot, CLI, API)
- **Orchestration Layer**: Query coordination and routing
- **RAG Layer**: Retrieval-augmented generation pipeline
- **LLM Layer**: Multi-provider LLM integration
- **Repository Layer**: Data access abstraction
- **Models Layer**: Domain entities and data structures

See [docs/architecture/](docs/architecture/) for detailed architecture documentation.

## Tech Stack

- **Language**: Python 3.11
- **Bot Framework**: discord.py 2.3+
- **Vector DB**: ChromaDB (embedded mode)
- **ORM**: SQLAlchemy 2.0
- **Migrations**: Alembic
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting**: ruff
- **Type Checking**: mypy
- **Logging**: structlog

## Contributing

### Code Standards

- **Line length**: 100 characters
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style, required for all public APIs
- **Complexity**: Max cyclomatic complexity of 10
- **Testing**: All new features must include tests

### Git Workflow

1. Create a feature branch
2. Make your changes
3. Ensure all tests pass: `poetry run pytest`
4. Ensure linting passes: `poetry run ruff check .`
5. Ensure type checking passes: `poetry run mypy src/`
6. Commit with conventional commit messages
7. Submit a pull request

### Commit Message Format

```
<type>: <description>

<optional body>
```

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

**Examples**:
```
feat: add hybrid retrieval with BM25
fix: prevent trivia answer spoiling
refactor: extract validation to shared utility
```

## License

[Add your license here]

## Acknowledgments

- Warhammer 40,000 lore sourced from [Warhammer 40k Fandom Wiki](https://warhammer40k.fandom.com/)
- Built with [discord.py](https://github.com/Rapptz/discord.py)
- Powered by OpenAI embeddings and LLM APIs
