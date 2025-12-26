# Coding Standards

**⚠️ CRITICAL: These standards are MANDATORY for all AI-generated code.**

## Core Standards

**Languages & Runtimes:**
- Python 3.11.x
- Bash for scripts only
- SQL via Alembic migrations

**Third-Party API SDKs:**
- ALWAYS use latest stable SDK versions for all third-party APIs
- Required SDKs (as of Dec 2025):
  - OpenAI SDK: ^2.14 (`poetry add openai`)
  - Anthropic SDK: ^0.75 (`poetry add anthropic`)
  - Google Gen AI SDK: ^1.56 (`poetry add google-genai`) - NOTE: `google-generativeai` deprecated
  - Discord.py: ^2.6 (`poetry add discord-py`)
- Update SDKs quarterly or when security patches released
- Pin exact versions in poetry.lock for reproducibility
- Test thoroughly after SDK upgrades before deployment

**Style & Linting:**
- Linter: ruff (replaces flake8, black, isort)
- Line length: 100 characters
- Pre-commit hooks: REQUIRED - activate with `poetry run pre-commit install`

**Type Checking:**
- Type hints REQUIRED for all function signatures
- Tool: mypy in strict mode

**Pre-Commit Configuration:**
To activate: `poetry run pre-commit install` (one-time setup per developer)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        files: ^src/

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']
```

## SOLID Principles - MANDATORY

**S - Single Responsibility:** One reason to change per class
**O - Open/Closed:** Extend via inheritance, not modification
**L - Liskov Substitution:** Subclasses substitutable for base
**I - Interface Segregation:** No unused methods in interfaces
**D - Dependency Inversion:** Depend on abstractions

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Module/File | snake_case | `query_orchestrator.py` |
| Class | PascalCase | `QueryOrchestrator` |
| Function | snake_case | `def process_query()` |
| Variable | snake_case | `user_id` |
| Constant | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| Private | _leading_underscore | `def _helper()` |

## Critical Rules

**1. Never use print() - use logger**
```python
# ❌ BAD
print(f"Processing: {query}")

# ✅ GOOD
logger.info("processing_query", query_text=query)
```

**2. All database ops via repositories**
```python
# ❌ BAD
result = await session.execute(select(QueryLog))

# ✅ GOOD
result = await query_log_repo.get_by_id(id)
```

**3. All LLM calls via router**
```python
# ❌ BAD
client = OpenAI()
response = client.chat.completions.create(...)

# ✅ GOOD
response = await llm_router.generate(prompt, provider="openai")
```

**4. Never hardcode secrets**
```python
# ❌ BAD
API_KEY = "sk-abc123"

# ✅ GOOD
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ConfigurationError("OPENAI_API_KEY not set")
```

**5. Use custom exception hierarchy**
```python
# ❌ BAD
raise Exception("Failed")

# ✅ GOOD
raise LLMProviderError("Generation failed", is_retryable=True)
```

**6. Await all async functions**
```python
# ❌ BAD
async def process():
    result = repo.get()  # Blocking!

# ✅ GOOD
async def process():
    result = await repo.get()
```

**7. No fallback logic - fail fast**
```python
# ❌ BAD
try:
    await provider_a.generate()
except:
    await provider_b.generate()  # No fallbacks!

# ✅ GOOD
await llm_router.generate()  # Fails if provider fails
```

**8. User-facing strings with emoji**
```python
# ❌ BAD
return "Error: rate limit"

# ✅ GOOD
return "⏱️ Rate limit exceeded. Try in 45 min."
```

**9. Use pathlib for files**
```python
# ❌ BAD
path = os.path.join("data", "file.json")

# ✅ GOOD
path = Path("data") / "file.json"
```

**10. SQL via Alembic only**
```python
# ❌ BAD
conn.execute("ALTER TABLE ...")

# ✅ GOOD
# alembic revision --autogenerate -m "add field"
# alembic upgrade head
```

**11. Never expose stack traces to users**
```python
# ❌ BAD
except Exception as e:
    await send_message(f"Error: {traceback.format_exc()}")

# ✅ GOOD
except Exception as e:
    logger.error("error", exc_info=True)
    await send_message("❌ Unexpected error occurred.")
```

**12. Use QueryResponse model**
```python
# ❌ BAD
return "Guilliman is..."

# ✅ GOOD
return QueryResponse(
    answer="Guilliman is...",
    sources=[chunk1, chunk2],
    metadata={"latency_ms": 1200}
)
```

**13. All imports at top level**
```python
# ❌ BAD
def process():
    from src.llm.router import LLMRouter

# ✅ GOOD
from src.llm.router import LLMRouter

def process():
    ...
```

**14. Avoid code duplication**
```python
# ❌ BAD - duplicated validation
def validate_query(q):
    if len(q) < 5: raise Error()
def validate_answer(a):
    if len(a) < 5: raise Error()

# ✅ GOOD - shared utility
def validate_min_length(text, min_len, field):
    if len(text) < min_len:
        raise InvalidError(f"{field} too short")
```

## Method and File Complexity

**Limits:**
- Methods: Max 50 statements
- Files: Max 500 statements
- Classes: Max 300 statements
- Cyclomatic complexity: Max 10

**Configuration (pyproject.toml):**
```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pylint]
max-statements = 50
```

## Documentation

**Docstrings - Required for:**
- All public classes
- All public functions
- Complex internal functions

**Format - Google Style:**
```python
def retrieve_chunks(query: str, top_k: int = 20) -> List[WikiChunk]:
    """Retrieve chunks using hybrid search.

    Args:
        query: User question
        top_k: Number of chunks

    Returns:
        List of WikiChunk objects

    Raises:
        RetrievalError: If search fails
    """
```

## Import Organization

```python
# Standard library
import os
from datetime import datetime
from typing import List

# Third-party
import discord
from sqlalchemy.ext.asyncio import AsyncSession

# Local
from src.models.wiki_chunk import WikiChunk
from src.repositories.vector_repository import VectorRepository
```

## Git Commit Conventions

```
<type>: <description>

<optional body>
```

**Types:** feat, fix, refactor, test, docs, chore

**Examples:**
```
feat: add hybrid retrieval with BM25

fix: prevent trivia answer spoiling

refactor: extract validation to shared utility
```
