# E2E CLI Tests Implementation Plan

## Overview
Implement end-to-end tests for CLI workflows to validate the complete data pipeline from XML parsing through to database storage. Tests will use minimal real XML data (10 pages) and run only when explicitly requested via `poetry run pytest -m e2e`.

## User Requirements
- Extract minimal XML from actual `warhammer40k_pages_current.xml`
- Store test data in existing `./test-data/` folder
- Basic validation + check for some IDs to exist
- Tests depend on each other to save setup time
- Skip dependent tests if previous ones fail
- Add pytest marker "e2e" (not run by default)
- Update PRD documentation
- API costs are acceptable (no dry-run mode needed)

## Test Workflows to Implement

Seven sequential workflows, each depending on previous:

1. **build-test-bed**: `build-test-bed <xml> --seed-id 58 --count 3`
2. **parse-wiki**: `parse-wiki <xml> --page-ids-file <path>`
3. **chunk**: `chunk --output <chunks_file>`
4. **embed**: `embed <chunks_file> --output <embeddings_file>` (requires API key)
5. **store**: `store <embeddings_file>`
6. **build-bm25**: `build-bm25 <chunks_file>` (independent after chunk)
7. **ingest**: `ingest` (all-in-one pipeline validation with full embedding generation)

## Implementation Steps

### 1. Create Pytest Configuration (pyproject.toml)

**File**: [pyproject.toml](../pyproject.toml)

Add markers and exclude e2e from default runs:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term-missing -m 'not e2e'"
asyncio_mode = "auto"
markers = [
    "e2e: End-to-end CLI workflow tests (not run by default)",
    "integration: Integration tests requiring external APIs (not run by default)",
]
```

### 2. Create Test Data Structure

**Directory structure**:
```
./test-data/
├── e2e-test-wiki.xml          # 10-page minimal XML (~50-100KB)
├── e2e-outputs/               # E2E test outputs (gitignored)
│   ├── test-bed-pages.txt
│   ├── markdown-archive/
│   ├── chunks.json
│   ├── embeddings.json
│   ├── chroma-db/
│   └── bm25-index/
└── .gitignore                 # Ignore e2e-outputs/
```

**File**: test-data/.gitignore (NEW)
```
e2e-outputs/
```

### 3. Create XML Extraction Script

**File**: tests/e2e/fixtures/extract_minimal_xml.py (NEW)

Script to extract 10 pages from `warhammer40k_pages_current.xml`:
- Page IDs to extract: 58, 8783, 206, 86, 1839, 116, 91375, 91130, 16481, 2579
- Uses WikiXMLParser streaming approach
- Fallback to synthetic XML if source unavailable
- Outputs to `test-data/e2e-test-wiki.xml`

Key functions:
```python
def extract_minimal_xml(source_xml: Path, page_ids: list[str], output_path: Path) -> None:
    """Extract specific pages from large XML export."""
    # Stream through XML, extract matching pages, write to output

def create_synthetic_xml(output_path: Path) -> None:
    """Create synthetic 10-page XML if real source unavailable."""
    # Use pattern from test_test_bed_builder.py SAMPLE_XML
```

### 4. Create E2E Test Fixtures

**File**: tests/e2e/conftest.py (NEW)

Session-scoped fixtures:
```python
@pytest.fixture(scope="session")
def e2e_test_xml_path() -> Path:
    """Return path to minimal E2E test XML file."""
    xml_path = Path("./test-data/e2e-test-wiki.xml")
    if not xml_path.exists():
        pytest.skip("E2E test XML not found. Run extract_minimal_xml.py first.")
    return xml_path

@pytest.fixture(scope="session")
def e2e_output_dir() -> Path:
    """Create persistent output directory for E2E test artifacts."""
    output_dir = Path("./test-data/e2e-outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

@pytest.fixture(scope="session", autouse=True)
def cleanup_e2e_outputs(e2e_output_dir: Path):
    """Clean up E2E outputs before test session."""
    import shutil
    if e2e_output_dir.exists():
        shutil.rmtree(e2e_output_dir)
    e2e_output_dir.mkdir(parents=True, exist_ok=True)
```

### 5. Create Main E2E Test File

**File**: tests/e2e/test_cli_workflows.py (NEW)

Test class with 7 sequential methods:

```python
"""End-to-end tests for CLI command workflows.

Run with: poetry run pytest -m e2e
"""

from pathlib import Path
import json
import os
import pytest
from click.testing import CliRunner

# Import CLI commands
from src.cli.build_test_bed import build_test_bed
from src.cli.parse_wiki import parse_wiki
from src.cli.chunk import chunk
from src.cli.embed import embed
from src.cli.store import store
from src.cli.build_bm25 import build_bm25
from src.cli.ingest import ingest


@pytest.mark.e2e
class TestCLIWorkflowE2E:
    """End-to-end tests for CLI command pipeline."""

    # Shared artifacts between tests
    artifacts = {}

    def test_01_build_test_bed(self, e2e_test_xml_path, e2e_output_dir):
        """Test build-test-bed command creates page ID file."""
        runner = CliRunner()
        test_bed_file = e2e_output_dir / "test-bed-pages.txt"

        result = runner.invoke(build_test_bed, [
            str(e2e_test_xml_path),
            "--seed-id", "58",
            "--count", "3",
            "--output", str(test_bed_file)
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert test_bed_file.exists()

        # Content validation
        content = test_bed_file.read_text()
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        assert len(lines) >= 1
        assert "58" in lines  # Seed ID must be present

        # Store for next tests
        self.artifacts['test_bed_file'] = test_bed_file

    def test_02_parse_wiki(self, e2e_test_xml_path, e2e_output_dir):
        """Test parse-wiki command creates markdown files."""
        if 'test_bed_file' not in self.artifacts:
            pytest.skip("test_bed_file not available from previous test")

        runner = CliRunner()
        markdown_dir = e2e_output_dir / "markdown-archive"

        # Set archive path via environment or create temporary config
        result = runner.invoke(parse_wiki, [
            str(e2e_test_xml_path),
            "--page-ids-file", str(self.artifacts['test_bed_file'])
        ], env={'MARKDOWN_ARCHIVE_PATH': str(markdown_dir)})

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Note: parse_wiki uses default data/markdown-archive/, need to handle
        # Check actual output location based on CLI implementation

        self.artifacts['markdown_dir'] = markdown_dir

    def test_03_chunk(self, e2e_output_dir):
        """Test chunk command creates chunks JSON."""
        if 'markdown_dir' not in self.artifacts:
            pytest.skip("markdown_dir not available from previous test")

        runner = CliRunner()
        chunks_file = e2e_output_dir / "chunks.json"

        result = runner.invoke(chunk, [
            "--archive-path", str(self.artifacts['markdown_dir']),
            "--output", str(chunks_file)
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert chunks_file.exists()

        # Structure validation
        with chunks_file.open() as f:
            data = json.load(f)
        assert "total_chunks" in data
        assert data["total_chunks"] > 0
        assert "chunks" in data
        assert len(data["chunks"]) > 0

        # Content validation
        chunk = data["chunks"][0]
        assert "wiki_page_id" in chunk
        assert "chunk_text" in chunk
        assert len(chunk["chunk_text"]) > 0

        self.artifacts['chunks_file'] = chunks_file

    def test_04_embed(self, e2e_output_dir):
        """Test embed command generates embeddings JSON."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping embed test")

        if 'chunks_file' not in self.artifacts:
            pytest.skip("chunks_file not available from previous test")

        runner = CliRunner()
        embeddings_file = e2e_output_dir / "embeddings.json"

        result = runner.invoke(embed, [
            str(self.artifacts['chunks_file']),
            "--output", str(embeddings_file),
            "--batch-size", "10"
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert embeddings_file.exists()

        # Structure validation
        with embeddings_file.open() as f:
            data = json.load(f)
        assert "embeddings" in data
        assert len(data["embeddings"]) > 0

        # Content validation
        embedding = data["embeddings"][0]
        assert "embedding" in embedding
        assert len(embedding["embedding"]) == 1536  # OpenAI dimension

        self.artifacts['embeddings_file'] = embeddings_file

    def test_05_store(self, e2e_output_dir):
        """Test store command populates Chroma DB and BM25 index."""
        if 'embeddings_file' not in self.artifacts:
            pytest.skip("embeddings_file not available from previous test")

        runner = CliRunner()
        chroma_dir = e2e_output_dir / "chroma-db"

        result = runner.invoke(store, [
            str(self.artifacts['embeddings_file']),
            "--chroma-path", str(chroma_dir)
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert chroma_dir.exists()

        # Chroma creates multiple files
        db_files = list(chroma_dir.glob("*"))
        assert len(db_files) > 0

        # BM25 index should also be created
        bm25_index = chroma_dir / "bm25_index.pkl"
        assert bm25_index.exists()

        self.artifacts['chroma_dir'] = chroma_dir

    def test_06_build_bm25(self, e2e_output_dir):
        """Test build-bm25 command creates BM25 index independently."""
        if 'chunks_file' not in self.artifacts:
            pytest.skip("chunks_file not available from previous test")

        runner = CliRunner()
        bm25_dir = e2e_output_dir / "bm25-index"
        bm25_dir.mkdir(exist_ok=True)
        bm25_file = bm25_dir / "bm25_index.pkl"

        result = runner.invoke(build_bm25, [
            str(self.artifacts['chunks_file']),
            "--output", str(bm25_file)
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert bm25_file.exists()
        assert bm25_file.stat().st_size > 0

    def test_07_ingest_full_pipeline(self, e2e_test_xml_path, e2e_output_dir):
        """Test ingest command (full pipeline with embeddings)."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping ingest test")

        if 'test_bed_file' not in self.artifacts:
            pytest.skip("test_bed_file not available from previous test")

        runner = CliRunner()
        ingest_output_dir = e2e_output_dir / "ingest-test"
        ingest_output_dir.mkdir(exist_ok=True)

        # Use a separate markdown archive for ingest test
        ingest_markdown_dir = ingest_output_dir / "markdown-archive"
        ingest_chroma_dir = ingest_output_dir / "chroma-db"

        # First parse wiki to the ingest markdown directory
        result = runner.invoke(parse_wiki, [
            str(e2e_test_xml_path),
            "--page-ids-file", str(self.artifacts['test_bed_file'])
        ], env={'MARKDOWN_ARCHIVE_PATH': str(ingest_markdown_dir)})

        assert result.exit_code == 0, f"Parse wiki failed: {result.output}"

        # Run full ingest pipeline
        result = runner.invoke(ingest, [
            "--archive-path", str(ingest_markdown_dir),
            "--chroma-path", str(ingest_chroma_dir),
            "--wiki-ids-file", str(self.artifacts['test_bed_file'])
        ])

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Check summary file
        summary_file = Path("logs/ingestion-summary.json")
        if summary_file.exists():
            with summary_file.open() as f:
                data = json.load(f)
            assert "articles_processed" in data
            assert data["articles_processed"] > 0
            assert "chunks_created" in data
            assert data["chunks_created"] > 0

        # Verify Chroma DB was created
        assert ingest_chroma_dir.exists()
        db_files = list(ingest_chroma_dir.glob("*"))
        assert len(db_files) > 0
```

Key patterns:
- Use `CliRunner` to invoke Click commands programmatically
- Store artifact paths in class attribute `artifacts` for sharing between tests
- Each test checks for prerequisites and skips if missing
- Basic validation: exit code, file existence
- Structure validation: JSON keys, counts > 0
- Content validation: Check for expected IDs/fields

### 6. Update PRD Documentation

**File**: [docs/prd/technical-assumptions.md](prd/technical-assumptions.md)

Add after line 30 (in Testing Requirements section):

```markdown
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

**Rationale:** E2E tests catch integration issues between CLI commands and validate the full data pipeline with realistic (but minimal) data. Separating them from unit tests keeps CI fast while providing comprehensive validation before releases.
```

**File**: [docs/prd.md](prd.md)

Update line 90 (FR29):

```markdown
**FR29:** System SHALL provide CLI commands for data ingestion, index rebuilding, and database management **with end-to-end test coverage validating complete workflows**
```

### 7. Add __init__.py Files

Create empty package markers:
- `tests/e2e/__init__.py`
- `tests/e2e/fixtures/__init__.py`

## Validation Strategy

Each test validates at three levels:

1. **Basic**: Exit code 0, file exists, non-zero size
2. **Structural**: JSON is valid, has required keys, types correct
3. **Content**: Check for expected IDs/counts from known test data

### Expected Test Data
From e2e-test-wiki.xml (10 pages):
- Page IDs: 58, 8783, 206, 86, 1839, 116, 91375, 91130, 16481, 2579
- Expected chunks: ~20-50 (depends on content size)
- Expected embeddings: Same as chunks
- Expected markdown files: 10

## Running E2E Tests

```bash
# Initial setup (one-time)
poetry run python tests/e2e/fixtures/extract_minimal_xml.py

# Run E2E tests (automatically reads OPENAI_API_KEY from .env file)
# Tests requiring API (embed, ingest) skip if key not set
poetry run pytest -m e2e

# Run specific test
poetry run pytest tests/e2e/test_cli_workflows.py::TestCLIWorkflowE2E::test_03_chunk -v
```

**Note:** The E2E tests automatically load environment variables from `.env` file. Make sure your `.env` file contains `OPENAI_API_KEY=sk-...` to run tests that require embeddings (test_04_embed and test_07_ingest_full_pipeline). Without the API key, these tests will be skipped gracefully.

## Files to Create/Modify

### New Files
1. `tests/e2e/__init__.py` - Package marker
2. `tests/e2e/conftest.py` - E2E fixtures (~50 lines)
3. `tests/e2e/fixtures/__init__.py` - Package marker
4. `tests/e2e/fixtures/extract_minimal_xml.py` - XML extraction script (~150 lines)
5. `tests/e2e/test_cli_workflows.py` - Main E2E tests (~400-500 lines)
6. `test-data/.gitignore` - Ignore e2e-outputs/
7. `test-data/e2e-test-wiki.xml` - Generated by extraction script (~50-100KB)

### Modified Files
1. `pyproject.toml` - Add e2e marker and configure exclusion (~5 lines)
2. `docs/prd/technical-assumptions.md` - Add E2E testing section (~12 lines)
3. `docs/prd.md` - Enhance FR29 (~1 line)

## Success Criteria

- [ ] `poetry run pytest -m e2e` passes all 7 tests (with OPENAI_API_KEY set)
- [ ] `poetry run pytest` excludes e2e tests (runs unit tests only)
- [ ] E2E tests create all expected artifacts
- [ ] Documentation updated (PRD)
- [ ] Test data committed to repo (e2e-test-wiki.xml)
- [ ] E2E test suite completes in < 3 minutes with API key
- [ ] Tests gracefully skip embed/ingest when API key unavailable

## Risk Mitigation

1. **API calls may fail**: Skip embed and ingest tests if `OPENAI_API_KEY` not set
2. **Test data staleness**: Commit e2e-test-wiki.xml to repo for version control
3. **Long test duration**: Exclude from default runs, only run on nightly CI
4. **Storage requirements**: Gitignore e2e-outputs/, ~50-100MB max
5. **API costs**: Using minimal 10-page dataset keeps costs under $0.10 per run

## Notes

- Tests use CliRunner to invoke Click commands programmatically
- Artifacts shared via class attribute (not pytest fixtures) for simplicity
- Each test checks prerequisites and skips gracefully if missing
- Full pipeline testing with real embeddings (no dry-run mode)
- BM25 test is independent of embed/store (can run after chunk)
- Ingest test validates the complete all-in-one workflow
