"""End-to-end tests for CLI command workflows.

Run with: poetry run pytest -m e2e

These tests validate the complete data pipeline from XML parsing through
to database storage using a minimal test dataset.

Note: Tests requiring OpenAI API (embed, ingest) will automatically use
OPENAI_API_KEY from .env file. Tests skip gracefully if key is not set.
"""

import json
import os
from pathlib import Path

import chromadb
import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

from src.cli.build_bm25 import build_bm25
from src.cli.build_test_bed import build_test_bed
from src.cli.chunk import chunk
from src.cli.embed import embed
from src.cli.ingest import ingest
from src.cli.parse_wiki import parse_wiki
from src.cli.purge_db import purge_db
from src.cli.show_chunk import show_chunk
from src.cli.store import store
from src.rag.vector_store import ChromaVectorStore

# Load environment variables from .env file before running tests
load_dotenv()


@pytest.mark.e2e
class TestCLIWorkflowE2E:
    """End-to-end tests for CLI command pipeline.

    Tests run sequentially and share artifacts via class attributes.
    Each test depends on the success of previous tests.
    """

    # Shared artifacts between tests
    artifacts: dict[str, Path] = {}

    def test_01_build_test_bed(self, e2e_test_xml_path: Path, e2e_output_dir: Path) -> None:
        """Test build-test-bed command creates page ID file.

        Validates:
        - Command exits successfully
        - Output file is created
        - File contains expected seed ID
        - File has at least one page ID
        """
        runner = CliRunner()
        test_bed_file = e2e_output_dir / "test-bed-pages.txt"

        result = runner.invoke(
            build_test_bed,
            [
                str(e2e_test_xml_path),
                "--seed-id",
                "58",
                "--count",
                "3",
                "--output",
                str(test_bed_file),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert test_bed_file.exists(), "Output file not created"

        # Content validation
        content = test_bed_file.read_text()
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.startswith("#")
        ]
        assert len(lines) >= 1, "No page IDs found in output"
        assert "58" in lines, "Seed ID not present in output"

        # Store for next tests
        self.artifacts["test_bed_file"] = test_bed_file

    def test_02_parse_wiki(self, e2e_test_xml_path: Path, e2e_output_dir: Path) -> None:
        """Test parse-wiki command creates markdown files.

        Validates:
        - Command exits successfully
        - Markdown files are created
        - At least one markdown file exists
        """
        if "test_bed_file" not in self.artifacts:
            pytest.skip("test_bed_file not available from previous test")

        runner = CliRunner()
        markdown_dir = e2e_output_dir / "markdown-archive"
        markdown_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            parse_wiki,
            [
                str(e2e_test_xml_path),
                "--page-ids-file",
                str(self.artifacts["test_bed_file"]),
                "--archive-path",
                str(markdown_dir),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Check that markdown files were created
        md_files = list(markdown_dir.glob("*.md"))
        assert len(md_files) > 0, "No markdown files created"

        self.artifacts["markdown_dir"] = markdown_dir

    def test_03_chunk(self, e2e_output_dir: Path) -> None:
        """Test chunk command creates chunks JSON.

        Validates:
        - Command exits successfully
        - Output file is created
        - JSON structure is valid
        - Contains chunks with required fields
        """
        if "markdown_dir" not in self.artifacts:
            pytest.skip("markdown_dir not available from previous test")

        runner = CliRunner()
        chunks_file = e2e_output_dir / "chunks.json"

        result = runner.invoke(
            chunk,
            [
                "--archive-path",
                str(self.artifacts["markdown_dir"]),
                "--output",
                str(chunks_file),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert chunks_file.exists(), "Output file not created"

        # Structure validation
        with chunks_file.open() as f:
            data = json.load(f)

        assert "total_chunks" in data, "Missing total_chunks field"
        assert data["total_chunks"] > 0, "No chunks created"
        assert "chunks" in data, "Missing chunks field"
        assert len(data["chunks"]) > 0, "Empty chunks list"

        # Content validation
        chunk_entry = data["chunks"][0]
        assert "wiki_page_id" in chunk_entry, "Missing wiki_page_id"
        assert "chunk_text" in chunk_entry, "Missing chunk_text"
        assert len(chunk_entry["chunk_text"]) > 0, "Empty chunk_text"
        assert "article_title" in chunk_entry, "Missing article_title"
        assert "section_path" in chunk_entry, "Missing section_path"
        assert "metadata" in chunk_entry, "Missing metadata"

        self.artifacts["chunks_file"] = chunks_file

    def test_04_embed(self, e2e_output_dir: Path) -> None:
        """Test embed command generates embeddings JSON.

        Validates:
        - Command exits successfully
        - Output file is created
        - JSON structure is valid
        - Embeddings have correct dimensions

        Note: Requires OPENAI_API_KEY environment variable.
        """
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping embed test")

        if "chunks_file" not in self.artifacts:
            pytest.skip("chunks_file not available from previous test")

        runner = CliRunner()
        embeddings_file = e2e_output_dir / "embeddings.json"

        result = runner.invoke(
            embed,
            [
                str(self.artifacts["chunks_file"]),
                "--output",
                str(embeddings_file),
                "--batch-size",
                "10",
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert embeddings_file.exists(), "Output file not created"

        # Structure validation
        with embeddings_file.open() as f:
            data = json.load(f)

        assert "embeddings" in data, "Missing embeddings field"
        assert len(data["embeddings"]) > 0, "No embeddings generated"

        # Content validation
        embedding_entry = data["embeddings"][0]
        assert "embedding" in embedding_entry, "Missing embedding field"
        assert "chunk_text" in embedding_entry, "Missing chunk_text"
        assert "wiki_page_id" in embedding_entry, "Missing wiki_page_id"

        self.artifacts["embeddings_file"] = embeddings_file

    def test_05_store(self, e2e_output_dir: Path) -> None:
        """Test store command populates Chroma DB and BM25 index.

        Validates:
        - Command exits successfully
        - Chroma DB directory is created
        - Database contains files
        - BM25 index file is created
        - ChromaDB contains chunks
        """
        if "embeddings_file" not in self.artifacts:
            pytest.skip("embeddings_file not available from previous test")

        runner = CliRunner()
        chroma_dir = e2e_output_dir / "chroma-db"

        result = runner.invoke(
            store,
            [
                str(self.artifacts["embeddings_file"]),
                "--chroma-path",
                str(chroma_dir),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert chroma_dir.exists(), "Chroma DB directory not created"

        # Chroma creates multiple files in the directory
        db_files = list(chroma_dir.glob("**/*"))
        assert len(db_files) > 0, "No files created in Chroma DB directory"

        # BM25 index should also be created by store command
        bm25_index = chroma_dir / "bm25_index.pkl"
        assert bm25_index.exists(), "BM25 index not created"
        assert bm25_index.stat().st_size > 0, "BM25 index file is empty"

        # Query ChromaDB to verify chunks were stored
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        chunk_count = collection.count()
        assert chunk_count > 0, f"No chunks found in ChromaDB, expected > 0, got {chunk_count}"

        self.artifacts["chroma_dir"] = chroma_dir

    def test_06_build_bm25(self, e2e_output_dir: Path) -> None:
        """Test build-bm25 command creates BM25 index independently.

        Validates:
        - Command exits successfully
        - Index file is created
        - File is not empty

        Note: This test is independent of the store command and can run
        after the chunk step.
        """
        if "chunks_file" not in self.artifacts:
            pytest.skip("chunks_file not available from previous test")

        runner = CliRunner()
        bm25_dir = e2e_output_dir / "bm25-index"
        bm25_dir.mkdir(exist_ok=True)
        bm25_file = bm25_dir / "bm25_index.pkl"

        result = runner.invoke(
            build_bm25,
            [
                str(self.artifacts["chunks_file"]),
                "--output",
                str(bm25_file),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert bm25_file.exists(), "BM25 index file not created"
        assert bm25_file.stat().st_size > 0, "BM25 index file is empty"

    def test_07_purge_db(self) -> None:
        """Test purge-db command clears ChromaDB.

        Validates:
        - Command exits successfully
        - ChromaDB is empty after purge

        This ensures the subsequent ingest test starts with a clean database.
        """
        if "chroma_dir" not in self.artifacts:
            pytest.skip("chroma_dir not available from previous test")

        runner = CliRunner()
        chroma_dir = self.artifacts["chroma_dir"]

        # Verify DB has chunks before purge
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        count_before = collection.count()
        assert count_before > 0, "DB should have chunks before purge"

        # Run purge command
        result = runner.invoke(
            purge_db,
            [
                "--chroma-path",
                str(chroma_dir),
                "--force",
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify DB is empty after purge
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        count_after = collection.count()
        assert count_after == 0, f"DB should be empty after purge, got {count_after} chunks"

    def test_08_ingest_full_pipeline(self, e2e_test_xml_path: Path, e2e_output_dir: Path) -> None:
        """Test ingest command (full pipeline with embeddings).

        Validates:
        - Command exits successfully
        - Chroma DB is created
        - ChromaDB contains chunks
        - Summary file contains expected data

        Note: Requires OPENAI_API_KEY environment variable.
        This test validates the complete all-in-one workflow starting with an empty DB.
        """
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping ingest test")

        if "test_bed_file" not in self.artifacts:
            pytest.skip("test_bed_file not available from previous test")

        runner = CliRunner()
        ingest_output_dir = e2e_output_dir / "ingest-test"
        ingest_output_dir.mkdir(exist_ok=True)

        # Use separate directories for ingest test
        ingest_markdown_dir = ingest_output_dir / "markdown-archive"
        ingest_chroma_dir = ingest_output_dir / "chroma-db"

        # First parse wiki to the ingest markdown directory
        result = runner.invoke(
            parse_wiki,
            [
                str(e2e_test_xml_path),
                "--page-ids-file",
                str(self.artifacts["test_bed_file"]),
                "--archive-path",
                str(ingest_markdown_dir),
            ],
        )

        assert result.exit_code == 0, f"Parse wiki failed: {result.output}"

        # Run full ingest pipeline
        result = runner.invoke(
            ingest,
            [
                "--archive-path",
                str(ingest_markdown_dir),
                "--chroma-path",
                str(ingest_chroma_dir),
                "--wiki-ids-file",
                str(self.artifacts["test_bed_file"]),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify Chroma DB was created
        assert ingest_chroma_dir.exists(), "Chroma DB directory not created"
        db_files = list(ingest_chroma_dir.glob("**/*"))
        assert len(db_files) > 0, "No files in Chroma DB directory"

        # Query ChromaDB to verify chunks were stored
        client = chromadb.PersistentClient(path=str(ingest_chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        chunk_count = collection.count()
        assert chunk_count > 0, f"No chunks found in ChromaDB, expected > 0, got {chunk_count}"

        # Check summary file if it exists
        summary_file = Path("logs/ingestion-summary.json")
        if summary_file.exists():
            with summary_file.open() as f:
                data = json.load(f)

            assert "articles_processed" in data, "Missing articles_processed"
            assert data["articles_processed"] > 0, "No articles processed"
            assert "chunks_created" in data, "Missing chunks_created"
            assert data["chunks_created"] > 0, "No chunks created"
            assert "embeddings_generated" in data, "Missing embeddings_generated"

        # Store for show-chunk test
        self.artifacts["ingest_chroma_dir"] = ingest_chroma_dir

    def test_09_show_chunk(self) -> None:
        """Test show-chunk command displays chunk details.

        Validates:
        - Command exits successfully
        - Output contains chunk information

        Note: Uses the ChromaDB populated by the ingest test.
        """
        if "ingest_chroma_dir" not in self.artifacts:
            pytest.skip("ingest_chroma_dir not available from previous test")

        runner = CliRunner()
        chroma_dir = self.artifacts["ingest_chroma_dir"]

        # Get the first chunk ID from ChromaDB
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)

        # Get first chunk
        results = collection.get(limit=1, include=["metadatas"])
        assert len(results["ids"]) > 0, "No chunks available for show-chunk test"

        chunk_id = results["ids"][0]

        # Run show-chunk command
        result = runner.invoke(
            show_chunk,
            [
                chunk_id,
                "--chroma-path",
                str(chroma_dir),
            ],
        )

        # Basic validation
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert chunk_id in result.output, "Chunk ID not found in output"
        assert len(result.output) > 0, "Output is empty"
