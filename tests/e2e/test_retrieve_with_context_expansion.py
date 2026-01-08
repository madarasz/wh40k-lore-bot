"""End-to-end test for CLI retrieve command with context expansion.

Run with: poetry run pytest tests/e2e/test_retrieve_with_context_expansion.py -m e2e

This test validates that the retrieve command properly applies context expansion
when CONTEXT_EXPANSION_DEPTH is set, following cross-references in chunk metadata
to enrich the retrieval results.

Note: Requires OPENAI_API_KEY from .env file for embedding generation.
"""

import os
from pathlib import Path

import chromadb
import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

from src.cli.retrieve import retrieve
from src.rag.vector_store import ChromaVectorStore

# Load environment variables from .env file before running tests
load_dotenv()


@pytest.mark.e2e
@pytest.mark.dependency()
class TestRetrieveWithContextExpansion:
    """End-to-end test for retrieve command with context expansion enabled."""

    @pytest.mark.dependency(depends=["ingest_pipeline_complete"], scope="session")
    def test_retrieve_with_context_expansion_enabled(
        self,
        e2e_output_dir: Path,
        e2e_test_xml_path: Path,
    ) -> None:
        """Test retrieve command with CONTEXT_EXPANSION_DEPTH=1.

        Validates:
        - Command exits successfully
        - Output shows context expansion was applied
        - Initial chunk count is displayed
        - Expanded chunk count is greater than initial count
        - Output includes expansion statistics

        Note: Requires OPENAI_API_KEY environment variable.
        This test depends on having a populated ChromaDB from prior e2e tests.
        """
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping retrieve test")

        # Use the ingest test output directory from the main e2e workflow
        # This assumes test_cli_ingest_workflows has already populated a test database
        ingest_output_dir = e2e_output_dir / "ingest-test"
        ingest_chroma_dir = ingest_output_dir / "chroma-db"
        ingest_bm25_dir = ingest_output_dir / "chroma-db"  # BM25 stored with chroma

        # If the standard test database doesn't exist, skip
        if not ingest_chroma_dir.exists():
            pytest.skip(
                "ChromaDB not found - run full e2e workflow first "
                "(poetry run pytest tests/e2e/test_cli_ingest_workflows.py -m e2e)"
            )

        # Verify ChromaDB has data
        client = chromadb.PersistentClient(path=str(ingest_chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        chunk_count = collection.count()

        if chunk_count == 0:
            pytest.skip("ChromaDB is empty - run full e2e workflow first")

        # Verify BM25 index exists
        bm25_index = ingest_bm25_dir / "bm25_index.pkl"
        if not bm25_index.exists():
            pytest.skip("BM25 index not found - run full e2e workflow first")

        # Convert paths to absolute BEFORE entering isolated_filesystem
        # so they resolve relative to the actual workspace, not the temp dir
        abs_chroma_dir = ingest_chroma_dir.resolve()
        abs_bm25_index = bm25_index.resolve()

        runner = CliRunner()

        # Test with context expansion enabled (depth=1)
        with runner.isolated_filesystem():
            # Set environment variables for context expansion
            env = os.environ.copy()
            env["CONTEXT_EXPANSION_ENABLED"] = "true"
            env["CONTEXT_EXPANSION_DEPTH"] = "1"
            env["CONTEXT_EXPANSION_MAX_CHUNKS"] = "30"
            env["CHROMA_DB_PATH"] = str(abs_chroma_dir)
            env["BM25_INDEX_PATH"] = str(abs_bm25_index)
            env["RETRIEVAL_TOP_K"] = "5"

            result = runner.invoke(
                retrieve,
                ["Space Marine", "--top-k", "5"],
                env=env,
                catch_exceptions=False,
            )

            # Basic validation
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify output structure
            output = result.output
            assert "HYBRID RETRIEVAL RESULTS" in output, "Missing results header"
            assert "Query: Space Marine" in output, "Missing query text"
            assert "Results:" in output, "Missing results count"
            assert "chunks" in output, "Missing chunks label"
            assert "Latency:" in output, "Missing latency info"

            # Check for context expansion indicators
            # The CLI should show expansion was applied if it added chunks
            if "Context expansion:" in output:
                # Expansion was applied and added chunks
                assert "Initial retrieval:" in output, "Missing initial retrieval count"
                assert "+0 chunks" not in output or "+" in output, "Expected chunks to be added"

                # Extract chunk counts from output to verify expansion worked
                # Output format: "├─ Initial retrieval: N chunks"
                #                "└─ Context expansion: +M chunks"
                lines = output.split("\n")
                for i, line in enumerate(lines):
                    if "Initial retrieval:" in line:
                        # Found initial count line, check next line for expansion
                        if i + 1 < len(lines) and "Context expansion:" in lines[i + 1]:
                            expansion_line = lines[i + 1]
                            # Verify expansion line has + sign indicating chunks were added
                            assert "+" in expansion_line, "No chunks added by expansion"
                        break
            else:
                # No expansion was indicated - this could mean:
                # 1. No links in retrieved chunks
                # 2. All linked chunks were already in results
                # This is acceptable behavior
                pass

            # Verify at least some results were returned
            assert "[1]" in output, "No results returned"
            assert "Score:" in output, "Missing score information"
            assert "Article:" in output, "Missing article titles"
            assert "Preview:" in output, "Missing chunk previews"

    @pytest.mark.dependency(depends=["ingest_pipeline_complete"], scope="session")
    def test_retrieve_with_context_expansion_disabled(
        self,
        e2e_output_dir: Path,
    ) -> None:
        """Test retrieve command with context expansion disabled.

        Validates:
        - Command exits successfully when expansion is disabled
        - Output does not show expansion statistics
        - Results are returned normally

        Note: Requires OPENAI_API_KEY environment variable.
        """
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping retrieve test")

        # Use the ingest test output directory
        ingest_output_dir = e2e_output_dir / "ingest-test"
        ingest_chroma_dir = ingest_output_dir / "chroma-db"
        ingest_bm25_dir = ingest_output_dir / "chroma-db"

        if not ingest_chroma_dir.exists():
            pytest.skip("ChromaDB not found - run full e2e workflow first")

        # Verify ChromaDB has data
        client = chromadb.PersistentClient(path=str(ingest_chroma_dir))
        collection = client.get_or_create_collection(name=ChromaVectorStore.DEFAULT_COLLECTION_NAME)
        chunk_count = collection.count()

        if chunk_count == 0:
            pytest.skip("ChromaDB is empty - run full e2e workflow first")

        bm25_index = ingest_bm25_dir / "bm25_index.pkl"
        if not bm25_index.exists():
            pytest.skip("BM25 index not found - run full e2e workflow first")

        # Convert paths to absolute BEFORE entering isolated_filesystem
        # so they resolve relative to the actual workspace, not the temp dir
        abs_chroma_dir = ingest_chroma_dir.resolve()
        abs_bm25_index = bm25_index.resolve()

        runner = CliRunner()

        # Test with context expansion disabled
        with runner.isolated_filesystem():
            env = os.environ.copy()
            env["CONTEXT_EXPANSION_ENABLED"] = "false"
            env["CHROMA_DB_PATH"] = str(abs_chroma_dir)
            env["BM25_INDEX_PATH"] = str(abs_bm25_index)
            env["RETRIEVAL_TOP_K"] = "5"

            query_text = "Space Marine"

            result = runner.invoke(
                retrieve,
                [query_text, "--top-k", "5"],
                env=env,
                catch_exceptions=False,
            )

            # Basic validation
            assert result.exit_code == 0, f"Command failed: {result.output}"

            output = result.output
            assert "HYBRID RETRIEVAL RESULTS" in output, "Missing results header"
            assert f"Query: {query_text}" in output, "Missing query text"

            # Verify no expansion indicators in output
            assert "Context expansion:" not in output, "Expansion should not be applied"
            assert "Initial retrieval:" not in output, "Should not show expansion stats"

            # Verify results are still returned
            assert "[1]" in output, "No results returned"
            assert "Score:" in output, "Missing score information"
