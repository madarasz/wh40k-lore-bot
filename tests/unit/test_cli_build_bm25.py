"""Unit tests for build-bm25 CLI command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.build_bm25 import _load_chunks_from_json, build_bm25
from src.ingestion.models import Chunk


class TestLoadChunksFromJSON:
    """Test _load_chunks_from_json helper function."""

    def test_load_chunks_from_valid_json(self, tmp_path: Path) -> None:
        """Test loading chunks from valid JSON file."""
        # Create test JSON file
        chunks_data = {
            "version": "1.0",
            "chunks": [
                {
                    "chunk_text": "Roboute Guilliman is the Primarch of the Ultramarines.",
                    "article_title": "Roboute Guilliman",
                    "section_path": "Root > Overview",
                    "chunk_index": 0,
                    "metadata": {"links": ["Ultramarines", "Primarch"]},
                },
                {
                    "chunk_text": "The Ultramarines are a Space Marine Chapter.",
                    "article_title": "Ultramarines",
                    "section_path": "Root",
                    "chunk_index": 0,
                    "metadata": {"links": ["Space Marines"]},
                },
            ],
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        # Load chunks
        chunks = _load_chunks_from_json(chunks_file)

        # Verify
        assert len(chunks) == 2
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].chunk_text == "Roboute Guilliman is the Primarch of the Ultramarines."
        assert chunks[0].article_title == "Roboute Guilliman"
        assert chunks[0].section_path == "Root > Overview"
        assert chunks[0].chunk_index == 0
        assert chunks[0].links == ["Ultramarines", "Primarch"]

        assert chunks[1].chunk_text == "The Ultramarines are a Space Marine Chapter."
        assert chunks[1].article_title == "Ultramarines"

    def test_load_chunks_from_json_without_links(self, tmp_path: Path) -> None:
        """Test loading chunks from JSON without links metadata."""
        chunks_data = {
            "chunks": [
                {
                    "chunk_text": "Test chunk",
                    "article_title": "Test Article",
                    "section_path": "Root",
                    "chunk_index": 0,
                    "metadata": {},
                }
            ]
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        chunks = _load_chunks_from_json(chunks_file)

        assert len(chunks) == 1
        assert chunks[0].links == []

    def test_load_chunks_from_empty_json_raises_error(self, tmp_path: Path) -> None:
        """Test that empty chunks list raises error."""
        chunks_data = {"chunks": []}

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        with pytest.raises(ValueError, match="No chunks found"):
            _load_chunks_from_json(chunks_file)

    def test_load_chunks_from_missing_field_raises_error(self, tmp_path: Path) -> None:
        """Test that missing required field raises error."""
        chunks_data = {
            "chunks": [
                {
                    "chunk_text": "Test",
                    "article_title": "Test",
                    # Missing section_path
                    "chunk_index": 0,
                }
            ]
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        with pytest.raises(ValueError, match="Missing required field"):
            _load_chunks_from_json(chunks_file)


class TestBuildBM25CLI:
    """Test build-bm25 CLI command."""

    def test_build_bm25_help_shows_options(self) -> None:
        """Test that --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(build_bm25, ["--help"])

        assert result.exit_code == 0
        assert "CHUNKS_FILE" in result.output
        assert "--output" in result.output
        assert "Build BM25 keyword search index" in result.output

    def test_build_bm25_success_with_mocked_repository(self, tmp_path: Path) -> None:
        """Test build-bm25 succeeds with mocked BM25Repository."""
        runner = CliRunner()

        # Create test chunks JSON
        chunks_data = {
            "chunks": [
                {
                    "chunk_text": "Roboute Guilliman is the Primarch.",
                    "article_title": "Roboute Guilliman",
                    "section_path": "Root",
                    "chunk_index": 0,
                    "metadata": {},
                }
            ]
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        output_file = tmp_path / "bm25_index.pkl"

        # Mock BM25Repository
        with patch("src.cli.build_bm25.BM25Repository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_index_stats.return_value = {
                "total_chunks": 1,
                "unique_tokens": 5,
            }

            # Mock the save_index method to create a file
            def create_index_file(path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("mock index")

            mock_repo.save_index.side_effect = create_index_file

            result = runner.invoke(build_bm25, [str(chunks_file), "--output", str(output_file)])

            # Verify
            assert result.exit_code == 0
            assert "BM25 Index Building Complete!" in result.output
            assert "Total Chunks: 1" in result.output
            assert "Unique Tokens: 5" in result.output

            # Verify BM25Repository methods were called
            mock_repo.build_index.assert_called_once()
            mock_repo.save_index.assert_called_once()

    def test_build_bm25_invalid_chunks_file_aborts(self) -> None:
        """Test that invalid chunks file causes abort."""
        runner = CliRunner()
        result = runner.invoke(build_bm25, ["nonexistent.json"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "error" in result.output.lower()

    def test_build_bm25_malformed_json_aborts(self, tmp_path: Path) -> None:
        """Test that malformed JSON causes abort."""
        runner = CliRunner()

        chunks_file = tmp_path / "bad.json"
        chunks_file.write_text("not valid json{{{")

        result = runner.invoke(build_bm25, [str(chunks_file)])

        assert result.exit_code != 0
        assert "Failed to load chunks" in result.output

    def test_build_bm25_uses_env_var_for_default_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that BM25_INDEX_PATH env var is used for default output."""
        runner = CliRunner()

        # Create test chunks JSON
        chunks_data = {
            "chunks": [
                {
                    "chunk_text": "Test chunk",
                    "article_title": "Test",
                    "section_path": "Root",
                    "chunk_index": 0,
                    "metadata": {},
                }
            ]
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        # Set env var
        custom_path = tmp_path / "custom-bm25.pkl"
        monkeypatch.setenv("BM25_INDEX_PATH", str(custom_path))

        # Mock BM25Repository
        with patch("src.cli.build_bm25.BM25Repository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_index_stats.return_value = {
                "total_chunks": 1,
                "unique_tokens": 3,
            }

            def create_index_file(path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("mock index")

            mock_repo.save_index.side_effect = create_index_file

            result = runner.invoke(build_bm25, [str(chunks_file)])

            # Verify env var path was used
            assert result.exit_code == 0
            assert str(custom_path) in result.output

    def test_build_bm25_repository_build_failure_aborts(self, tmp_path: Path) -> None:
        """Test that BM25Repository build failure causes abort."""
        runner = CliRunner()

        chunks_data = {
            "chunks": [
                {
                    "chunk_text": "Test",
                    "article_title": "Test",
                    "section_path": "Root",
                    "chunk_index": 0,
                    "metadata": {},
                }
            ]
        }

        chunks_file = tmp_path / "chunks.json"
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f)

        # Mock BM25Repository to raise error on build
        with patch("src.cli.build_bm25.BM25Repository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.build_index.side_effect = RuntimeError("Index build failed")

            result = runner.invoke(build_bm25, [str(chunks_file)])

            assert result.exit_code != 0
            assert "Index building" in result.output or "failed" in result.output
