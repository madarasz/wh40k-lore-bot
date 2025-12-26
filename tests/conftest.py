"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Return path to temporary test database."""
    return tmp_path / "test.db"
