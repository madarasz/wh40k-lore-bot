"""E2E test fixtures and configuration."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def e2e_test_xml_path() -> Path:
    """Return path to minimal E2E test XML file.

    Returns:
        Path to the E2E test XML file

    Raises:
        pytest.skip: If XML file not found
    """
    xml_path = Path("./test-data/e2e-test-wiki.xml")
    if not xml_path.exists():
        pytest.skip(
            "E2E test XML not found. "
            "Run 'poetry run python tests/e2e/fixtures/extract_minimal_xml.py' first."
        )
    return xml_path


@pytest.fixture(scope="session")
def e2e_output_dir() -> Path:
    """Create persistent output directory for E2E test artifacts.

    Returns:
        Path to the E2E output directory
    """
    output_dir = Path("./test-data/e2e-outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session", autouse=True)
def cleanup_e2e_outputs(e2e_output_dir: Path) -> None:
    """Clean up E2E outputs before test session.

    This fixture runs automatically at the start of each E2E test session
    to ensure a clean state.

    Args:
        e2e_output_dir: Path to the E2E output directory
    """
    if e2e_output_dir.exists():
        shutil.rmtree(e2e_output_dir)
    e2e_output_dir.mkdir(parents=True, exist_ok=True)
