"""Shared utilities for CLI commands."""

from pathlib import Path


def load_wiki_ids(wiki_ids_file: Path) -> list[str]:
    """Load wiki IDs from a text file (one per line).

    Args:
        wiki_ids_file: Path to file containing wiki IDs

    Returns:
        List of wiki ID strings

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    if not wiki_ids_file.exists():
        raise FileNotFoundError(f"Wiki IDs file not found: {wiki_ids_file}")

    wiki_ids = []
    with wiki_ids_file.open("r") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                wiki_ids.append(stripped_line)

    if not wiki_ids:
        raise ValueError(f"No wiki IDs found in file: {wiki_ids_file}")

    return wiki_ids
