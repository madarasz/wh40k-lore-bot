"""Utilities for chunk ID generation."""


def generate_chunk_id(wiki_page_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID from wiki_page_id and chunk_index.

    Format: {wiki_page_id}_{chunk_index}
    Example: "58_0", "58_1", "100_5"

    This enables reliable upsert/delete operations for re-ingestion.

    Args:
        wiki_page_id: Wiki page ID from markdown frontmatter
        chunk_index: Zero-based index of chunk within article

    Returns:
        Deterministic ID string
    """
    return f"{wiki_page_id}_{chunk_index}"
