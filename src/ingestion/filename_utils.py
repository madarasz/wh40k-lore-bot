"""Filename sanitization utilities for markdown archive."""

import re
import unicodedata


def sanitize_filename(title: str, max_length: int = 255) -> str:
    """Sanitize article title for use as filename.

    Transforms the title to be filesystem-safe by:
    - Converting spaces to underscores
    - Converting slashes and colons to hyphens
    - Removing special characters (keeping only alphanumeric, underscore, hyphen)
    - Normalizing unicode characters to ASCII equivalents
    - Limiting length to max_length characters

    Args:
        title: The article title to sanitize
        max_length: Maximum filename length (default: 255)

    Returns:
        Sanitized filename string (without .md extension)

    Raises:
        ValueError: If title is empty or becomes empty after sanitization
    """
    if not title or not title.strip():
        raise ValueError("Title cannot be empty")

    # Normalize unicode to decomposed form, then encode to ASCII
    # This converts accented characters like "é" to "e"
    normalized = unicodedata.normalize("NFKD", title)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")

    # Convert spaces to underscores
    sanitized = ascii_str.replace(" ", "_")

    # Convert slashes and colons to hyphens
    sanitized = sanitized.replace("/", "-")
    sanitized = sanitized.replace(":", "-")

    # Remove all characters except alphanumeric, underscore, and hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", sanitized)

    # Remove consecutive identical underscores or hyphens
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)

    # Strip leading/trailing underscores and hyphens
    sanitized = sanitized.strip("_-")

    if not sanitized:
        raise ValueError(f"Title '{title}' resulted in empty filename after sanitization")

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_-")

    return sanitized
