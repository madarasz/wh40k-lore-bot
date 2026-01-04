"""Metadata extraction from wiki chunks.

This module has been simplified as part of chunk schema refactoring.
Faction, era, spoiler, and content_type detection have been removed
as they were never meaningfully populated from wiki data.
"""

import re
from collections import Counter
from typing import Any

import structlog

from src.common.constants import SOURCE_BOOK_PATTERNS
from src.ingestion.models import Chunk

logger = structlog.get_logger(__name__)


class MetadataExtractor:
    """Extracts metadata from wiki chunk content.

    Currently extracts:
    - character_names: Notable names from wiki links
    - source_books: Source book references
    """

    def _extract_character_names(self, text: str) -> list[str]:
        """Extract character names from wiki links.

        Extracts names from markdown wiki links: [[Name]] or [[Name|Display]].
        Returns top 5 most mentioned.

        Args:
            text: Content to analyze

        Returns:
            List of up to 5 most mentioned character names
        """
        # Extract wiki links: [[Name]] or [[Name|Display]]
        pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        matches = re.findall(pattern, text)

        if not matches:
            return []

        # Count mentions
        counter = Counter(matches)

        # Return top 5
        top_names = [name for name, _count in counter.most_common(5)]
        logger.debug("character_names_extracted", names=top_names, count=len(top_names))
        return top_names

    def _extract_source_books(self, text: str) -> list[str]:
        """Extract source book references from text.

        Uses regex patterns to find book titles like:
        - Codex: Space Marines
        - Horus Heresy: Book 1: Betrayal
        - White Dwarf 123

        Args:
            text: Content to analyze

        Returns:
            List of source book titles (deduplicated)
        """
        books_found = set()

        for pattern in SOURCE_BOOK_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                # Handle tuple results from multi-group patterns
                if isinstance(match, tuple):
                    # Join non-empty groups
                    book_title = " ".join(str(m) for m in match if m).strip()
                else:
                    book_title = str(match).strip()

                if book_title:
                    books_found.add(book_title)

        result = sorted(books_found)
        if result:
            logger.debug("source_books_extracted", books=result, count=len(result))
        return result

    def extract_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Extract metadata from a chunk.

        Args:
            chunk: Chunk object containing text to analyze

        Returns:
            Dictionary with metadata fields:
                - character_names: List[str]
                - source_books: List[str]

        Raises:
            ValueError: If chunk or chunk_text is invalid
        """
        if not chunk or not chunk.chunk_text:
            raise ValueError("Chunk and chunk_text cannot be empty")

        text = chunk.chunk_text

        # Extract remaining useful metadata
        character_names = self._extract_character_names(text)
        source_books = self._extract_source_books(text)

        metadata = {
            "character_names": character_names,
            "source_books": source_books,
        }

        logger.debug(
            "metadata_extracted",
            article_title=chunk.article_title,
            chunk_index=chunk.chunk_index,
            characters_count=len(character_names),
            books_count=len(source_books),
        )

        return metadata
