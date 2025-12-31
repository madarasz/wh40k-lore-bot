"""Metadata extraction from wiki chunks."""

import re
from collections import Counter
from typing import Any

import structlog

from src.common.constants import (
    CONTENT_TYPE_KEYWORDS,
    ERAS,
    FACTION_ALIASES,
    FACTIONS,
    SOURCE_BOOK_PATTERNS,
    SPOILER_KEYWORDS,
)
from src.ingestion.models import Chunk

logger = structlog.get_logger(__name__)

# Constants for content type classification
CHARACTER_DENSITY_THRESHOLD = 3  # Minimum character names to classify as "character" type


class MetadataExtractor:
    """Extracts metadata from wiki chunk content.

    Performs keyword-based detection for factions, eras, character names,
    content types, spoilers, and source books.
    """

    def _detect_faction(self, text: str) -> str | None:
        """Detect the most frequently mentioned faction in text.

        Uses word-boundary regex matching to avoid substring collisions
        (e.g., "Eldar" within "Craftworld Eldar"). Case-insensitive.

        Args:
            text: Content to analyze

        Returns:
            Most frequent faction name, or None if no faction found
        """
        text_lower = text.lower()
        faction_counts: dict[str, int] = {}

        # Count occurrences of each faction using word-boundary regex
        for faction in FACTIONS:
            # Escape special regex characters and create word-boundary pattern
            escaped_faction = re.escape(faction.lower())
            pattern = r"\b" + escaped_faction + r"\b"
            matches = re.findall(pattern, text_lower)
            count = len(matches)

            if count > 0:
                # Normalize using aliases if available
                normalized = FACTION_ALIASES.get(faction.lower(), faction)
                faction_counts[normalized] = faction_counts.get(normalized, 0) + count

        if not faction_counts:
            return None

        # Return most frequent faction
        most_common = max(faction_counts, key=faction_counts.get)  # type: ignore
        logger.debug(
            "faction_detected",
            faction=most_common,
            count=faction_counts[most_common],
        )
        return most_common

    def _detect_eras(self, text: str) -> list[str]:
        """Detect all eras mentioned in text.

        Args:
            text: Content to analyze

        Returns:
            List of era names found (deduplicated)
        """
        text_lower = text.lower()
        eras_found = set()

        for era in ERAS:
            if era.lower() in text_lower:
                eras_found.add(era)

        result = sorted(eras_found)
        if result:
            logger.debug("eras_detected", eras=result, count=len(result))
        return result

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

    def _classify_content_type(self, text: str, character_names: list[str]) -> str:
        """Classify content type based on keyword analysis.

        Types:
        - "character": High density of character names (3+)
        - "military": Battle/war related content
        - "technology": Weapons/equipment related content
        - "lore": Default for general content

        Args:
            text: Content to analyze
            character_names: List of character names already extracted

        Returns:
            Content type classification
        """
        text_lower = text.lower()

        # Check character density first
        if len(character_names) >= CHARACTER_DENSITY_THRESHOLD:
            logger.debug("content_type_classified", type="character", reason="high_name_density")
            return "character"

        # Score each content type by keyword matches
        scores = {}
        for content_type, keywords in CONTENT_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[content_type] = score

        # Return type with highest score, or "lore" as default
        if max(scores.values(), default=0) > 0:
            best_type = max(scores, key=scores.get)  # type: ignore
            logger.debug(
                "content_type_classified",
                type=best_type,
                score=scores[best_type],
            )
            return best_type

        logger.debug("content_type_classified", type="lore", reason="default")
        return "lore"

    def _detect_spoiler(self, text: str) -> bool:
        """Detect if content contains spoiler keywords.

        Default is False since Fandom wiki is primarily canon content.

        Args:
            text: Content to analyze

        Returns:
            True if spoiler keywords detected, False otherwise
        """
        text_lower = text.lower()

        for keyword in SPOILER_KEYWORDS:
            if keyword.lower() in text_lower:
                logger.debug("spoiler_detected", keyword=keyword)
                return True

        return False

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
        """Extract all metadata from a chunk.

        Combines all detection methods to produce complete metadata dictionary.

        Args:
            chunk: Chunk object containing text to analyze

        Returns:
            Dictionary with metadata fields:
                - faction: str | None
                - eras: List[str]
                - character_names: List[str]
                - content_type: str
                - spoiler_flag: bool
                - source_books: List[str]

        Raises:
            ValueError: If chunk or chunk_text is invalid
        """
        if not chunk or not chunk.chunk_text:
            raise ValueError("Chunk and chunk_text cannot be empty")

        text = chunk.chunk_text

        # Extract all metadata
        faction = self._detect_faction(text)
        eras = self._detect_eras(text)
        character_names = self._extract_character_names(text)
        content_type = self._classify_content_type(text, character_names)
        spoiler_flag = self._detect_spoiler(text)
        source_books = self._extract_source_books(text)

        metadata = {
            "faction": faction,
            "eras": eras,
            "character_names": character_names,
            "content_type": content_type,
            "spoiler_flag": spoiler_flag,
            "source_books": source_books,
        }

        logger.info(
            "metadata_extracted",
            article_title=chunk.article_title,
            chunk_index=chunk.chunk_index,
            faction=faction,
            eras_count=len(eras),
            characters_count=len(character_names),
            content_type=content_type,
            spoiler=spoiler_flag,
            books_count=len(source_books),
        )

        return metadata
