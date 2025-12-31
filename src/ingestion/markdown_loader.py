"""Markdown file loader for ingestion pipeline.

Loads markdown files with YAML frontmatter from the markdown archive
and converts them to WikiArticle objects for processing.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.ingestion.models import WikiArticle

logger = structlog.get_logger(__name__)


class MarkdownLoader:
    """Load markdown files from archive directory.

    Supports:
    - Glob pattern matching for file selection
    - Filtering by wiki_id list
    - YAML frontmatter parsing
    - Memory-efficient iteration

    Example:
        >>> loader = MarkdownLoader()
        >>> for article in loader.load_all():
        ...     print(f"{article.title}: {article.word_count} words")

        >>> # Filter by wiki IDs
        >>> for article in loader.load_all(wiki_ids=["58", "100"]):
        ...     process(article)
    """

    DEFAULT_ARCHIVE_PATH = Path("data/markdown-archive")

    def __init__(self, archive_path: str | Path | None = None) -> None:
        """Initialize loader.

        Args:
            archive_path: Path to markdown archive directory.
                         Defaults to data/markdown-archive
        """
        self.archive_path = Path(archive_path) if archive_path else self.DEFAULT_ARCHIVE_PATH
        self.logger = logger.bind(component="markdown_loader")
        self.files_loaded = 0
        self.files_skipped = 0

    def load_all(
        self,
        wiki_ids: list[str] | None = None,
        pattern: str = "*.md",
    ) -> Iterator[WikiArticle]:
        """Load all markdown files from archive.

        Args:
            wiki_ids: Optional filter - only load files matching these wiki IDs.
                     If None, loads all files.
            pattern: Glob pattern for file matching (default: "*.md")

        Yields:
            WikiArticle objects for each valid markdown file

        Raises:
            FileNotFoundError: If archive directory doesn't exist
        """
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive directory not found: {self.archive_path}")

        if not self.archive_path.is_dir():
            raise NotADirectoryError(f"Archive path is not a directory: {self.archive_path}")

        # Convert wiki_ids to set for O(1) lookup
        wiki_id_set = set(wiki_ids) if wiki_ids else None

        self.logger.info(
            "loading_markdown_files",
            archive_path=str(self.archive_path),
            pattern=pattern,
            wiki_ids_filter=bool(wiki_id_set),
            wiki_ids_count=len(wiki_id_set) if wiki_id_set else None,
        )

        # Reset counters
        self.files_loaded = 0
        self.files_skipped = 0

        # Iterate through files
        for file_path in sorted(self.archive_path.glob(pattern)):
            article = self.load_file(file_path)

            if article is None:
                self.files_skipped += 1
                continue

            # Apply wiki_id filter if provided
            if wiki_id_set and article.wiki_id not in wiki_id_set:
                self.files_skipped += 1
                continue

            self.files_loaded += 1

            # Log progress every 100 files
            if self.files_loaded % 100 == 0:
                self.logger.info(
                    "loading_progress",
                    files_loaded=self.files_loaded,
                    files_skipped=self.files_skipped,
                )

            yield article

        self.logger.info(
            "loading_complete",
            total_loaded=self.files_loaded,
            total_skipped=self.files_skipped,
        )

    def load_file(self, file_path: Path) -> WikiArticle | None:
        """Load a single markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            WikiArticle if successful, None if parsing fails
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            frontmatter, markdown_content = self._parse_frontmatter(content)

            # Validate required fields
            if not self._validate_frontmatter(frontmatter):
                self.logger.warning(
                    "invalid_frontmatter",
                    file_path=str(file_path),
                    frontmatter_keys=list(frontmatter.keys()),
                )
                return None

            # Create WikiArticle
            return WikiArticle(
                title=frontmatter["title"],
                wiki_id=str(frontmatter["wiki_id"]),
                last_updated=str(frontmatter["last_updated"]),
                content=markdown_content,
                word_count=frontmatter.get("word_count", len(markdown_content.split())),
            )

        except (OSError, yaml.YAMLError, ValueError) as e:
            self.logger.error(
                "file_load_error",
                file_path=str(file_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content.

        Args:
            content: Full file content including frontmatter

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)

        Raises:
            ValueError: If frontmatter is missing or invalid
        """
        # Split into lines, preserving line structure
        lines = content.split("\n")

        # Check for opening delimiter (handle optional \r from CRLF)
        if not lines or lines[0].rstrip("\r") != "---":
            raise ValueError("Missing YAML frontmatter (file must start with '---')")

        # Find closing delimiter
        closing_index = None
        for i in range(1, len(lines)):
            if lines[i].rstrip("\r") == "---":
                closing_index = i
                break

        if closing_index is None:
            raise ValueError("Invalid YAML frontmatter (missing closing '---')")

        # Extract frontmatter lines (between delimiters), strip \r from each line
        frontmatter_lines = [line.rstrip("\r") for line in lines[1:closing_index]]
        frontmatter_str = "\n".join(frontmatter_lines)
        frontmatter = yaml.safe_load(frontmatter_str)

        if not isinstance(frontmatter, dict):
            raise ValueError("YAML frontmatter must be a dictionary")

        # Extract content after frontmatter, strip \r and join
        content_lines = [line.rstrip("\r") for line in lines[closing_index + 1 :]]
        markdown_content = "\n".join(content_lines).strip()

        return frontmatter, markdown_content

    def _validate_frontmatter(self, frontmatter: dict[str, Any]) -> bool:
        """Validate required frontmatter fields.

        Required fields: title, wiki_id, last_updated

        Args:
            frontmatter: Parsed frontmatter dictionary

        Returns:
            True if all required fields present and valid
        """
        required_fields = ["title", "wiki_id", "last_updated"]

        for field in required_fields:
            if field not in frontmatter:
                self.logger.warning("missing_frontmatter_field", field=field)
                return False

            if frontmatter[field] is None or str(frontmatter[field]).strip() == "":
                self.logger.warning("empty_frontmatter_field", field=field)
                return False

        return True

    def get_statistics(self) -> dict[str, int]:
        """Get loader statistics.

        Returns:
            Dictionary with files_loaded, files_skipped counts
        """
        return {
            "files_loaded": self.files_loaded,
            "files_skipped": self.files_skipped,
        }

    def get_all_wiki_ids(self, pattern: str = "*.md") -> list[str]:
        """Get all wiki IDs from archive without loading full content.

        Useful for planning/filtering before full load.

        Args:
            pattern: Glob pattern for file matching

        Returns:
            List of wiki_id strings
        """
        wiki_ids = []

        for file_path in self.archive_path.glob(pattern):
            try:
                # Read just enough to get frontmatter
                content = file_path.read_text(encoding="utf-8")
                frontmatter, _ = self._parse_frontmatter(content)

                if "wiki_id" in frontmatter:
                    wiki_ids.append(str(frontmatter["wiki_id"]))

            except (OSError, yaml.YAMLError, ValueError):
                continue

        return wiki_ids

    def count_files(self, pattern: str = "*.md") -> int:
        """Count number of markdown files in archive.

        Args:
            pattern: Glob pattern for file matching

        Returns:
            Number of matching files
        """
        return len(list(self.archive_path.glob(pattern)))
