"""Intelligent markdown text chunking with hierarchical splitting strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
import tiktoken

if TYPE_CHECKING:
    from src.ingestion.models import Chunk

logger = structlog.get_logger(__name__)

# Chunking constants
MAX_TOKENS = 500
MIN_TOKENS = 50

# Regex pattern for markdown internal links: [Display Text](Link Title)
# Excludes external links (http/https) and anchors (#)
# Handles one level of nested parentheses in link targets (e.g., "Space Marine (Game)")
INTERNAL_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


@dataclass
class Section:
    """Represents a markdown section with hierarchy.

    Attributes:
        text: Section content (including header)
        section_path: Hierarchical path (e.g., "History > The Great Crusade")
        level: Header level (2 for ##, 3 for ###, etc.)
    """

    text: str
    section_path: str
    level: int


class MarkdownChunker:
    """Intelligently chunks markdown into semantically coherent pieces.

    Chunking strategy hierarchy:
    1. PRIMARY: Split by section headers (##, ###, ####)
    2. SECONDARY: If section > 500 tokens, split by paragraphs
    3. TERTIARY: If paragraph > 500 tokens, split by sentences
    4. POST-PROCESSING: Merge chunks < 50 tokens

    All chunks maintain section hierarchy in metadata.
    """

    def __init__(self) -> None:
        """Initialize chunker with tiktoken encoder."""
        self._encoder = tiktoken.get_encoding("cl100k_base")
        logger.info("markdown_chunker_initialized", encoding="cl100k_base")

    def chunk_markdown(
        self,
        markdown: str,
        article_title: str,
        infobox: str | None = None,
        infobox_links: list[str] | None = None,
    ) -> list[Chunk]:
        """Chunk markdown into semantically coherent pieces.

        If an infobox is provided, it becomes chunk_index=0 with section_path="Infobox".
        Content chunks follow with sequential indices.

        Args:
            markdown: Markdown content to chunk
            article_title: Title of the source article
            infobox: Optional formatted infobox text
            infobox_links: Optional list of links from the infobox

        Returns:
            List of Chunk objects with text and metadata

        Raises:
            ValueError: If markdown is empty or article_title is empty
        """
        if not markdown or not markdown.strip():
            raise ValueError("Markdown content cannot be empty")
        if not article_title or not article_title.strip():
            raise ValueError("Article title cannot be empty")

        all_chunks: list[Chunk] = []

        # Add infobox as first chunk if present
        if infobox:
            # Import here to avoid circular dependency at module level
            from src.ingestion.models import Chunk as ChunkModel  # noqa: PLC0415

            infobox_chunk = ChunkModel(
                chunk_text=infobox,
                article_title=article_title,
                section_path="Infobox",
                chunk_index=0,
                links=infobox_links or [],
            )
            all_chunks.append(infobox_chunk)

        # Parse sections from markdown
        sections = self._parse_sections(markdown)

        # Handle edge case: no sections (article with no headers)
        if not sections:
            # Treat entire article as single section with article_title as path
            sections = [Section(text=markdown, section_path=article_title, level=1)]

        # Chunk each section
        for section in sections:
            section_chunks = self._chunk_section(section, article_title)
            all_chunks.extend(section_chunks)

        # Merge tiny chunks across section boundaries (skip infobox)
        # Only merge if sections are header-only (very small)
        if infobox and len(all_chunks) > 1:
            # Keep infobox separate, merge only content chunks
            infobox_chunk = all_chunks[0]
            content_chunks = self._merge_header_only_chunks(all_chunks[1:])
            all_chunks = [infobox_chunk] + content_chunks
        else:
            all_chunks = self._merge_header_only_chunks(all_chunks)

        # Assign chunk indices
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx

        logger.info(
            "markdown_chunked",
            article_title=article_title,
            total_chunks=len(all_chunks),
            has_infobox=infobox is not None,
            avg_tokens=sum(self._count_tokens(c.chunk_text) for c in all_chunks) // len(all_chunks)
            if all_chunks
            else 0,
        )

        return all_chunks

    def _parse_sections(self, markdown: str) -> list[Section]:
        """Parse markdown into sections based on headers.

        Args:
            markdown: Markdown content

        Returns:
            List of Section objects with hierarchy
        """
        sections: list[Section] = []
        lines = markdown.split("\n")

        # Track section hierarchy
        hierarchy: list[str] = []
        current_section_lines: list[str] = []
        current_level = 0

        for line in lines:
            # Check for header (##, ###, ####)
            header_match = re.match(r"^(#{2,4})\s+(.+)$", line)

            if header_match:
                # Save previous section if exists
                if current_section_lines:
                    section_text = "\n".join(current_section_lines).strip()
                    if section_text:  # Skip empty sections
                        section_path = " > ".join(hierarchy) if hierarchy else "Root"
                        sections.append(
                            Section(
                                text=section_text, section_path=section_path, level=current_level
                            )
                        )
                    current_section_lines = []

                # Extract new header info
                level = len(header_match.group(1))  # Count #'s
                title = header_match.group(2).strip()

                # Update hierarchy
                if not hierarchy:
                    hierarchy = [title]
                elif level == current_level:
                    # Same level: replace last
                    hierarchy[-1] = title
                elif level > current_level:
                    # Deeper level: append
                    hierarchy.append(title)
                else:
                    # Shallower level: pop and replace
                    depth_diff = current_level - level
                    hierarchy = hierarchy[: -(depth_diff + 1)] + [title]

                current_level = level
                current_section_lines.append(line)  # Include header in section
            else:
                current_section_lines.append(line)

        # Don't forget last section
        if current_section_lines:
            section_text = "\n".join(current_section_lines).strip()
            if section_text and hierarchy:  # Only add if we found headers
                section_path = " > ".join(hierarchy)
                sections.append(
                    Section(text=section_text, section_path=section_path, level=current_level)
                )

        return sections

    def _chunk_section(  # noqa: PLR0912
        self, section: Section, article_title: str
    ) -> list[Chunk]:
        """Chunk a single section using hierarchical strategy.

        Args:
            section: Section to chunk
            article_title: Source article title

        Returns:
            List of Chunk objects
        """
        # Import here to avoid circular dependency at module level
        from src.ingestion.models import Chunk  # noqa: PLC0415

        token_count = self._count_tokens(section.text)

        # Case 1: Section fits in one chunk
        if token_count <= MAX_TOKENS:
            # Still check minimum
            if token_count < MIN_TOKENS:
                # Too small, but we'll keep it (will merge later if needed)
                pass
            return [
                Chunk(
                    chunk_text=section.text,
                    article_title=article_title,
                    section_path=section.section_path,
                    chunk_index=0,  # Will be reassigned later
                    links=self._extract_links(section.text),
                )
            ]

        # Case 2: Section too large, split by paragraphs
        paragraphs = self._split_by_paragraphs(section.text)

        # Extract header if present
        header = ""
        header_match = re.match(r"^(#{2,4}\s+.+)$", paragraphs[0], re.MULTILINE)
        if header_match:
            header = paragraphs[0].split("\n")[0] + "\n\n"
            # Remove header from first paragraph
            paragraphs[0] = "\n".join(paragraphs[0].split("\n")[1:]).strip()

        chunks: list[Chunk] = []
        current_chunk_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self._count_tokens(para)

            # If single paragraph is too large, split by sentences
            if para_tokens > MAX_TOKENS:
                # Flush current chunk
                if current_chunk_parts:
                    chunk_text = header + "\n\n".join(current_chunk_parts)
                    chunks.append(
                        Chunk(
                            chunk_text=chunk_text,
                            article_title=article_title,
                            section_path=section.section_path,
                            chunk_index=0,
                            links=self._extract_links(chunk_text),
                        )
                    )
                    current_chunk_parts = []
                    current_tokens = 0

                # Split paragraph by sentences
                sentences = self._split_by_sentences(para)
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence)

                    if current_tokens + sentence_tokens > MAX_TOKENS:
                        # Flush current chunk
                        if current_chunk_parts:
                            chunk_text = header + " ".join(current_chunk_parts)
                            chunks.append(
                                Chunk(
                                    chunk_text=chunk_text,
                                    article_title=article_title,
                                    section_path=section.section_path,
                                    chunk_index=0,
                                    links=self._extract_links(chunk_text),
                                )
                            )
                            current_chunk_parts = []
                            current_tokens = 0

                    current_chunk_parts.append(sentence)
                    current_tokens += sentence_tokens

            else:
                # Paragraph fits, check if adding it exceeds limit
                if current_tokens + para_tokens > MAX_TOKENS:
                    # Flush current chunk
                    if current_chunk_parts:
                        chunk_text = header + "\n\n".join(current_chunk_parts)
                        chunks.append(
                            Chunk(
                                chunk_text=chunk_text,
                                article_title=article_title,
                                section_path=section.section_path,
                                chunk_index=0,
                                links=self._extract_links(chunk_text),
                            )
                        )
                        current_chunk_parts = []
                        current_tokens = 0

                current_chunk_parts.append(para)
                current_tokens += para_tokens

        # Don't forget last chunk
        if current_chunk_parts:
            chunk_text = header + "\n\n".join(current_chunk_parts)
            chunks.append(
                Chunk(
                    chunk_text=chunk_text,
                    article_title=article_title,
                    section_path=section.section_path,
                    chunk_index=0,
                    links=self._extract_links(chunk_text),
                )
            )

        # Post-process: merge tiny chunks
        chunks = self._merge_tiny_chunks(chunks)

        return chunks

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraph boundaries.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = re.split(r"\n\n+", text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentence boundaries.

        Handles edge cases:
        - Abbreviations (Dr., Mr., vs.)
        - Decimals (3.14, 99.9%)
        - Ellipsis (...)

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Pattern: sentence boundary is period/exclamation/question
        # followed by space and capital letter
        # but NOT after single capital letter (abbreviation) or ellipsis
        # Note: We don't exclude digits since "3.14. Next sentence" should split
        pattern = r"(?<=[.!?])(?<![A-Z]\.)(?<!\.\.\.)\s+(?=[A-Z\"])"

        # Split but handle common abbreviations
        # First, protect common abbreviations
        protected_text = text
        abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "vs.", "etc.", "i.e.", "e.g."]
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            protected_text = protected_text.replace(abbr, placeholder)

        # Split on sentence boundaries
        sentences = re.split(pattern, protected_text)

        # Restore abbreviations
        result = []
        for sent in sentences:
            restored_sent = sent
            for placeholder, abbr in placeholders.items():
                restored_sent = restored_sent.replace(placeholder, abbr)
            if restored_sent.strip():
                result.append(restored_sent.strip())

        return result

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def _extract_links(self, text: str) -> list[str]:
        """Extract internal wiki links from markdown text.

        Extracts link targets from markdown-formatted internal links.
        Excludes external links (http/https) and anchor links (#).

        Args:
            text: Markdown text to extract links from

        Returns:
            Deduplicated list of internal link targets
        """
        if not text:
            return []

        links: list[str] = []
        for match in INTERNAL_LINK_PATTERN.finditer(text):
            link_target = match.group(2)
            # Skip external links and anchors
            if link_target.startswith(("http://", "https://", "#")):
                continue
            links.append(link_target)

        # Return deduplicated list preserving order
        return list(dict.fromkeys(links))

    def _merge_chunks_by_threshold(self, chunks: list[Chunk], threshold: int) -> list[Chunk]:
        """Merge chunks below token threshold with adjacent chunks."""
        if not chunks:
            return chunks

        merged: list[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            current_tokens = self._count_tokens(current.chunk_text)

            if current_tokens < threshold and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                merged_text = current.chunk_text + "\n\n" + next_chunk.chunk_text
                next_chunk.chunk_text = merged_text
                # Merge links from both chunks (deduplicated)
                combined_links = current.links + next_chunk.links
                next_chunk.links = list(dict.fromkeys(combined_links))
                i += 1
            else:
                merged.append(current)
                i += 1

        return merged

    def _merge_tiny_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return self._merge_chunks_by_threshold(chunks, MIN_TOKENS)

    def _merge_header_only_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return self._merge_chunks_by_threshold(chunks, 10)
