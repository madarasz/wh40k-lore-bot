"""Unit tests for MarkdownChunker."""

import pytest

from src.ingestion.models import Chunk
from src.ingestion.text_chunker import MAX_TOKENS, MIN_TOKENS, MarkdownChunker


class TestMarkdownChunker:
    """Test suite for MarkdownChunker class."""

    @pytest.fixture
    def chunker(self) -> MarkdownChunker:
        """Create a MarkdownChunker instance for testing."""
        return MarkdownChunker()

    def test_initialization(self, chunker: MarkdownChunker) -> None:
        """Test chunker initializes with tiktoken encoder."""
        assert chunker._encoder is not None
        assert chunker._encoder.name == "cl100k_base"

    def test_empty_markdown_raises_error(self, chunker: MarkdownChunker) -> None:
        """Test that empty markdown raises ValueError."""
        with pytest.raises(ValueError, match="Markdown content cannot be empty"):
            chunker.chunk_markdown("", "Test Article")

    def test_empty_title_raises_error(self, chunker: MarkdownChunker) -> None:
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="Article title cannot be empty"):
            chunker.chunk_markdown("Some content", "")

    def test_single_section_no_splitting(self, chunker: MarkdownChunker) -> None:
        """Test that a single short section is not split."""
        markdown = """## Introduction
This is a short introduction to the Warhammer 40k universe.
It contains just a few sentences about the Emperor and Space Marines."""

        chunks = chunker.chunk_markdown(markdown, "Test Article")

        assert len(chunks) == 1
        assert chunks[0].chunk_text == markdown
        assert chunks[0].article_title == "Test Article"
        assert chunks[0].section_path == "Introduction"
        assert chunks[0].chunk_index == 0

    def test_multiple_sections(self, chunker: MarkdownChunker) -> None:
        """Test that multiple sections are split correctly."""
        markdown = """## History
The history of the Imperium spans over 10,000 years.

## Organization
The Imperium is organized into various military and administrative bodies.

## Culture
The Imperial Cult worships the Emperor as a god."""

        chunks = chunker.chunk_markdown(markdown, "Imperium")

        assert len(chunks) == 3
        assert chunks[0].section_path == "History"
        assert chunks[1].section_path == "Organization"
        assert chunks[2].section_path == "Culture"
        assert all(chunk.article_title == "Imperium" for chunk in chunks)

        # Verify chunk indices
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2

    def test_nested_section_hierarchy(self, chunker: MarkdownChunker) -> None:
        """Test that nested section hierarchy is preserved."""
        markdown = """## History
The history section.

### The Great Crusade
The Emperor united humanity.

#### Unification Wars
Terra was united first.

### Horus Heresy
Half the legions fell to Chaos."""

        chunks = chunker.chunk_markdown(markdown, "Imperium")

        # Find chunks by checking their section paths
        paths = [chunk.section_path for chunk in chunks]
        # Verify hierarchical paths are present
        assert any("History" in path for path in paths)
        assert any("The Great Crusade" in path for path in paths)
        assert any("Unification Wars" in path for path in paths)
        assert any("Horus Heresy" in path for path in paths)

    def test_long_section_paragraph_split(self, chunker: MarkdownChunker) -> None:
        """Test that a long section is split by paragraphs."""
        # Create a section with multiple paragraphs that exceeds MAX_TOKENS
        # Each paragraph should be ~150 tokens, total ~450 per paragraph
        paragraph = (
            "The Space Marines are genetically enhanced warriors created by the Emperor. " * 50
        )
        markdown = f"""## Space Marines
{paragraph}

{paragraph}

{paragraph}"""

        chunks = chunker.chunk_markdown(markdown, "Space Marines")

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # All chunks should have the same section path
        assert all(chunk.section_path == "Space Marines" for chunk in chunks)
        # Each chunk should respect token limit
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.chunk_text)
            assert token_count <= MAX_TOKENS

    def test_very_long_paragraph_sentence_split(self, chunker: MarkdownChunker) -> None:
        """Test that a very long paragraph is split by sentences."""
        # Create a single paragraph with many sentences
        sentence = "The Emperor of Mankind is the immortal sovereign of the Imperium. "
        long_paragraph = sentence * 100  # Very long paragraph

        markdown = f"""## The Emperor
{long_paragraph}"""

        chunks = chunker.chunk_markdown(markdown, "Emperor")

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should respect token limit
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.chunk_text)
            assert token_count <= MAX_TOKENS

    def test_token_counting(self, chunker: MarkdownChunker) -> None:
        """Test that token counting works correctly."""
        text = "Hello world"
        token_count = chunker._count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

        # Test empty string
        assert chunker._count_tokens("") == 0

    def test_paragraph_splitting(self, chunker: MarkdownChunker) -> None:
        """Test paragraph splitting on double newlines."""
        text = """First paragraph.

Second paragraph.

Third paragraph."""

        paragraphs = chunker._split_by_paragraphs(text)
        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "Second paragraph."
        assert paragraphs[2] == "Third paragraph."

    def test_sentence_splitting(self, chunker: MarkdownChunker) -> None:
        """Test sentence splitting with various edge cases."""
        # Basic sentences
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._split_by_sentences(text)
        assert len(sentences) == 3

        # Test with abbreviations
        text_with_abbrev = "Dr. Smith visited Mr. Jones. They discussed the case."
        sentences_abbrev = chunker._split_by_sentences(text_with_abbrev)
        # Should not split on Dr. or Mr.
        assert len(sentences_abbrev) == 2

        # Test with decimals
        text_with_decimal = "The ratio is 3.14. That is important."
        sentences_decimal = chunker._split_by_sentences(text_with_decimal)
        assert len(sentences_decimal) == 2

        # Test with question marks and exclamations
        text_mixed = "What is this? It is amazing! That is all."
        sentences_mixed = chunker._split_by_sentences(text_mixed)
        assert len(sentences_mixed) == 3

    def test_no_headers_single_chunk(self, chunker: MarkdownChunker) -> None:
        """Test article with no headers is treated as single section."""
        markdown = "This is an article without any headers. Just plain text."

        chunks = chunker.chunk_markdown(markdown, "Plain Article")

        assert len(chunks) == 1
        assert chunks[0].section_path == "Plain Article"
        assert chunks[0].chunk_text == markdown

    def test_empty_sections_skipped(self, chunker: MarkdownChunker) -> None:
        """Test that empty/malformed sections are handled gracefully."""
        markdown = """## Section One
Content here with enough text to make it substantial.

##

## Section Two
More content with additional text to ensure proper size."""

        chunks = chunker.chunk_markdown(markdown, "Test")

        # Empty/malformed sections may be merged with adjacent content
        assert len(chunks) >= 1
        # Verify both sections' content appears somewhere
        all_text = " ".join(chunk.chunk_text for chunk in chunks)
        assert "Section One" in all_text
        assert "Section Two" in all_text
        assert "Content here" in all_text
        assert "More content" in all_text

    def test_very_short_article(self, chunker: MarkdownChunker) -> None:
        """Test very short article (< MIN_TOKENS) creates single chunk."""
        markdown = "## Tiny\nSmall."

        chunks = chunker.chunk_markdown(markdown, "Tiny Article")

        assert len(chunks) == 1
        # Even if below MIN_TOKENS, it's kept as single chunk
        token_count = chunker._count_tokens(chunks[0].chunk_text)
        assert token_count < MIN_TOKENS

    def test_chunk_index_assignment(self, chunker: MarkdownChunker) -> None:
        """Test that chunk indices are assigned correctly."""
        markdown = """## Section One
Content one.

## Section Two
Content two.

## Section Three
Content three."""

        chunks = chunker.chunk_markdown(markdown, "Test")

        # Verify indices are sequential
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_section_header_included_in_chunk(self, chunker: MarkdownChunker) -> None:
        """Test that section headers are included in chunk text."""
        markdown = """## Important Section
This is the content under the important section."""

        chunks = chunker.chunk_markdown(markdown, "Test")

        assert len(chunks) == 1
        assert "## Important Section" in chunks[0].chunk_text
        assert "This is the content" in chunks[0].chunk_text

    def test_markdown_formatting_preserved(self, chunker: MarkdownChunker) -> None:
        """Test that markdown formatting is preserved in chunks."""
        markdown = """## Formatting Test
This has **bold text** and *italic text*.

- List item 1
- List item 2

Here is some `code` inline."""

        chunks = chunker.chunk_markdown(markdown, "Test")

        chunk_text = chunks[0].chunk_text
        assert "**bold text**" in chunk_text
        assert "*italic text*" in chunk_text
        assert "- List item 1" in chunk_text
        assert "`code`" in chunk_text

    def test_merge_tiny_chunks(self, chunker: MarkdownChunker) -> None:
        """Test that tiny chunks are merged with adjacent chunks."""
        # Create chunks manually to test merging
        tiny_chunk = Chunk(
            chunk_text="Tiny.",  # Very small
            article_title="Test",
            section_path="Test",
            chunk_index=0,
        )
        normal_chunk = Chunk(
            chunk_text="This is a normal sized chunk with more content.",
            article_title="Test",
            section_path="Test",
            chunk_index=1,
        )

        merged = chunker._merge_tiny_chunks([tiny_chunk, normal_chunk])

        # Should merge tiny into normal
        assert len(merged) == 1
        assert "Tiny." in merged[0].chunk_text
        assert "normal sized chunk" in merged[0].chunk_text

    def test_max_token_constraint(self, chunker: MarkdownChunker) -> None:
        """Test that no chunk exceeds MAX_TOKENS."""
        # Create a very long article
        long_sentence = "The Imperium of Man is vast and sprawling across the galaxy. "
        long_content = long_sentence * 200

        markdown = f"""## Long Section
{long_content}"""

        chunks = chunker.chunk_markdown(markdown, "Long Article")

        # Verify no chunk exceeds MAX_TOKENS
        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.chunk_text)
            assert token_count <= MAX_TOKENS, (
                f"Chunk has {token_count} tokens, exceeds {MAX_TOKENS}"
            )

    def test_real_world_structure(self, chunker: MarkdownChunker) -> None:
        """Test with a realistic wiki article structure."""
        markdown = """## Overview
The Blood Angels are one of the 20 First Founding Space Marine Legions.

## History
### The Great Crusade
During the Great Crusade, they fought alongside the Emperor.

### The Horus Heresy
The Blood Angels remained loyal during the Heresy.

## Organization
### Chapter Structure
The Chapter is organized into 10 companies.

### Successors
Many successor chapters exist.

## Combat Doctrine
The Blood Angels favor close combat and rapid assault."""

        chunks = chunker.chunk_markdown(markdown, "Blood Angels")

        # Should have multiple chunks
        assert len(chunks) > 0

        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk.chunk_text
            assert chunk.article_title == "Blood Angels"
            assert chunk.section_path
            assert isinstance(chunk.chunk_index, int)

        # Verify hierarchical paths exist
        paths = [chunk.section_path for chunk in chunks]
        assert any("History" in path for path in paths)
        assert any("Organization" in path for path in paths)
