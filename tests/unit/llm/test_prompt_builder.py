"""Unit tests for PromptBuilder."""

from pathlib import Path

import pytest

from src.llm.prompt_builder import PromptBuilder, prompt_builder
from src.utils.exceptions import ConfigurationError


class TestPromptBuilder:
    """Test PromptBuilder implementation."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create PromptBuilder instance."""
        return PromptBuilder()

    def test_initialization(self, builder: PromptBuilder) -> None:
        """Test builder initializes with prompts directory."""
        assert builder.prompts_dir.exists()
        assert (builder.prompts_dir / "system.md").exists()
        assert (builder.prompts_dir / "user.md").exists()
        assert (builder.prompts_dir / "persona-default.md").exists()

    def test_initialization_missing_directory_fails(self, tmp_path: Path) -> None:
        """Test initialization fails with missing directory."""
        with pytest.raises(ConfigurationError, match="Prompts directory not found"):
            PromptBuilder(prompts_dir=tmp_path / "nonexistent")

    def test_load_persona_default(self, builder: PromptBuilder) -> None:
        """Test loading default persona template."""
        persona = builder.load_persona()
        assert isinstance(persona, str)
        assert len(persona) > 0
        # Persona should contain some text about Imperial scholar or WH40K
        assert any(
            word in persona.lower()
            for word in ["imperial", "scholar", "emperor", "warhammer", "gravitas"]
        )

    def test_load_persona_grimdark(self, builder: PromptBuilder) -> None:
        """Test loading grimdark persona template."""
        persona = builder.load_persona("grimdark")
        assert isinstance(persona, str)
        assert len(persona) > 0
        # Grimdark persona should contain dramatic language
        assert any(
            word in persona.lower()
            for word in ["grimdark", "narrator", "dramatic", "darkness", "war"]
        )

    def test_build_system_prompt_contains_persona(self, builder: PromptBuilder) -> None:
        """Test building system prompt includes persona."""
        prompt = builder.build_system_prompt()
        assert isinstance(prompt, str)
        # Should contain persona content
        assert "Imperial" in prompt or "scholar" in prompt.lower()

    def test_build_system_prompt_contains_language_detection(self, builder: PromptBuilder) -> None:
        """Test system prompt includes language detection instructions."""
        prompt = builder.build_system_prompt()
        assert "Language Detection" in prompt
        assert "Hungarian" in prompt
        assert "English" in prompt

    def test_build_system_prompt_contains_json_format(self, builder: PromptBuilder) -> None:
        """Test system prompt contains JSON response format."""
        prompt = builder.build_system_prompt()
        assert '"language": "HU" or "EN"' in prompt

    def test_build_user_prompt(self, builder: PromptBuilder) -> None:
        """Test building user prompt."""
        chunks = "Chunk 1: Some context\nChunk 2: More context"
        question = "What is the Emperor?"

        prompt = builder.build_user_prompt(chunks=chunks, question=question)

        assert isinstance(prompt, str)
        assert "Some context" in prompt
        assert "What is the Emperor?" in prompt

    def test_build_user_prompt_placeholders(self, builder: PromptBuilder) -> None:
        """Test user prompt placeholders are replaced."""
        prompt = builder.build_user_prompt(
            chunks="TEST_CHUNKS_CONTENT",
            question="TEST_QUESTION_CONTENT",
        )
        assert "TEST_CHUNKS_CONTENT" in prompt
        assert "TEST_QUESTION_CONTENT" in prompt
        assert "{chunks}" not in prompt
        assert "{question}" not in prompt

    def test_template_caching(self, builder: PromptBuilder) -> None:
        """Test templates are cached."""
        # First load
        builder.load_persona()
        assert "persona-default.md" in builder._template_cache

        # Cache should be used on second load
        cached_content = builder._template_cache["persona-default.md"]
        persona2 = builder.load_persona()
        assert persona2.strip() == cached_content.strip()

    def test_clear_cache(self, builder: PromptBuilder) -> None:
        """Test cache clearing."""
        builder.load_persona()
        assert len(builder._template_cache) > 0

        builder.clear_cache()
        assert len(builder._template_cache) == 0

    def test_missing_template_fails(self, builder: PromptBuilder) -> None:
        """Test loading missing template fails."""
        with pytest.raises(ConfigurationError, match="Template file not found"):
            builder._load_template("nonexistent.md")

    def test_singleton_instance(self) -> None:
        """Test singleton instance is available."""
        assert prompt_builder is not None
        assert isinstance(prompt_builder, PromptBuilder)
