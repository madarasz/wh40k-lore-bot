"""Unit tests for structured output Pydantic models."""

import pytest
from pydantic import ValidationError

from src.llm.structured_output import LLMStructuredResponse


class TestLLMStructuredResponse:
    """Test LLMStructuredResponse validation."""

    def test_valid_lore_response(self) -> None:
        """Test valid lore response with answer and sources."""
        response = LLMStructuredResponse(
            answer="Guilliman is the Primarch of the Ultramarines.",
            personality_reply="By the Emperor's grace, Guilliman leads the Ultramarines!",
            sources=["https://wh40k.fandom.com/wiki/Roboute_Guilliman"],
            smalltalk=False,
            language="EN",
        )
        assert response.answer == "Guilliman is the Primarch of the Ultramarines."
        assert response.smalltalk is False
        assert len(response.sources) == 1
        assert response.language == "EN"

    def test_valid_smalltalk_response(self) -> None:
        """Test valid smalltalk response without answer or sources."""
        response = LLMStructuredResponse(
            personality_reply="The Emperor protects, citizen.",
            smalltalk=True,
            language="EN",
        )
        assert response.answer is None
        assert response.sources is None
        assert response.smalltalk is True
        assert response.language == "EN"

    def test_smalltalk_with_optional_answer(self) -> None:
        """Test smalltalk can have optional answer and sources."""
        response = LLMStructuredResponse(
            answer="Just chatting!",
            personality_reply="Indeed, conversation is welcome.",
            sources=["https://wh40k.fandom.com/wiki/Example"],
            smalltalk=True,
            language="EN",
        )
        assert response.answer == "Just chatting!"
        assert response.smalltalk is True

    def test_hungarian_language(self) -> None:
        """Test response with Hungarian language."""
        response = LLMStructuredResponse(
            answer="Guilliman az Ultramarines primarcha.",
            personality_reply="A Császár véd.",
            sources=["https://wh40k.fandom.com/wiki/Roboute_Guilliman"],
            smalltalk=False,
            language="HU",
        )
        assert response.language == "HU"

    def test_missing_answer_for_lore_fails(self) -> None:
        """Test that lore query without answer raises validation error."""
        with pytest.raises(ValidationError, match="answer required when smalltalk=false"):
            LLMStructuredResponse(
                personality_reply="By the Emperor!",
                sources=["https://wh40k.fandom.com/wiki/Example"],
                smalltalk=False,
                language="EN",
            )

    def test_missing_sources_for_lore_fails(self) -> None:
        """Test that lore query without sources raises validation error."""
        with pytest.raises(ValidationError, match="sources required when smalltalk=false"):
            LLMStructuredResponse(
                answer="Guilliman is a Primarch.",
                personality_reply="By the Emperor!",
                smalltalk=False,
                language="EN",
            )

    def test_missing_personality_reply_fails(self) -> None:
        """Test that missing personality_reply raises validation error."""
        with pytest.raises(ValidationError, match="Field required"):
            LLMStructuredResponse(  # type: ignore[call-arg]
                answer="Test answer",
                sources=["https://wh40k.fandom.com/wiki/Example"],
                smalltalk=False,
                language="EN",
            )

    def test_missing_language_fails(self) -> None:
        """Test that missing language raises validation error."""
        with pytest.raises(ValidationError, match="Field required"):
            LLMStructuredResponse(  # type: ignore[call-arg]
                answer="Test answer",
                personality_reply="Test reply",
                sources=["https://wh40k.fandom.com/wiki/Example"],
                smalltalk=False,
            )

    def test_invalid_language_fails(self) -> None:
        """Test that invalid language value raises validation error."""
        with pytest.raises(ValidationError, match="Input should be 'HU' or 'EN'"):
            LLMStructuredResponse(
                answer="Test answer",
                personality_reply="Test reply",
                sources=["https://wh40k.fandom.com/wiki/Example"],
                smalltalk=False,
                language="DE",  # type: ignore[arg-type]
            )

    def test_invalid_url_format_fails(self) -> None:
        """Test that invalid URL format raises validation error."""
        with pytest.raises(ValidationError, match="Input should be a valid URL"):
            LLMStructuredResponse(
                answer="Test answer",
                personality_reply="Test reply",
                sources=["not-a-valid-url"],  # type: ignore[list-item]
                smalltalk=False,
                language="EN",
            )

    def test_json_schema_generation(self) -> None:
        """Test that JSON schema can be generated for LLM providers."""
        schema = LLMStructuredResponse.model_json_schema_for_llm()
        assert "properties" in schema
        assert "answer" in schema["properties"]
        assert "personality_reply" in schema["properties"]
        assert "sources" in schema["properties"]
        assert "smalltalk" in schema["properties"]
        assert "language" in schema["properties"]
        assert "required" in schema
        assert "personality_reply" in schema["required"]
        assert "smalltalk" in schema["required"]
        assert "language" in schema["required"]
