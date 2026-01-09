"""Unit tests for ResponseFormatter."""

import pytest
from pydantic import HttpUrl

from src.llm.response_formatter import ResponseFormatter
from src.llm.structured_output import LLMStructuredResponse


@pytest.fixture
def formatter() -> ResponseFormatter:
    """Create a ResponseFormatter instance."""
    return ResponseFormatter()


@pytest.fixture
def lore_response() -> LLMStructuredResponse:
    """Create a sample lore question response (smalltalk=false)."""
    return LLMStructuredResponse(
        answer="Roboute Guilliman a Birodalom egyik legerősebb harcosa.",
        personality_reply="Az Császár védelme legyen veled, halandó.",
        sources=[
            HttpUrl("https://warhammer40k.fandom.com/wiki/Roboute_Guilliman"),
            HttpUrl("https://warhammer40k.fandom.com/wiki/Ultramarines"),
        ],
        smalltalk=False,
    )


@pytest.fixture
def smalltalk_response() -> LLMStructuredResponse:
    """Create a sample smalltalk response (smalltalk=true)."""
    return LLMStructuredResponse(
        answer=None,
        personality_reply="Üdvözöllek a 41. évezredben! Készen állok válaszolni.",
        sources=None,
        smalltalk=True,
    )


class TestFormatCliResponseLoreQuestions:
    """Tests for format_cli_response with lore questions."""

    def test_lore_response_contains_answer(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that lore response includes the answer."""
        result = formatter.format_cli_response(lore_response)

        assert lore_response.answer in result

    def test_lore_response_contains_personality_reply(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that lore response includes personality reply."""
        result = formatter.format_cli_response(lore_response)

        assert lore_response.personality_reply in result

    def test_lore_response_contains_sources(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that lore response includes sources."""
        result = formatter.format_cli_response(lore_response)

        assert "https://warhammer40k.fandom.com/wiki/Roboute_Guilliman" in result
        assert "https://warhammer40k.fandom.com/wiki/Ultramarines" in result

    def test_lore_response_format_structure(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that lore response has correct structure: answer, personality, sources."""
        result = formatter.format_cli_response(lore_response)

        answer_pos = result.find(str(lore_response.answer))
        personality_pos = result.find(lore_response.personality_reply)
        sources_pos = result.find("📚 Források:")

        # Answer comes first, then personality reply, then sources
        assert answer_pos < personality_pos < sources_pos


class TestFormatCliResponseSmalltalk:
    """Tests for format_cli_response with smalltalk."""

    def test_smalltalk_returns_only_personality_reply(
        self, formatter: ResponseFormatter, smalltalk_response: LLMStructuredResponse
    ) -> None:
        """Test that smalltalk returns only personality reply."""
        result = formatter.format_cli_response(smalltalk_response)

        assert result == smalltalk_response.personality_reply

    def test_smalltalk_no_sources_header(
        self, formatter: ResponseFormatter, smalltalk_response: LLMStructuredResponse
    ) -> None:
        """Test that smalltalk response has no sources header."""
        result = formatter.format_cli_response(smalltalk_response)

        assert "📚 Források:" not in result
        assert "📚 Sources:" not in result


class TestSourceUrlFormatting:
    """Tests for source URL formatting."""

    def test_full_urls_displayed(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that full wiki URLs are displayed."""
        result = formatter.format_cli_response(lore_response)

        # URLs should be displayed in full, not as article titles
        assert "https://warhammer40k.fandom.com/wiki/Roboute_Guilliman" in result
        assert "- https://warhammer40k.fandom.com/wiki/" in result

    def test_sources_formatted_as_bullet_list(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that sources are formatted as bullet list."""
        result = formatter.format_cli_response(lore_response)

        assert "- https://warhammer40k.fandom.com/wiki/Roboute_Guilliman" in result
        assert "- https://warhammer40k.fandom.com/wiki/Ultramarines" in result


class TestSourceLimitEnforcement:
    """Tests for source limit enforcement."""

    def test_max_five_sources_displayed(self, formatter: ResponseFormatter) -> None:
        """Test that only top 5 sources are displayed when >5 provided."""
        response = LLMStructuredResponse(
            answer="Answer text here.",
            personality_reply="Personality reply here.",
            sources=[
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source1"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source2"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source3"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source4"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source5"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source6"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source7"),
            ],
            smalltalk=False,
        )

        result = formatter.format_cli_response(response)

        # First 5 sources should be present
        assert "Source1" in result
        assert "Source2" in result
        assert "Source3" in result
        assert "Source4" in result
        assert "Source5" in result

        # 6th and 7th sources should be truncated
        assert "Source6" not in result
        assert "Source7" not in result

    def test_exactly_five_sources_all_displayed(self, formatter: ResponseFormatter) -> None:
        """Test that exactly 5 sources are all displayed."""
        response = LLMStructuredResponse(
            answer="Answer text here.",
            personality_reply="Personality reply here.",
            sources=[
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source1"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source2"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source3"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source4"),
                HttpUrl("https://warhammer40k.fandom.com/wiki/Source5"),
            ],
            smalltalk=False,
        )

        result = formatter.format_cli_response(response)

        assert "Source1" in result
        assert "Source2" in result
        assert "Source3" in result
        assert "Source4" in result
        assert "Source5" in result


class TestLanguageSupport:
    """Tests for Hungarian and English language support."""

    def test_hungarian_header(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test Hungarian sources header."""
        result = formatter.format_cli_response(lore_response, language="hu")

        assert "📚 Források:" in result
        assert "📚 Sources:" not in result

    def test_english_header(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test English sources header."""
        result = formatter.format_cli_response(lore_response, language="en")

        assert "📚 Sources:" in result
        assert "📚 Források:" not in result

    def test_default_language_is_hungarian(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that default language is Hungarian."""
        result = formatter.format_cli_response(lore_response)

        assert "📚 Források:" in result

    def test_unknown_language_falls_back_to_english(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that unknown language falls back to English."""
        result = formatter.format_cli_response(lore_response, language="de")

        assert "📚 Sources:" in result


class TestEmptySourcesEdgeCase:
    """Tests for empty sources edge case."""

    def test_smalltalk_with_empty_sources(self, formatter: ResponseFormatter) -> None:
        """Test smalltalk response handles empty/None sources correctly."""
        response = LLMStructuredResponse(
            answer=None,
            personality_reply="Hello there!",
            sources=None,
            smalltalk=True,
        )

        result = formatter.format_cli_response(response)

        assert result == "Hello there!"
        assert "📚" not in result


class TestFormatDiscordResponse:
    """Tests for format_discord_response stub."""

    def test_discord_response_returns_cli_format(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that Discord response currently returns CLI format."""
        cli_result = formatter.format_cli_response(lore_response)
        discord_result = formatter.format_discord_response(lore_response)

        assert cli_result == discord_result

    def test_discord_response_respects_language(
        self, formatter: ResponseFormatter, lore_response: LLMStructuredResponse
    ) -> None:
        """Test that Discord response respects language parameter."""
        result_hu = formatter.format_discord_response(lore_response, language="hu")
        result_en = formatter.format_discord_response(lore_response, language="en")

        assert "📚 Források:" in result_hu
        assert "📚 Sources:" in result_en


class TestResponseFormatterConstants:
    """Tests for ResponseFormatter constants."""

    def test_headers_dict(self, formatter: ResponseFormatter) -> None:
        """Test that headers dict contains expected languages."""
        assert "hu" in formatter.HEADERS
        assert "en" in formatter.HEADERS
        assert formatter.HEADERS["hu"] == "📚 Források:"
        assert formatter.HEADERS["en"] == "📚 Sources:"

    def test_max_sources_constant(self, formatter: ResponseFormatter) -> None:
        """Test that MAX_SOURCES is set to 5."""
        assert formatter.MAX_SOURCES == 5
