"""Unit tests for Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import APIConnectionError, AuthenticationError, RateLimitError
from pydantic import ValidationError as PydanticValidationError

from src.llm.base_provider import GenerationOptions
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.structured_output import LLMStructuredResponse
from src.utils.exceptions import ConfigurationError, LLMProviderError


class TestAnthropicProvider:
    """Test AnthropicProvider implementation."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        """Create Anthropic provider instance."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return AnthropicProvider()

    @pytest.fixture
    def generation_options(self) -> GenerationOptions:
        """Create generation options."""
        return GenerationOptions(model="claude-sonnet-4-5")

    async def test_provider_name(self, provider: AnthropicProvider) -> None:
        """Test provider name is correct."""
        assert provider.get_provider_name() == "anthropic"

    async def test_supported_models(self, provider: AnthropicProvider) -> None:
        """Test supported models list."""
        assert provider.SUPPORTED_MODELS == [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        ]

    async def test_default_model(self, provider: AnthropicProvider) -> None:
        """Test default model."""
        assert provider.DEFAULT_MODEL == "claude-sonnet-4-5"

    async def test_generate_structured_success(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test successful structured generation."""
        mock_parsed = LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test personality",
            sources=["https://wh40k.fandom.com/wiki/Test"],
            smalltalk=False,
            language="EN",
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_parsed

        with patch.object(
            provider.client.beta.messages, "parse", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.generate_structured(
                "Test prompt", generation_options, LLMStructuredResponse
            )

            assert result.answer == "Test answer"
            assert result.personality_reply == "Test personality"
            assert result.smalltalk is False

    async def test_generate_structured_uses_default_model(
        self,
        provider: AnthropicProvider,
    ) -> None:
        """Test structured generation uses default model when not specified."""
        options = GenerationOptions(model="")
        mock_parsed = LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test personality",
            sources=["https://wh40k.fandom.com/wiki/Test"],
            smalltalk=False,
            language="EN",
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_parsed

        with patch.object(
            provider.client.beta.messages, "parse", new=AsyncMock(return_value=mock_response)
        ) as mock_parse:
            await provider.generate_structured("Test prompt", options, LLMStructuredResponse)

            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-5"

    @pytest.mark.long
    async def test_generate_structured_rate_limit_error(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test rate limit error is propagated."""
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        error = RateLimitError(message="Rate limit", response=mock_response, body=None)

        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(RateLimitError):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    async def test_generate_structured_authentication_error(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test authentication error is propagated."""
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        error = AuthenticationError(message="Invalid key", response=mock_response, body=None)

        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(AuthenticationError):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    async def test_generate_structured_generic_error_wrapped(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test generic errors are wrapped in LLMProviderError."""
        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(side_effect=Exception("Unknown error")),
        ):
            with pytest.raises(LLMProviderError, match="Anthropic structured generation failed"):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    async def test_generate_structured_validation_error_fails_fast(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test validation error fails immediately without retry."""
        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(side_effect=PydanticValidationError.from_exception_data("test", [])),
        ):
            with pytest.raises(PydanticValidationError):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    @pytest.mark.long
    async def test_retry_logic_succeeds_on_second_attempt(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test retry logic succeeds on second attempt."""
        mock_parsed = LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test personality",
            sources=["https://wh40k.fandom.com/wiki/Test"],
            smalltalk=False,
            language="EN",
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_parsed

        # Create connection error with required request argument
        mock_request = MagicMock()
        error = APIConnectionError(message="Connection error", request=mock_request)

        # Fail first, succeed second
        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(
                side_effect=[
                    error,
                    mock_response,
                ]
            ),
        ):
            result = await provider.generate_structured(
                "Test prompt", generation_options, LLMStructuredResponse
            )
            assert result.answer == "Test answer"

    @pytest.mark.long
    async def test_retry_logic_exhausts_after_three_attempts(
        self,
        provider: AnthropicProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test retry logic exhausts after 3 attempts."""
        # Create connection error with required request argument
        mock_request = MagicMock()
        error = APIConnectionError(message="Connection error", request=mock_request)

        with patch.object(
            provider.client.beta.messages,
            "parse",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(LLMProviderError, match="Anthropic structured generation failed"):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    def test_initialization_without_api_key_fails(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()
