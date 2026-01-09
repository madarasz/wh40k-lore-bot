"""Unit tests for OpenAI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError
from pydantic import ValidationError as PydanticValidationError

from src.llm.base_provider import GenerationOptions
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.structured_output import LLMStructuredResponse
from src.utils.exceptions import ConfigurationError


class TestOpenAIProvider:
    """Test OpenAIProvider implementation."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        """Create OpenAI provider instance."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            return OpenAIProvider()

    @pytest.fixture
    def generation_options(self) -> GenerationOptions:
        """Create generation options."""
        return GenerationOptions(model="gpt-4.1")

    async def test_provider_name(self, provider: OpenAIProvider) -> None:
        """Test provider name is correct."""
        assert provider.get_provider_name() == "openai"

    async def test_supported_model(self, provider: OpenAIProvider) -> None:
        """Test supported model constant."""
        assert provider.SUPPORTED_MODEL == "gpt-4.1"

    async def test_generate_success(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.generate("Test prompt", generation_options)

            assert result.text == "Test response"
            assert result.provider == "openai"
            assert result.model == "gpt-4.1"
            assert result.tokens_prompt == 100
            assert result.tokens_completion == 50
            assert result.cost_usd > 0

    async def test_generate_uses_default_model(
        self,
        provider: OpenAIProvider,
    ) -> None:
        """Test generation uses default model when not specified."""
        options = GenerationOptions(model="")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ) as mock_create:
            result = await provider.generate("Test prompt", options)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "gpt-4.1"
            assert result.model == "gpt-4.1"

    async def test_generate_rate_limit_error(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test rate limit error is propagated."""
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        error = RateLimitError(message="Rate limit", response=mock_response, body=None)

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(RateLimitError):
                await provider.generate("Test prompt", generation_options)

    async def test_generate_authentication_error(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test authentication error is propagated."""
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        error = AuthenticationError(message="Auth error", response=mock_response, body=None)

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(AuthenticationError):
                await provider.generate("Test prompt", generation_options)

    async def test_generate_generic_error_wrapped(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test generic errors are wrapped in LLMProviderError."""
        from src.utils.exceptions import LLMProviderError

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=Exception("Unknown error")),
        ):
            with pytest.raises(LLMProviderError, match="OpenAI generation failed"):
                await provider.generate("Test prompt", generation_options)

    async def test_generate_structured_success(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test successful structured generation."""
        mock_parsed = LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test personality",
            sources=["https://wh40k.fandom.com/wiki/Test"],
            smalltalk=False,
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(parsed=mock_parsed))]

        with patch.object(
            provider.client.beta.chat.completions,
            "parse",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await provider.generate_structured(
                "Test prompt", generation_options, LLMStructuredResponse
            )

            assert result.answer == "Test answer"
            assert result.personality_reply == "Test personality"
            assert result.smalltalk is False

    async def test_generate_structured_fallback_client_side(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test fallback to client-side validation."""
        # Simulate beta API not available
        with patch.object(
            provider.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=AttributeError("Beta API not available")),
        ):
            mock_response = MagicMock()
            json_content = (
                '{"answer": "Test", "personality_reply": "Reply", '
                '"sources": ["https://wh40k.fandom.com/wiki/Test"], "smalltalk": false}'
            )
            mock_response.choices = [MagicMock(message=MagicMock(content=json_content))]

            with patch.object(
                provider.client.chat.completions,
                "create",
                new=AsyncMock(return_value=mock_response),
            ):
                result = await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

                assert result.answer == "Test"
                assert result.personality_reply == "Reply"

    async def test_generate_structured_validation_error_fails_fast(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test validation error fails immediately without retry."""
        with patch.object(
            provider.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=PydanticValidationError.from_exception_data("test", [])),
        ):
            with pytest.raises(PydanticValidationError):
                await provider.generate_structured(
                    "Test prompt", generation_options, LLMStructuredResponse
                )

    async def test_retry_logic_succeeds_on_second_attempt(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test retry logic succeeds on second attempt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        # Create proper exception instance
        mock_request = MagicMock()
        error = APIConnectionError(request=mock_request)

        # Fail first, succeed second
        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(
                side_effect=[
                    error,
                    mock_response,
                ]
            ),
        ):
            result = await provider.generate("Test prompt", generation_options)
            assert result.text == "Response"

    async def test_retry_logic_exhausts_after_three_attempts(
        self,
        provider: OpenAIProvider,
        generation_options: GenerationOptions,
    ) -> None:
        """Test retry logic exhausts after 3 attempts and wraps error."""
        from src.utils.exceptions import LLMProviderError

        # Create proper exception instance
        mock_request = MagicMock()
        error = APIConnectionError(request=mock_request)

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=error),
        ):
            # After all retries, APIConnectionError is wrapped in LLMProviderError
            with pytest.raises(LLMProviderError, match="OpenAI generation failed"):
                await provider.generate("Test prompt", generation_options)

    def test_initialization_without_api_key_fails(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
                OpenAIProvider()
