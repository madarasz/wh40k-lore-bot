"""Unit tests for Multi-LLM router."""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm.base_provider import GenerationOptions, LLMResponse
from src.llm.llm_router import MultiLLMRouter
from src.llm.structured_output import LLMStructuredResponse
from src.utils.exceptions import LLMProviderError


class TestMultiLLMRouter:
    """Test MultiLLMRouter implementation."""

    @pytest.fixture
    def router(self) -> MultiLLMRouter:
        """Create router with mocked providers."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"},
        ):
            return MultiLLMRouter()

    @pytest.fixture
    def mock_llm_response(self) -> LLMResponse:
        """Create mock LLM response."""
        return LLMResponse(
            text="Test response",
            provider="test",
            model="test-model",
            tokens_prompt=100,
            tokens_completion=50,
            cost_usd=0.01,
            latency_ms=100,
        )

    @pytest.fixture
    def mock_structured_response(self) -> LLMStructuredResponse:
        """Create mock structured response."""
        return LLMStructuredResponse(
            answer="Test answer",
            personality_reply="Test personality",
            sources=["https://wh40k.fandom.com/wiki/Test"],
            smalltalk=False,
        )

    def test_router_initialization(self, router: MultiLLMRouter) -> None:
        """Test router initializes with both providers."""
        assert "openai" in router.providers
        assert "anthropic" in router.providers
        assert router.default_model == "claude-sonnet-4-5"

    def test_router_initialization_with_custom_default(self) -> None:
        """Test router initialization with custom default model."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "ANTHROPIC_API_KEY": "test-key",
                "LLM_DEFAULT_MODEL": "gpt-4.1",
            },
        ):
            router = MultiLLMRouter()
            assert router.default_model == "gpt-4.1"

    def test_get_provider_for_claude_model(self, router: MultiLLMRouter) -> None:
        """Test provider selection for Claude models."""
        provider = router._get_provider_for_model("claude-sonnet-4-5")
        assert provider.get_provider_name() == "anthropic"

        provider = router._get_provider_for_model("claude-haiku-4-5")
        assert provider.get_provider_name() == "anthropic"

    def test_get_provider_for_gpt_model(self, router: MultiLLMRouter) -> None:
        """Test provider selection for GPT models."""
        provider = router._get_provider_for_model("gpt-4.1")
        assert provider.get_provider_name() == "openai"

    def test_get_provider_unknown_model_fails(self, router: MultiLLMRouter) -> None:
        """Test unknown model raises error."""
        with pytest.raises(LLMProviderError, match="Unknown model"):
            router._get_provider_for_model("unknown-model")

    async def test_generate_routes_to_anthropic(
        self,
        router: MultiLLMRouter,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test generate routes to Anthropic for Claude models."""
        options = GenerationOptions(model="claude-sonnet-4-5")

        with patch.object(
            router.providers["anthropic"],
            "generate",
            new=AsyncMock(return_value=mock_llm_response),
        ) as mock_generate:
            await router.generate("Test prompt", options)
            mock_generate.assert_called_once()

    async def test_generate_routes_to_openai(
        self,
        router: MultiLLMRouter,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test generate routes to OpenAI for GPT models."""
        options = GenerationOptions(model="gpt-4.1")

        with patch.object(
            router.providers["openai"],
            "generate",
            new=AsyncMock(return_value=mock_llm_response),
        ) as mock_generate:
            await router.generate("Test prompt", options)
            mock_generate.assert_called_once()

    async def test_generate_uses_default_model(
        self,
        router: MultiLLMRouter,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test generate uses default model when not specified."""
        with patch.object(
            router.providers["anthropic"],
            "generate",
            new=AsyncMock(return_value=mock_llm_response),
        ) as mock_generate:
            await router.generate("Test prompt")
            mock_generate.assert_called_once()
            # Check that options was created with default model
            call_args = mock_generate.call_args
            options = call_args[0][1]  # Second positional argument is options
            assert options.model == "claude-sonnet-4-5"

    async def test_generate_with_options(
        self,
        router: MultiLLMRouter,
        mock_llm_response: LLMResponse,
    ) -> None:
        """Test generate passes options correctly."""
        options = GenerationOptions(
            model="claude-haiku-4-5",
            temperature=0.5,
            max_tokens=500,
        )

        with patch.object(
            router.providers["anthropic"],
            "generate",
            new=AsyncMock(return_value=mock_llm_response),
        ) as mock_generate:
            await router.generate("Test prompt", options)
            mock_generate.assert_called_once_with("Test prompt", options)

    async def test_generate_unknown_model_fails(
        self,
        router: MultiLLMRouter,
    ) -> None:
        """Test generate fails for unknown model."""
        options = GenerationOptions(model="unknown-model")

        with pytest.raises(LLMProviderError, match="Unknown model"):
            await router.generate("Test prompt", options)

    async def test_generate_structured_routes_to_anthropic(
        self,
        router: MultiLLMRouter,
        mock_structured_response: LLMStructuredResponse,
    ) -> None:
        """Test generate_structured routes to Anthropic for Claude models."""
        options = GenerationOptions(model="claude-sonnet-4-5")

        with patch.object(
            router.providers["anthropic"],
            "generate_structured",
            new=AsyncMock(return_value=mock_structured_response),
        ) as mock_generate:
            await router.generate_structured("Test prompt", options)
            mock_generate.assert_called_once()

    async def test_generate_structured_routes_to_openai(
        self,
        router: MultiLLMRouter,
        mock_structured_response: LLMStructuredResponse,
    ) -> None:
        """Test generate_structured routes to OpenAI for GPT models."""
        options = GenerationOptions(model="gpt-4.1")

        with patch.object(
            router.providers["openai"],
            "generate_structured",
            new=AsyncMock(return_value=mock_structured_response),
        ) as mock_generate:
            await router.generate_structured("Test prompt", options)
            mock_generate.assert_called_once()

    async def test_generate_structured_uses_default_model(
        self,
        router: MultiLLMRouter,
        mock_structured_response: LLMStructuredResponse,
    ) -> None:
        """Test generate_structured uses default model."""
        with patch.object(
            router.providers["anthropic"],
            "generate_structured",
            new=AsyncMock(return_value=mock_structured_response),
        ) as mock_generate:
            await router.generate_structured("Test prompt")
            mock_generate.assert_called_once()

    async def test_generate_structured_unknown_model_fails(
        self,
        router: MultiLLMRouter,
    ) -> None:
        """Test generate_structured fails for unknown model."""
        options = GenerationOptions(model="unknown-model")

        with pytest.raises(LLMProviderError, match="Unknown model"):
            await router.generate_structured("Test prompt", options)
