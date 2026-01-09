"""Multi-LLM router for provider selection and request routing."""

import os

import structlog
from pydantic import BaseModel

from src.llm.base_provider import GenerationOptions, LLMProvider, LLMResponse
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.structured_output import LLMStructuredResponse
from src.utils.exceptions import LLMProviderError

logger = structlog.get_logger(__name__)


class MultiLLMRouter:
    """Multi-LLM router with model-based provider selection.

    Manages provider registry and auto-detects provider from model name.
    No fallback logic - fail fast if provider fails.

    Model-to-provider mapping:
    - claude-* → AnthropicProvider
    - gpt-* → OpenAIProvider
    """

    # Default model if none specified
    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(self) -> None:
        """Initialize router with provider registry."""
        self.providers: dict[str, LLMProvider] = {}
        self.default_model = os.getenv("LLM_DEFAULT_MODEL", self.DEFAULT_MODEL)

        # Validate default model at initialization (fail fast)
        if not self.default_model.startswith(("claude-", "gpt-")):
            raise LLMProviderError(
                f"Invalid LLM_DEFAULT_MODEL: {self.default_model}. "
                "Model name must start with 'claude-' or 'gpt-'"
            )

        # Initialize providers
        self._register_provider("openai", OpenAIProvider())
        self._register_provider("anthropic", AnthropicProvider())

        logger.info(
            "llm_router_initialized",
            providers=list(self.providers.keys()),
            default_model=self.default_model,
        )

    def _register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register a provider in the registry.

        Args:
            name: Provider identifier (e.g., "openai", "anthropic")
            provider: LLMProvider implementation instance
        """
        self.providers[name] = provider
        logger.debug("provider_registered", provider_name=name)

    def _get_provider_for_model(self, model: str) -> LLMProvider:
        """Auto-detect provider from model name.

        Args:
            model: Model identifier (e.g., "gpt-4.1", "claude-sonnet-4-5")

        Returns:
            LLMProvider instance for the model

        Raises:
            LLMProviderError: If model prefix not recognized
        """
        if model.startswith("claude-"):
            return self.providers["anthropic"]
        elif model.startswith("gpt-"):
            return self.providers["openai"]
        else:
            raise LLMProviderError(
                f"Unknown model: {model}. Model name must start with 'claude-' or 'gpt-'"
            )

    async def generate(
        self,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        """Generate unstructured text response.

        Args:
            prompt: User prompt to generate response for
            options: Optional generation options (uses defaults if not specified)

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            LLMProviderError: If generation fails or model not recognized
            RateLimitError: If rate limit exceeded
            AuthenticationError: If API key invalid
        """
        if options is None:
            options = GenerationOptions(model=self.default_model)

        model = options.model or self.default_model
        selected_provider = self._get_provider_for_model(model)

        logger.info(
            "routing_generation_request",
            provider=selected_provider.get_provider_name(),
            model=model,
            prompt_length=len(prompt),
        )

        return await selected_provider.generate(prompt, options)

    async def generate_structured(
        self,
        prompt: str,
        options: GenerationOptions | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMStructuredResponse:
        """Generate structured JSON response with Pydantic validation.

        Args:
            prompt: User prompt to generate response for
            options: Optional generation options (uses defaults if not specified)
            response_schema: Pydantic model class (uses LLMStructuredResponse if not specified)

        Returns:
            Validated Pydantic model instance (typically LLMStructuredResponse)

        Raises:
            LLMProviderError: If generation fails or model not recognized
            RateLimitError: If rate limit exceeded
            AuthenticationError: If API key invalid
            ValidationError: If LLM returned invalid JSON schema
        """
        if options is None:
            options = GenerationOptions(model=self.default_model)

        model = options.model or self.default_model
        selected_provider = self._get_provider_for_model(model)

        if response_schema is None:
            response_schema = LLMStructuredResponse

        logger.info(
            "routing_structured_generation_request",
            provider=selected_provider.get_provider_name(),
            model=model,
            prompt_length=len(prompt),
            response_schema=response_schema.__name__,
        )

        return await selected_provider.generate_structured(prompt, options, response_schema)
