"""Base provider and data classes for LLM integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from src.llm.structured_output import LLMStructuredResponse


@dataclass
class GenerationOptions:
    """Options for LLM text generation.

    Args:
        model: Model identifier (e.g., "gpt-4.1", "claude-sonnet-4-5")
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
        max_tokens: Maximum tokens to generate (default: 800, increased for structured output)
        response_language: Language code for response (default: "hu" for Hungarian)
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 800
    response_language: str = "hu"


@dataclass
class LLMResponse:
    """Response from LLM text generation (backward compatibility).

    This is used for unstructured text generation. For structured output,
    use LLMStructuredResponse from structured_output.py instead.

    Args:
        text: Generated response text
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model used for generation
        tokens_prompt: Number of tokens in the prompt
        tokens_completion: Number of tokens in the completion
        cost_usd: Estimated cost in USD
        latency_ms: Response latency in milliseconds
    """

    text: str
    provider: str
    model: str
    tokens_prompt: int
    tokens_completion: int
    cost_usd: float
    latency_ms: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, Gemini, etc.) must implement
    this interface to ensure consistent behavior across the system.
    """

    @abstractmethod
    async def generate(self, prompt: str, options: GenerationOptions) -> LLMResponse:
        """Generate unstructured text response.

        Args:
            prompt: User prompt to generate response for
            options: Generation configuration options

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit exceeded
            AuthenticationError: If API key invalid
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
    ) -> LLMStructuredResponse:
        """Generate structured JSON response with Pydantic validation.

        Args:
            prompt: User prompt to generate response for
            options: Generation configuration options
            response_schema: Pydantic model class for response validation

        Returns:
            Validated Pydantic model instance (typically LLMStructuredResponse)

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit exceeded
            AuthenticationError: If API key invalid
            ValidationError: If LLM returned invalid JSON schema
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider identifier.

        Returns:
            Provider name (e.g., "openai", "anthropic", "gemini")
        """
        pass
