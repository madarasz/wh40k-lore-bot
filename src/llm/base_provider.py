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
        system_prompt: Optional system prompt for context/instructions
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 800
    system_prompt: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Anthropic, Gemini, etc.) must implement
    this interface to ensure consistent behavior across the system.
    """

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
