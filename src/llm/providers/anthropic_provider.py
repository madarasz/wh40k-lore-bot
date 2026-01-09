"""Anthropic LLM provider implementation with structured output support."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

import structlog
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from src.llm.base_provider import GenerationOptions, LLMProvider
from src.llm.structured_output import LLMStructuredResponse
from src.utils.config import get_required_env
from src.utils.exceptions import LLMProviderError

logger = structlog.get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider with structured output support.

    Supports:
    - claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-5
    - Structured output via beta.messages.parse() with constrained decoding
    - Retry logic with exponential backoff
    """

    # Supported models
    SUPPORTED_MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"]
    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(self) -> None:
        """Initialize Anthropic provider with API key from environment."""
        api_key = get_required_env("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=api_key)

        logger.info(
            "anthropic_provider_initialized",
            supported_models=self.SUPPORTED_MODELS,
            default_model=self.DEFAULT_MODEL,
        )

    def get_provider_name(self) -> str:
        """Get provider identifier."""
        return "anthropic"

    async def generate_structured(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
    ) -> LLMStructuredResponse:
        """Generate structured JSON response with Pydantic validation.

        Uses beta.messages.parse() with constrained decoding for server-side validation.

        Args:
            prompt: User prompt to generate response for
            options: Generation configuration options
            response_schema: Pydantic model class for response validation

        Returns:
            Validated LLMStructuredResponse instance

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit exceeded
            AuthenticationError: If API key invalid
            ValidationError: If LLM returned invalid JSON schema
        """
        start_time = time.time()
        model = options.model or self.DEFAULT_MODEL

        try:
            # Build API call kwargs
            api_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": options.max_tokens,
                "temperature": options.temperature,
                "response_model": response_schema,
                "extra_headers": {"anthropic-beta": "structured-outputs-2025-11-13"},
            }
            if options.system_prompt:
                api_kwargs["system"] = options.system_prompt

            message = await self._retry_with_backoff(
                self.client.beta.messages.parse,
                **api_kwargs,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "anthropic_structured_generation_success",
                model=model,
                latency_ms=latency_ms,
            )

            # Return the parsed Pydantic model
            return message.parsed  # type: ignore[no-any-return]

        except (RateLimitError, AuthenticationError):
            raise
        except PydanticValidationError:
            # LLM returned invalid schema - fail fast
            logger.error("anthropic_validation_failed")
            raise
        except Exception as e:
            logger.error("anthropic_structured_generation_failed", error=str(e), exc_info=True)
            raise LLMProviderError(f"Anthropic structured generation failed: {e}") from e

    async def _retry_with_backoff(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Retry with exponential backoff.

        Retries on: RateLimitError, APIConnectionError, APITimeoutError
        No retry on: AuthenticationError, InvalidRequestError, ValidationError

        Args:
            func: Async function to retry
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from successful function call

        Raises:
            Exception: Last exception if all retries exhausted
        """
        wait_times = [2, 4, 8]  # seconds

        for attempt, wait in enumerate(wait_times, start=1):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == len(wait_times):
                    logger.error(
                        "anthropic_retry_exhausted",
                        attempts=attempt,
                        error=str(e),
                    )
                    raise

                logger.warning(
                    "anthropic_retrying",
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)
            except (AuthenticationError, PydanticValidationError):
                # Fail fast on non-retryable errors
                raise
