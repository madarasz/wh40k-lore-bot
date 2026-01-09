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

from src.llm.base_provider import GenerationOptions, LLMProvider, LLMResponse
from src.llm.pricing import pricing_calculator
from src.llm.structured_output import LLMStructuredResponse
from src.utils.config import get_required_env
from src.utils.exceptions import LLMProviderError

logger = structlog.get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider with structured output support.

    Supports:
    - claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-5
    - Structured output via beta.messages.parse() with constrained decoding
    - Fallback to tool use pattern + client-side validation
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
        start_time = time.time()
        model = options.model or self.DEFAULT_MODEL

        try:
            response = await self._retry_with_backoff(
                self.client.messages.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            tokens_prompt = response.usage.input_tokens
            tokens_completion = response.usage.output_tokens
            cost_usd = pricing_calculator.calculate_cost(model, tokens_prompt, tokens_completion)

            text = response.content[0].text if response.content else ""

            logger.info(
                "anthropic_generation_success",
                model=model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                text=text,
                provider=self.get_provider_name(),
                model=model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )

        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:
            logger.error("anthropic_generation_failed", error=str(e), exc_info=True)
            raise LLMProviderError(f"Anthropic generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
    ) -> LLMStructuredResponse:
        """Generate structured JSON response with Pydantic validation.

        Uses beta.messages.parse() with constrained decoding for server-side
        validation with fallback to tool use pattern if beta API fails.

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

        # Try server-side constrained decoding first
        try:
            message = await self._retry_with_backoff(
                self.client.beta.messages.parse,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=options.max_tokens,
                response_model=response_schema,
                extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
            )

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "anthropic_structured_generation_success",
                validation_type="server_side",
                model=model,
                latency_ms=latency_ms,
            )

            # Return the parsed Pydantic model
            return message.parsed  # type: ignore[no-any-return]

        except AttributeError:
            # Beta API not available, fall back to tool use pattern
            logger.warning("anthropic_beta_api_unavailable", fallback="tool_use_pattern")
            return await self._generate_structured_fallback(
                prompt, options, response_schema, start_time
            )

        except (RateLimitError, AuthenticationError):
            raise
        except PydanticValidationError:
            # LLM returned invalid schema - fail fast
            logger.error("anthropic_validation_failed", validation_type="server_side")
            raise
        except Exception as e:
            logger.error("anthropic_structured_generation_failed", error=str(e), exc_info=True)
            raise LLMProviderError(f"Anthropic structured generation failed: {e}") from e

    async def _generate_structured_fallback(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
        start_time: float,
    ) -> LLMStructuredResponse:
        """Fallback to tool use pattern when beta API unavailable.

        Args:
            prompt: User prompt
            options: Generation options
            response_schema: Pydantic model class
            start_time: Request start timestamp

        Returns:
            Validated LLMStructuredResponse instance

        Raises:
            ValidationError: If LLM returned invalid JSON schema
        """
        model = options.model or self.DEFAULT_MODEL

        # Create tool definition from Pydantic schema
        tool_definition = {
            "name": "respond",
            "description": "Respond to the user query with structured data",
            "input_schema": response_schema.model_json_schema(),
        }

        try:
            response = await self._retry_with_backoff(
                self.client.messages.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=options.max_tokens,
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": "respond"},
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract tool use result
            tool_use = next(
                (block for block in response.content if block.type == "tool_use"),
                None,
            )

            if not tool_use:
                raise LLMProviderError("No tool use in response")

            # Validate with Pydantic
            validated_response = response_schema.model_validate(tool_use.input)

            logger.info(
                "anthropic_structured_generation_success",
                validation_type="client_side",
                model=model,
                latency_ms=latency_ms,
            )

            return validated_response  # type: ignore[return-value]

        except PydanticValidationError:
            logger.error("anthropic_validation_failed", validation_type="client_side")
            raise

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
