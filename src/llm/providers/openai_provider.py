"""OpenAI LLM provider implementation with structured output support."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import structlog
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider with structured output support.

    Supports:
    - GPT-4.1 (default)
    - Structured output via beta.chat.completions.parse()
    - Fallback to JSON schema + client-side validation
    - Retry logic with exponential backoff
    """

    # Supported model
    SUPPORTED_MODEL = "gpt-4.1"

    def __init__(self) -> None:
        """Initialize OpenAI provider with API key from environment."""
        api_key = get_required_env("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)

        logger.info(
            "openai_provider_initialized",
            supported_model=self.SUPPORTED_MODEL,
        )

    def get_provider_name(self) -> str:
        """Get provider identifier."""
        return "openai"

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
        model = options.model or self.SUPPORTED_MODEL

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._retry_with_backoff(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            tokens_prompt = response.usage.prompt_tokens if response.usage else 0
            tokens_completion = response.usage.completion_tokens if response.usage else 0
            cost_usd = pricing_calculator.calculate_cost(model, tokens_prompt, tokens_completion)

            text = response.choices[0].message.content or ""

            logger.info(
                "openai_generation_success",
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
            logger.error("openai_generation_failed", error=str(e), exc_info=True)
            raise LLMProviderError(f"OpenAI generation failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
    ) -> LLMStructuredResponse:
        """Generate structured JSON response with Pydantic validation.

        Uses beta.chat.completions.parse() for server-side validation with
        fallback to client-side validation if beta API fails.

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
        model = options.model or self.SUPPORTED_MODEL

        messages = [{"role": "user", "content": prompt}]

        # Try server-side validation first
        try:
            completion = await self._retry_with_backoff(
                self.client.beta.chat.completions.parse,
                model=model,
                messages=messages,
                response_format=response_schema,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "openai_structured_generation_success",
                validation_type="server_side",
                model=model,
                latency_ms=latency_ms,
            )

            # Return the parsed Pydantic model
            return completion.choices[0].message.parsed  # type: ignore[no-any-return]

        except AttributeError:
            # Beta API not available, fall back to client-side validation
            logger.warning("openai_beta_api_unavailable", fallback="client_side_validation")
            return await self._generate_structured_fallback(
                prompt, options, response_schema, messages, start_time
            )

        except (RateLimitError, AuthenticationError):
            raise
        except PydanticValidationError:
            # LLM returned invalid schema - fail fast
            logger.error("openai_validation_failed", validation_type="server_side")
            raise
        except Exception as e:
            logger.error("openai_structured_generation_failed", error=str(e), exc_info=True)
            raise LLMProviderError(f"OpenAI structured generation failed: {e}") from e

    async def _generate_structured_fallback(
        self,
        prompt: str,
        options: GenerationOptions,
        response_schema: type[BaseModel],
        messages: list[dict[str, str]],
        start_time: float,
    ) -> LLMStructuredResponse:
        """Fallback to client-side validation when beta API unavailable.

        Args:
            prompt: User prompt
            options: Generation options
            response_schema: Pydantic model class
            messages: Prepared messages list
            start_time: Request start timestamp

        Returns:
            Validated LLMStructuredResponse instance

        Raises:
            ValidationError: If LLM returned invalid JSON schema
        """
        model = options.model or self.SUPPORTED_MODEL

        try:
            json_schema = response_schema.model_json_schema()
            completion = await self._retry_with_backoff(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": json_schema},
                },
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse and validate JSON
            json_content = completion.choices[0].message.content or "{}"
            json_response = json.loads(json_content)
            validated_response = response_schema.model_validate(json_response)

            logger.info(
                "openai_structured_generation_success",
                validation_type="client_side",
                model=model,
                latency_ms=latency_ms,
            )

            return validated_response  # type: ignore[return-value]

        except PydanticValidationError:
            logger.error("openai_validation_failed", validation_type="client_side")
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
                        "openai_retry_exhausted",
                        attempts=attempt,
                        error=str(e),
                    )
                    raise

                logger.warning(
                    "openai_retrying",
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)
            except (AuthenticationError, PydanticValidationError):
                # Fail fast on non-retryable errors
                raise
