"""Structured output models for LLM responses using Pydantic validation."""

from typing import Any

from pydantic import BaseModel, HttpUrl, model_validator


class LLMStructuredResponse(BaseModel):
    """Structured LLM response with validation.

    This model enforces:
    - When smalltalk=false (lore query): answer and sources are required
    - When smalltalk=true: answer and sources are optional
    - personality_reply is always required for all responses
    - sources must be valid HTTP URLs
    """

    answer: str | None = None
    personality_reply: str
    sources: list[HttpUrl] | None = None
    smalltalk: bool

    @model_validator(mode="after")
    def validate_lore_fields(self) -> "LLMStructuredResponse":
        """Validate that answer and sources are provided for lore queries (smalltalk=false)."""
        if not self.smalltalk:
            if not self.answer:
                raise ValueError("answer required when smalltalk=false")
            if not self.sources:
                raise ValueError("sources required when smalltalk=false")
        return self

    @classmethod
    def model_json_schema_for_llm(cls) -> dict[str, Any]:
        """Generate JSON schema for LLM structured output.

        Returns a JSON schema suitable for OpenAI's response_format
        or Anthropic's response_model parameter.
        """
        return cls.model_json_schema()
