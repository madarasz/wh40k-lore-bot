"""Centralized pricing calculator for all LLM and embedding operations."""

import structlog

logger = structlog.get_logger(__name__)


class PricingCalculator:
    """Centralized pricing for all LLM and embedding operations.

    Provides cost estimation for all supported models including LLM providers
    (OpenAI, Anthropic) and embedding models. All prices are in USD per 1K tokens.
    """

    # Pricing per 1K tokens (USD) - updated December 2025
    PRICING: dict[str, dict[str, float]] = {
        # OpenAI LLM
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        # Anthropic LLM
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
        "claude-opus-4-5": {"input": 0.015, "output": 0.075},
        # OpenAI Embeddings
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    }

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
        """Calculate cost for a generation or embedding request.

        Args:
            model: Model identifier (e.g., "gpt-4.1", "claude-sonnet-4-5")
            prompt_tokens: Number of tokens in the prompt/input
            completion_tokens: Number of tokens in the completion/output (0 for embeddings)

        Returns:
            Estimated cost in USD

        Raises:
            ValueError: If model is not found in pricing table
        """
        pricing = self.PRICING.get(model)
        if not pricing:
            available = ", ".join(self.PRICING.keys())
            raise ValueError(f"Unknown model: {model}. Available models: {available}")

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        logger.debug(
            "cost_calculated",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(total_cost, 6),
        )

        return total_cost

    def get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing information for a specific model.

        Args:
            model: Model identifier

        Returns:
            Dictionary with 'input' and 'output' pricing per 1K tokens

        Raises:
            ValueError: If model is not found in pricing table
        """
        pricing = self.PRICING.get(model)
        if not pricing:
            available = ", ".join(self.PRICING.keys())
            raise ValueError(f"Unknown model: {model}. Available models: {available}")
        return pricing

    def get_supported_models(self) -> list[str]:
        """Get list of all supported models.

        Returns:
            List of model identifiers
        """
        return list(self.PRICING.keys())


# Singleton instance for convenience
pricing_calculator = PricingCalculator()
