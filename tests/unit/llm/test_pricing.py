"""Unit tests for PricingCalculator."""

import pytest

from src.llm.pricing import PricingCalculator, pricing_calculator


class TestPricingCalculator:
    """Test PricingCalculator implementation."""

    @pytest.fixture
    def calculator(self) -> PricingCalculator:
        """Create PricingCalculator instance."""
        return PricingCalculator()

    def test_calculate_cost_gpt41(self, calculator: PricingCalculator) -> None:
        """Test cost calculation for gpt-4.1."""
        cost = calculator.calculate_cost("gpt-4.1", 1000, 500)
        # input: 1000/1000 * 0.002 = 0.002
        # output: 500/1000 * 0.008 = 0.004
        # total: 0.006
        assert abs(cost - 0.006) < 0.0001

    def test_calculate_cost_claude_sonnet(self, calculator: PricingCalculator) -> None:
        """Test cost calculation for claude-sonnet-4-5."""
        cost = calculator.calculate_cost("claude-sonnet-4-5", 1000, 500)
        # input: 1000/1000 * 0.003 = 0.003
        # output: 500/1000 * 0.015 = 0.0075
        # total: 0.0105
        assert abs(cost - 0.0105) < 0.0001

    def test_calculate_cost_claude_haiku(self, calculator: PricingCalculator) -> None:
        """Test cost calculation for claude-haiku-4-5."""
        cost = calculator.calculate_cost("claude-haiku-4-5", 1000, 500)
        # input: 1000/1000 * 0.0008 = 0.0008
        # output: 500/1000 * 0.004 = 0.002
        # total: 0.0028
        assert abs(cost - 0.0028) < 0.0001

    def test_calculate_cost_claude_opus(self, calculator: PricingCalculator) -> None:
        """Test cost calculation for claude-opus-4-5."""
        cost = calculator.calculate_cost("claude-opus-4-5", 1000, 500)
        # input: 1000/1000 * 0.015 = 0.015
        # output: 500/1000 * 0.075 = 0.0375
        # total: 0.0525
        assert abs(cost - 0.0525) < 0.0001

    def test_calculate_cost_embedding(self, calculator: PricingCalculator) -> None:
        """Test cost calculation for text-embedding-3-small."""
        cost = calculator.calculate_cost("text-embedding-3-small", 1000, 0)
        # input: 1000/1000 * 0.00002 = 0.00002
        # output: 0
        # total: 0.00002
        assert abs(cost - 0.00002) < 0.000001

    def test_calculate_cost_unknown_model_fails(self, calculator: PricingCalculator) -> None:
        """Test cost calculation fails for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            calculator.calculate_cost("unknown-model", 1000, 500)

    def test_get_model_pricing(self, calculator: PricingCalculator) -> None:
        """Test getting model pricing."""
        pricing = calculator.get_model_pricing("gpt-4.1")
        assert pricing["input"] == 0.002
        assert pricing["output"] == 0.008

    def test_get_model_pricing_unknown_fails(self, calculator: PricingCalculator) -> None:
        """Test getting pricing for unknown model fails."""
        with pytest.raises(ValueError, match="Unknown model"):
            calculator.get_model_pricing("unknown-model")

    def test_get_supported_models(self, calculator: PricingCalculator) -> None:
        """Test getting supported models list."""
        models = calculator.get_supported_models()
        assert "gpt-4.1" in models
        assert "claude-sonnet-4-5" in models
        assert "claude-haiku-4-5" in models
        assert "claude-opus-4-5" in models
        assert "text-embedding-3-small" in models

    def test_singleton_instance(self) -> None:
        """Test singleton instance is available."""
        assert pricing_calculator is not None
        assert isinstance(pricing_calculator, PricingCalculator)
