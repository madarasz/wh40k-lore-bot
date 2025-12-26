"""Tests for custom exception hierarchy."""

from src.utils.exceptions import (
    ConfigurationError,
    DatabaseError,
    IngestionError,
    LLMProviderError,
    RetrievalError,
    ValidationError,
    WH40KLoreBotError,
)


def test_base_exception_message() -> None:
    """Test that base exception stores message."""
    error = WH40KLoreBotError("test error")

    assert str(error) == "test error"
    assert error.message == "test error"
    assert error.is_retryable is False


def test_base_exception_retryable() -> None:
    """Test that base exception can be marked as retryable."""
    error = WH40KLoreBotError("test error", is_retryable=True)

    assert error.is_retryable is True


def test_configuration_error_inheritance() -> None:
    """Test that ConfigurationError inherits from base exception."""
    error = ConfigurationError("config error")

    assert isinstance(error, WH40KLoreBotError)
    assert str(error) == "config error"


def test_database_error_inheritance() -> None:
    """Test that DatabaseError inherits from base exception."""
    error = DatabaseError("db error")

    assert isinstance(error, WH40KLoreBotError)


def test_llm_provider_error_inheritance() -> None:
    """Test that LLMProviderError inherits from base exception."""
    error = LLMProviderError("llm error", is_retryable=True)

    assert isinstance(error, WH40KLoreBotError)
    assert error.is_retryable is True


def test_retrieval_error_inheritance() -> None:
    """Test that RetrievalError inherits from base exception."""
    error = RetrievalError("retrieval error")

    assert isinstance(error, WH40KLoreBotError)


def test_ingestion_error_inheritance() -> None:
    """Test that IngestionError inherits from base exception."""
    error = IngestionError("ingestion error")

    assert isinstance(error, WH40KLoreBotError)


def test_validation_error_inheritance() -> None:
    """Test that ValidationError inherits from base exception."""
    error = ValidationError("validation error")

    assert isinstance(error, WH40KLoreBotError)
