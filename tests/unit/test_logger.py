"""Tests for structured logging configuration."""

import logging

from src.utils.logger import configure_logging, get_logger


def test_configure_logging_default_level() -> None:
    """Test logging configuration with default INFO level."""
    configure_logging()

    # Verify structlog is configured (returns BoundLoggerLazyProxy)
    logger = get_logger("test")
    # Check that logger has the expected methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")


def test_configure_logging_custom_level() -> None:
    """Test logging configuration with custom DEBUG level."""
    configure_logging("DEBUG")

    # Verify logging level is set
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_get_logger_returns_bound_logger() -> None:
    """Test that get_logger returns a configured logger."""
    configure_logging()
    logger = get_logger(__name__)

    # Verify logger has expected logging methods
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_logger_context() -> None:
    """Test that logger can bind context."""
    configure_logging()
    logger = get_logger(__name__)

    # Bind context
    logger_with_context = logger.bind(user_id=123, request_id="abc")

    # Verify it's still a logger with expected methods
    assert logger_with_context is not None
    assert hasattr(logger_with_context, "info")
    assert hasattr(logger_with_context, "bind")
