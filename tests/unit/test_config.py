"""Tests for configuration management."""

from unittest.mock import patch

import pytest

from src.utils.config import Config, ConfigurationError


def test_config_missing_required_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Config raises error when required env var is missing."""
    # Clear all required env vars
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Mock load_dotenv to prevent loading from .env file
    with patch("src.utils.config.load_dotenv"):
        with pytest.raises(ConfigurationError, match="DISCORD_BOT_TOKEN"):
            Config()


def test_config_loads_required_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Config loads all required environment variables."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test_discord_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")

    with patch("src.utils.config.load_dotenv"):
        config = Config()

        assert config.discord_bot_token == "test_discord_token"
        assert config.openai_api_key == "test_openai_key"
        assert config.database_url == "sqlite:///test.db"


def test_config_default_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Config uses default INFO log level."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test_discord_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    with patch("src.utils.config.load_dotenv"):
        config = Config()

        assert config.log_level == "INFO"


def test_config_custom_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Config respects custom LOG_LEVEL."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test_discord_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    with patch("src.utils.config.load_dotenv"):
        config = Config()

        assert config.log_level == "DEBUG"


def test_config_get_optional_with_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_optional returns default when env var not set."""
    monkeypatch.delenv("OPTIONAL_VAR", raising=False)

    value = Config.get_optional("OPTIONAL_VAR", "default_value")

    assert value == "default_value"


def test_config_get_optional_with_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_optional returns value when env var is set."""
    monkeypatch.setenv("OPTIONAL_VAR", "custom_value")

    value = Config.get_optional("OPTIONAL_VAR", "default_value")

    assert value == "custom_value"
