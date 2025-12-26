"""Configuration management for environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        """Load configuration from .env file and environment."""
        # Load .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        # Required configuration
        self.discord_bot_token = self._get_required("DISCORD_BOT_TOKEN")
        self.openai_api_key = self._get_required("OPENAI_API_KEY")
        self.database_url = self._get_required("DATABASE_URL")

        # Optional configuration with defaults
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def _get_required(self, key: str) -> str:
        """Get required environment variable or raise error.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ConfigurationError: If variable is not set
        """
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(f"{key} environment variable is not set")
        return value

    @staticmethod
    def get_optional(key: str, default: str | None = None) -> str | None:
        """Get optional environment variable with default value.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
