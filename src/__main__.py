"""Main entry point for the WH40K Lore Bot application."""

import sys

from src.utils.config import Config, ConfigurationError
from src.utils.logger import configure_logging, get_logger


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load configuration
        config = Config()

        # Configure logging
        configure_logging(config.log_level)
        logger = get_logger(__name__)

        logger.info("application_starting", version="0.1.0")

        # TODO: Initialize and run Discord bot (Story 3.x)
        logger.info(
            "discord_bot_not_implemented",
            message="Discord bot will be implemented in Epic 3",
        )

        return 0

    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print(
            "Please check your .env file and ensure all required variables are set.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
