"""Custom exception hierarchy for the application."""


class WH40KLoreBotError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, is_retryable: bool = False) -> None:
        """Initialize exception.

        Args:
            message: Error message
            is_retryable: Whether the operation can be retried
        """
        super().__init__(message)
        self.message = message
        self.is_retryable = is_retryable


class ConfigurationError(WH40KLoreBotError):
    """Configuration or environment setup error."""

    pass


class DatabaseError(WH40KLoreBotError):
    """Database operation error."""

    pass


class LLMProviderError(WH40KLoreBotError):
    """LLM provider API error."""

    pass


class RetrievalError(WH40KLoreBotError):
    """Vector/text retrieval error."""

    pass


class IngestionError(WH40KLoreBotError):
    """Data ingestion pipeline error."""

    pass


class EmbeddingGenerationError(IngestionError):
    """Embedding generation error."""

    pass


class ValidationError(WH40KLoreBotError):
    """Input validation error."""

    pass
