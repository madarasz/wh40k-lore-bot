"""Response formatter for LLM structured outputs."""

import structlog

from src.llm.structured_output import LLMStructuredResponse

logger = structlog.get_logger(__name__)


class ResponseFormatter:
    """Format LLM structured responses for different output targets.

    Handles formatting of structured LLM responses for CLI and Discord outputs,
    including personality replies, source citations, and smalltalk handling.
    """

    HEADERS: dict[str, str] = {
        "hu": "📚 Források:",
        "en": "📚 Sources:",
    }
    MAX_SOURCES: int = 5

    def format_cli_response(
        self,
        llm_response: LLMStructuredResponse,
        language: str = "hu",
    ) -> str:
        """Format structured LLM response for CLI output.

        Args:
            llm_response: Structured response from LLM with answer, personality_reply,
                sources, and smalltalk flag.
            language: Language code for headers ("hu" or "en"). Defaults to "hu".

        Returns:
            Formatted string for CLI display. For lore questions includes answer,
            personality reply, and sources. For smalltalk returns only personality reply.
        """
        if llm_response.smalltalk:
            logger.debug("formatting_smalltalk_response")
            return llm_response.personality_reply

        parts: list[str] = []

        # Answer
        if llm_response.answer:
            parts.append(llm_response.answer)

        # Personality reply (styled differently in CLI adapter)
        parts.append(f"\n{llm_response.personality_reply}")

        # Sources
        if llm_response.sources:
            header = self.HEADERS.get(language, self.HEADERS["en"])
            sources = llm_response.sources[: self.MAX_SOURCES]
            source_lines = [f"- {str(url)}" for url in sources]
            parts.append(f"\n{header}\n" + "\n".join(source_lines))

            if len(llm_response.sources) > self.MAX_SOURCES:
                logger.debug(
                    "sources_truncated",
                    total_sources=len(llm_response.sources),
                    displayed_sources=self.MAX_SOURCES,
                )

        logger.debug(
            "formatted_cli_response",
            has_answer=bool(llm_response.answer),
            source_count=len(llm_response.sources) if llm_response.sources else 0,
            language=language,
        )

        return "\n".join(parts)

    def format_discord_response(
        self,
        llm_response: LLMStructuredResponse,
        language: str = "hu",
    ) -> str:
        """Format structured LLM response for Discord (stub for Epic 3).

        Args:
            llm_response: Structured response from LLM.
            language: Language code for headers ("hu" or "en"). Defaults to "hu".

        Returns:
            Formatted string for Discord. Currently returns CLI format as placeholder.
        """
        # TODO: Epic 3 - Implement Discord Embed formatting
        return self.format_cli_response(llm_response, language)
