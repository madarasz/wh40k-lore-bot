"""Prompt template loading and rendering for LLM interactions."""

from pathlib import Path

import structlog

from src.utils.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)

# Default prompts directory relative to project root
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptBuilder:
    """Load and render prompt templates for LLM interactions.

    Templates are loaded from the prompts/ directory and support
    placeholder substitution using {placeholder} syntax.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        """Initialize PromptBuilder with template directory.

        Args:
            prompts_dir: Directory containing prompt templates.
                        Defaults to <project_root>/prompts/
        """
        self.prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

        if not self.prompts_dir.exists():
            raise ConfigurationError(f"Prompts directory not found: {self.prompts_dir}")

        # Cache loaded templates
        self._template_cache: dict[str, str] = {}

        logger.info(
            "prompt_builder_initialized",
            prompts_dir=str(self.prompts_dir),
        )

    def _load_template(self, template_name: str) -> str:
        """Load a template file from the prompts directory.

        Args:
            template_name: Name of the template file (e.g., "system.txt")

        Returns:
            Template content as string

        Raises:
            ConfigurationError: If template file not found
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.prompts_dir / template_name

        if not template_path.exists():
            raise ConfigurationError(f"Template file not found: {template_path}")

        content = template_path.read_text(encoding="utf-8")
        self._template_cache[template_name] = content

        logger.debug("template_loaded", template=template_name)
        return content

    def load_persona(self) -> str:
        """Load the persona template.

        Returns:
            Persona definition string
        """
        return self._load_template("persona.md").strip()

    def build_system_prompt(self, language: str = "hu") -> str:
        """Build the system prompt with persona and language.

        Args:
            language: Language code for response (default: "hu" for Hungarian)

        Returns:
            Rendered system prompt
        """
        template = self._load_template("system.md")
        persona = self.load_persona()

        # Map language codes to full names
        language_names = {
            "hu": "Hungarian",
            "en": "English",
        }
        language_name = language_names.get(language, language)

        return template.format(persona=persona, language=language_name)

    def build_user_prompt(self, chunks: str, question: str) -> str:
        """Build the user prompt with context and question.

        Args:
            chunks: Retrieved context chunks formatted as string
            question: User's question

        Returns:
            Rendered user prompt
        """
        template = self._load_template("user.md")
        return template.format(chunks=chunks, question=question)

    def clear_cache(self) -> None:
        """Clear the template cache to force reload on next access."""
        self._template_cache.clear()
        logger.debug("template_cache_cleared")


# Singleton instance for convenience
prompt_builder = PromptBuilder()
