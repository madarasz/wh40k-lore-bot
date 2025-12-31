"""Data quality validation functions for ingestion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree

from src.ingestion.models import Chunk

# Validation constants
MIN_MARKDOWN_LENGTH = 50
NORM_LOWER_BOUND = 0.99
NORM_UPPER_BOUND = 1.01


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        valid: Whether the validation passed
        issues: List of error messages (validation failures)
        warnings: List of warning messages (non-critical issues)
    """

    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if validation passed with no issues.

        Returns:
            True if valid and no issues found
        """
        return self.valid and len(self.issues) == 0

    def has_warnings(self) -> bool:
        """Check if validation has warnings.

        Returns:
            True if warnings exist
        """
        return len(self.warnings) > 0


@dataclass
class ValidationMetrics:
    """Metrics for tracking validation statistics during ingestion.

    Attributes:
        articles_processed: Number of articles successfully processed
        articles_skipped: Number of articles skipped due to validation failures
        chunks_created: Total number of chunks created
        chunks_discarded: Number of chunks discarded (too short/long)
        embeddings_generated: Number of successful embeddings
        embedding_failures: Number of embedding generation failures
        metadata_extractions: Number of metadata extraction attempts
        metadata_failures: Number of metadata extraction failures
    """

    articles_processed: int = 0
    articles_skipped: int = 0
    chunks_created: int = 0
    chunks_discarded: int = 0
    embeddings_generated: int = 0
    embedding_failures: int = 0
    metadata_extractions: int = 0
    metadata_failures: int = 0

    def get_skip_rate(self) -> float:
        """Calculate article skip rate as percentage.

        Returns:
            Skip rate (0.0 to 100.0)
        """
        total = self.articles_processed + self.articles_skipped
        if total == 0:
            return 0.0
        return (self.articles_skipped / total) * 100.0

    def get_discard_rate(self) -> float:
        """Calculate chunk discard rate as percentage.

        Returns:
            Discard rate (0.0 to 100.0)
        """
        total = self.chunks_created + self.chunks_discarded
        if total == 0:
            return 0.0
        return (self.chunks_discarded / total) * 100.0

    def get_failure_rate(self) -> float:
        """Calculate embedding failure rate as percentage.

        Returns:
            Failure rate (0.0 to 100.0)
        """
        total = self.embeddings_generated + self.embedding_failures
        if total == 0:
            return 0.0
        return (self.embedding_failures / total) * 100.0


def validate_xml(xml_path: str) -> ValidationResult:
    """Validate XML file structure and content.

    Args:
        xml_path: Path to XML file

    Returns:
        ValidationResult with validation status
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Check file exists
    path = Path(xml_path)
    if not path.exists():
        issues.append(f"XML file not found: {xml_path}")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Check file is readable
    if not path.is_file():
        issues.append(f"Path is not a file: {xml_path}")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Validate XML is well-formed
    try:
        tree = etree.parse(str(path))
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        issues.append(f"Malformed XML: {e}")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Validate MediaWiki namespace
    namespace = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    if "mediawiki.org" not in namespace:
        warnings.append(f"Unexpected XML namespace: {namespace}")

    # Check for required elements
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    pages = root.findall(f".//{ns}page")
    if not pages:
        issues.append("No <page> elements found in XML")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Sample first page for structure validation
    first_page = pages[0]
    if first_page.find(f"{ns}title") is None:
        warnings.append("First page missing <title> element")
    if first_page.find(f"{ns}revision/{ns}text") is None:
        warnings.append("First page missing <text> element")

    return ValidationResult(
        valid=True,
        issues=issues,
        warnings=warnings,
    )


def validate_markdown(markdown: str) -> ValidationResult:
    """Validate markdown content quality.

    Args:
        markdown: Markdown content to validate

    Returns:
        ValidationResult with validation status
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Check non-empty
    if not markdown or not markdown.strip():
        issues.append("Markdown content is empty")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Check has actual content (not just whitespace/frontmatter)
    lines = [line.strip() for line in markdown.split("\n") if line.strip()]
    content_lines = [line for line in lines if not line.startswith("---")]

    if len(content_lines) == 0:
        issues.append("Markdown has no content (only frontmatter)")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Check for reasonable content length
    if len(markdown) < MIN_MARKDOWN_LENGTH:
        warnings.append(f"Markdown content is very short: {len(markdown)} characters")

    return ValidationResult(
        valid=True,
        issues=issues,
        warnings=warnings,
    )


def validate_chunk(chunk: Chunk, min_tokens: int = 50, max_tokens: int = 500) -> ValidationResult:
    """Validate chunk quality and token count.

    Args:
        chunk: Chunk object to validate
        min_tokens: Minimum acceptable token count
        max_tokens: Maximum acceptable token count

    Returns:
        ValidationResult with validation status
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Check chunk_text is non-empty
    if not chunk.chunk_text or not chunk.chunk_text.strip():
        issues.append("Chunk text is empty")
        return ValidationResult(valid=False, issues=issues, warnings=warnings)

    # Check required fields
    if not chunk.article_title:
        issues.append("Chunk missing article_title")
    if not chunk.section_path:
        issues.append("Chunk missing section_path")

    # Estimate token count (rough approximation: 4 chars ≈ 1 token)
    estimated_tokens = len(chunk.chunk_text) // 4

    # Validate token count bounds
    if estimated_tokens < min_tokens:
        issues.append(f"Chunk too short: ~{estimated_tokens} tokens (min: {min_tokens})")
    if estimated_tokens > max_tokens:
        issues.append(f"Chunk too long: ~{estimated_tokens} tokens (max: {max_tokens})")

    valid = len(issues) == 0

    return ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
    )


def validate_embedding(embedding: np.ndarray, expected_dim: int = 1536) -> ValidationResult:
    """Validate embedding quality.

    Args:
        embedding: Embedding vector (numpy array)
        expected_dim: Expected embedding dimensions

    Returns:
        ValidationResult with validation status
    """
    issues = []
    warnings = []

    # Check dimensions
    if embedding.shape[0] != expected_dim:
        issues.append(f"Invalid dimensions: {embedding.shape[0]} (expected {expected_dim})")

    # Check for NaN values
    if np.isnan(embedding).any():
        issues.append("Embedding contains NaN values")

    # Check for Inf values
    if np.isinf(embedding).any():
        issues.append("Embedding contains Inf values")

    # Check dtype
    if embedding.dtype not in [np.float32, np.float64]:
        warnings.append(f"Unexpected dtype: {embedding.dtype}")

    # Check normalization (L2 norm should be close to 1.0)
    norm = np.linalg.norm(embedding)
    if not (NORM_LOWER_BOUND <= norm <= NORM_UPPER_BOUND):
        warnings.append(f"Embedding not normalized: L2 norm = {norm:.4f}")

    valid = len(issues) == 0

    return ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
    )


def validate_metadata(metadata: dict[str, object]) -> ValidationResult:
    """Validate metadata completeness and structure.

    Args:
        metadata: Metadata dictionary to validate

    Returns:
        ValidationResult with validation status
    """
    issues = []
    warnings = []

    # Required fields
    required_fields = ["content_type", "spoiler_flag"]

    for field_name in required_fields:
        if field_name not in metadata:
            issues.append(f"Missing required field: {field_name}")

    # Validate types
    if "content_type" in metadata and not isinstance(metadata["content_type"], str):
        issues.append(f"Invalid type for content_type: {type(metadata['content_type'])}")

    if "spoiler_flag" in metadata and not isinstance(metadata["spoiler_flag"], bool):
        issues.append(f"Invalid type for spoiler_flag: {type(metadata['spoiler_flag'])}")

    # Optional fields with type validation
    if "faction" in metadata and not isinstance(metadata["faction"], str):
        warnings.append(f"Invalid type for faction: {type(metadata['faction'])}")

    if "eras" in metadata and not isinstance(metadata["eras"], list):
        warnings.append(f"Invalid type for eras: {type(metadata['eras'])}")

    valid = len(issues) == 0

    return ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
    )
