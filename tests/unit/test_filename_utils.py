"""Unit tests for filename sanitization utilities."""

import pytest

from src.ingestion.filename_utils import sanitize_filename


class TestSanitizeFilename:
    """Test cases for sanitize_filename function."""

    def test_spaces_to_underscores(self) -> None:
        """Test that spaces are converted to underscores."""
        assert sanitize_filename("Blood Angels") == "Blood_Angels"
        assert sanitize_filename("Space Marine Chapter") == "Space_Marine_Chapter"

    def test_slashes_to_hyphens(self) -> None:
        """Test that slashes are converted to hyphens."""
        assert sanitize_filename("Adeptus Mechanicus/Forge Worlds") == (
            "Adeptus_Mechanicus-Forge_Worlds"
        )
        assert sanitize_filename("Imperial/Chaos") == "Imperial-Chaos"

    def test_colons_to_hyphens(self) -> None:
        """Test that colons are converted to hyphens."""
        assert sanitize_filename("Chapter Master: Dante") == "Chapter_Master-_Dante"
        assert sanitize_filename("Title: Subtitle") == "Title-_Subtitle"

    def test_special_character_removal(self) -> None:
        """Test that special characters are removed."""
        assert sanitize_filename("Test!@#$%^&*()File") == "TestFile"
        assert sanitize_filename("File(with)brackets[here]") == "Filewithbracketshere"
        assert sanitize_filename("Quotes'n\"stuff") == "Quotesnstuff"

    def test_unicode_normalization(self) -> None:
        """Test that unicode characters are normalized to ASCII."""
        assert sanitize_filename("Château") == "Chateau"
        assert sanitize_filename("Naïve") == "Naive"
        assert sanitize_filename("Björk") == "Bjork"
        assert sanitize_filename("Señor") == "Senor"

    def test_long_filename_truncation(self) -> None:
        """Test that filenames longer than max_length are truncated."""
        long_title = "A" * 300
        result = sanitize_filename(long_title, max_length=255)
        assert len(result) == 255
        assert result == "A" * 255

    def test_long_filename_with_trailing_special_chars(self) -> None:
        """Test that trailing hyphens/underscores are removed after truncation."""
        long_title = "A" * 250 + "_____"
        result = sanitize_filename(long_title, max_length=255)
        # Should truncate and strip trailing underscores
        assert not result.endswith("_")
        assert len(result) <= 255

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            sanitize_filename("")

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Title cannot be empty"):
            sanitize_filename("   ")

    def test_all_special_chars_raises_error(self) -> None:
        """Test that string with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="resulted in empty filename"):
            sanitize_filename("!@#$%^&*()")

    def test_unicode_only_removal(self) -> None:
        """Test handling of unicode-only strings that become empty."""
        # These characters don't have ASCII equivalents
        with pytest.raises(ValueError, match="resulted in empty filename"):
            sanitize_filename("★☆✦✧")

    def test_consecutive_underscores_hyphens_collapsed(self) -> None:
        """Test that consecutive underscores/hyphens are collapsed to single char."""
        assert sanitize_filename("Test___Multiple___Underscores") == ("Test_Multiple_Underscores")
        assert sanitize_filename("Test---Multiple---Hyphens") == "Test-Multiple-Hyphens"

    def test_leading_trailing_special_chars_stripped(self) -> None:
        """Test that leading/trailing underscores and hyphens are stripped."""
        assert sanitize_filename("___Leading Underscores") == "Leading_Underscores"
        assert sanitize_filename("Trailing Hyphens---") == "Trailing_Hyphens"
        assert sanitize_filename("___Both___") == "Both"

    def test_mixed_transformations(self) -> None:
        """Test realistic filenames with multiple transformations."""
        assert sanitize_filename("Adeptus Mechanicus/Forge Worlds") == (
            "Adeptus_Mechanicus-Forge_Worlds"
        )
        assert sanitize_filename("Chapter Master: Gabriel Angelos (Blood Ravens)") == (
            "Chapter_Master-_Gabriel_Angelos_Blood_Ravens"
        )

    def test_preserves_alphanumeric_underscore_hyphen(self) -> None:
        """Test that valid characters are preserved."""
        assert sanitize_filename("Valid_File-Name123") == "Valid_File-Name123"
        assert sanitize_filename("ABC_123-xyz") == "ABC_123-xyz"
