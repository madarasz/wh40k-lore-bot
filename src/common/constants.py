"""Constants for metadata extraction.

Note: Faction, era, content type, and spoiler constants have been removed
as part of chunk schema refactoring. These were never meaningfully populated
from wiki data and the detection code has been removed.
"""

# Source Book Regex Patterns
SOURCE_BOOK_PATTERNS = [
    r"Codex:\s*([^,.\n]+)",
    r"Index:\s*([^,.\n]+)",
    r"Horus Heresy:\s*Book\s+(\d+)[:\s]*([^,.\n]+)",
    r"Horus Heresy:\s*([^,.\n]+)",
    r"White Dwarf\s+(\d+)",
    r"Campaign Book:\s*([^,.\n]+)",
    r"Imperial Armour:\s*([^,.\n]+)",
    r"Codex Supplement:\s*([^,.\n]+)",
]
