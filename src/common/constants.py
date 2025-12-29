"""Constants for metadata extraction and content classification."""

# Warhammer 40k Factions
FACTIONS = [
    # Imperium - Space Marines
    "Space Marines",
    "Adeptus Astartes",
    "Ultramarines",
    "Blood Angels",
    "Dark Angels",
    "Space Wolves",
    "Imperial Fists",
    "Salamanders",
    "Raven Guard",
    "Iron Hands",
    "White Scars",
    "Black Templars",
    "Grey Knights",
    "Deathwatch",
    # Imperium - Imperial Guard / Astra Militarum
    "Imperial Guard",
    "Astra Militarum",
    "Cadian Shock Troops",
    "Catachan Jungle Fighters",
    "Death Korps of Krieg",
    # Imperium - Other
    "Adeptus Mechanicus",
    "Adeptus Custodes",
    "Sisters of Battle",
    "Adepta Sororitas",
    "Imperial Navy",
    "Inquisition",
    "Officio Assassinorum",
    "Adeptus Arbites",
    "Adeptus Ministorum",
    # Chaos - Traitor Legions
    "Chaos Space Marines",
    "Death Guard",
    "Thousand Sons",
    "World Eaters",
    "Emperor's Children",
    "Black Legion",
    "Iron Warriors",
    "Night Lords",
    "Alpha Legion",
    "Word Bearers",
    "Chaos Daemons",
    "Chaos Cultists",
    # Xenos - Major
    "Tyranids",
    "Orks",
    "Eldar",
    "Aeldari",
    "Craftworld Eldar",
    "Dark Eldar",
    "Drukhari",
    "Tau",
    "T'au Empire",
    "Necrons",
    # Xenos - Minor
    "Genestealer Cults",
    "Harlequins",
    "Ynnari",
    "Kroot",
    "Vespid",
]

# Faction Aliases (for normalization)
FACTION_ALIASES = {
    "astra militarum": "Imperial Guard",
    "imperial guard": "Imperial Guard",
    "aeldari": "Eldar",
    "eldar": "Eldar",
    "drukhari": "Dark Eldar",
    "dark eldar": "Dark Eldar",
    "t'au empire": "Tau",
    "tau": "Tau",
    "adeptus astartes": "Space Marines",
    "space marines": "Space Marines",
    "adepta sororitas": "Sisters of Battle",
    "sisters of battle": "Sisters of Battle",
}

# Warhammer 40k Timeline Eras
ERAS = [
    # Pre-Imperium
    "Age of Terra",
    "Age of Technology",
    "Dark Age of Technology",
    "Age of Strife",
    "Old Night",
    # Unification and Crusade
    "Unification Wars",
    "Great Crusade",
    # Heresy and Aftermath
    "Horus Heresy",
    "The Scouring",
    # Imperial History
    "Age of Rebirth",
    "Forging",
    "Age of Redemption",
    "Nova Terra Interregnum",
    "Age of Apostasy",
    "Reign of Blood",
    "Age of Forging",
    "Age of Waning",
    "Time of Ending",
    "13th Black Crusade",
    # Current Era
    "Fall of Cadia",
    "Indomitus Crusade",
    "Era Indomitus",
    "Dark Imperium",
]

# Content Type Classification Keywords
CONTENT_TYPE_KEYWORDS = {
    "military": [
        "battle",
        "war",
        "warfare",
        "campaign",
        "siege",
        "tactics",
        "strategy",
        "combat",
        "assault",
        "invasion",
        "deployment",
        "regiment",
        "legion",
        "chapter",
        "crusade",
    ],
    "technology": [
        "weapon",
        "weapons",
        "armor",
        "armour",
        "vehicle",
        "tech",
        "technology",
        "equipment",
        "bolter",
        "power armor",
        "power armour",
        "lasgun",
        "plasma",
        "melta",
        "tank",
        "ship",
        "spacecraft",
        "wargear",
    ],
}

# Spoiler Detection Keywords
SPOILER_KEYWORDS = [
    "spoiler",
    "spoilers",
    "recent lore",
    "latest edition",
    "current timeline",
    "9th edition",
    "10th edition",
    "primaris",
    "guilliman's return",
    "cicatrix maledictum",
]

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
