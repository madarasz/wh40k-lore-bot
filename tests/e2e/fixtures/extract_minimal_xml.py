"""Extract minimal XML subset for E2E testing.

This script extracts specific pages from the full WH40K wiki XML export to create
a minimal test dataset for end-to-end CLI workflow testing.
"""

import sys
from pathlib import Path

from lxml import etree

# MediaWiki XML namespace
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Page IDs to extract for E2E testing
TARGET_PAGE_IDS = {"58", "8783", "206", "86", "1839", "116", "91375", "91130", "16481", "2579"}


def extract_minimal_xml(source_xml: Path, page_ids: set[str], output_path: Path) -> None:
    """Extract specific pages from large XML export.

    Args:
        source_xml: Path to the full XML export
        page_ids: Set of page IDs to extract
        output_path: Path to write minimal XML file

    Raises:
        FileNotFoundError: If source XML doesn't exist
    """
    if not source_xml.exists():
        raise FileNotFoundError(f"Source XML not found: {source_xml}")

    print(f"Extracting {len(page_ids)} pages from {source_xml}...")

    # Start building output XML
    root = etree.Element(
        "mediawiki",
        xmlns="http://www.mediawiki.org/xml/export-0.11/",
        version="0.11",
    )
    root.set("{http://www.w3.org/XML/1998/namespace}lang", "en")

    # Track found pages
    found_pages = 0

    # Stream through source XML
    context = etree.iterparse(str(source_xml), events=("end",), tag=f"{NS}page")

    try:
        for _, elem in context:
            try:
                # Extract page ID
                id_elem = elem.find(f"{NS}id")
                if id_elem is None or id_elem.text is None:
                    elem.clear()
                    continue

                page_id = id_elem.text.strip()

                # Check if this is a page we want
                if page_id in page_ids:
                    # Make a deep copy of the element and add to our output
                    # We need to strip namespaces for proper copying
                    page_copy = _copy_element_without_ns(elem)
                    root.append(page_copy)

                    found_pages += 1
                    print(f"  Found page {page_id} ({found_pages}/{len(page_ids)})")

                    # Stop early if we've found all pages
                    if found_pages >= len(page_ids):
                        break

            finally:
                # Always clear element to free memory
                elem.clear()

    finally:
        # Write output XML
        tree = etree.ElementTree(root)
        tree.write(
            str(output_path),
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
        )

    print(f"\nExtracted {found_pages} pages to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def _copy_element_without_ns(elem: etree._Element) -> etree._Element:
    """Create a copy of an element, stripping namespace prefixes.

    Args:
        elem: Element to copy

    Returns:
        New element without namespace in tag names
    """
    # Create new element with tag name (namespace already in it from source)
    new_elem = etree.Element(elem.tag, attrib=dict(elem.attrib))  # type: ignore[misc]

    # Copy text
    new_elem.text = elem.text
    new_elem.tail = elem.tail

    # Recursively copy children
    for child in elem:
        new_elem.append(_copy_element_without_ns(child))

    return new_elem


def create_synthetic_xml(output_path: Path) -> None:
    """Create synthetic 10-page XML if real source unavailable.

    Args:
        output_path: Path to write synthetic XML file
    """
    print("Creating synthetic XML with 10 test pages...")

    # Create minimal synthetic XML using pattern from test_test_bed_builder.py
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11" xml:lang="en">
  <page>
    <id>58</id>
    <ns>0</ns>
    <title>Blood Angels</title>
    <revision>
      <timestamp>2024-01-15T12:00:00Z</timestamp>
      <text>==Overview==
The '''Blood Angels''' are a [[Space Marine]] chapter known for their nobility and artistry.

They were led by their [[Primarch]], [[Sanguinius]].
They fought in the [[Horus Heresy]] and remain loyal to the [[Imperium]].

Their homeworld is [[Baal]].</text>
    </revision>
  </page>
  <page>
    <id>8783</id>
    <ns>0</ns>
    <title>Ultramarines</title>
    <revision>
      <timestamp>2024-01-16T10:00:00Z</timestamp>
      <text>==The Ultramarines==
The [[Ultramarines]] are a [[Space Marine]] chapter and one of the most renowned.

Their [[Primarch]] was [[Roboute Guilliman]].
They hail from [[Ultramar]] and follow the [[Codex Astartes]].</text>
    </revision>
  </page>
  <page>
    <id>206</id>
    <ns>0</ns>
    <title>Sanguinius</title>
    <revision>
      <timestamp>2024-01-17T14:30:00Z</timestamp>
      <text>==Sanguinius==
[[Sanguinius]] was the [[Primarch]] of the [[Blood Angels]].

He died defending the [[Emperor]] during the [[Horus Heresy]].
He had angelic wings and was beloved by all.</text>
    </revision>
  </page>
  <page>
    <id>86</id>
    <ns>0</ns>
    <title>Space Marine</title>
    <revision>
      <timestamp>2024-01-18T09:15:00Z</timestamp>
      <text>==Space Marines==
The [[Space Marine]]s, or Adeptus Astartes, are superhuman warriors.

They serve the [[Imperium]] and are organized into [[Chapter]]s.
Famous chapters include the [[Blood Angels]] and [[Ultramarines]].</text>
    </revision>
  </page>
  <page>
    <id>1839</id>
    <ns>0</ns>
    <title>Primarch</title>
    <revision>
      <timestamp>2024-01-19T11:00:00Z</timestamp>
      <text>==Primarchs==
The [[Primarch]]s were demigod-like beings created by the [[Emperor]].

[[Sanguinius]] and [[Roboute Guilliman]] were among the twenty Primarchs.
Each led a Legion of [[Space Marine]]s.</text>
    </revision>
  </page>
  <page>
    <id>116</id>
    <ns>0</ns>
    <title>Horus Heresy</title>
    <revision>
      <timestamp>2024-01-20T15:30:00Z</timestamp>
      <text>==The Horus Heresy==
The [[Horus Heresy]] was a galaxy-spanning civil war in the 31st millennium.

[[Horus]], once the favored son, turned against the [[Emperor]].
Loyal Primarchs like [[Sanguinius]] fought to defend the [[Imperium]].</text>
    </revision>
  </page>
  <page>
    <id>91375</id>
    <ns>0</ns>
    <title>Emperor of Mankind</title>
    <revision>
      <timestamp>2024-01-21T08:00:00Z</timestamp>
      <text>==The Emperor==
The [[Emperor of Mankind]] is the immortal ruler of the [[Imperium]].

He created the twenty [[Primarch]]s and led the [[Great Crusade]].
Now he sits upon the [[Golden Throne]] on [[Terra]].</text>
    </revision>
  </page>
  <page>
    <id>91130</id>
    <ns>0</ns>
    <title>Imperium</title>
    <revision>
      <timestamp>2024-01-22T12:00:00Z</timestamp>
      <text>==Imperium of Man==
The [[Imperium]] of Man spans a million worlds across the galaxy.

The [[Emperor]] rules from [[Terra]], protected by the [[Adeptus Custodes]].
[[Space Marine]]s defend humanity from countless threats.</text>
    </revision>
  </page>
  <page>
    <id>16481</id>
    <ns>0</ns>
    <title>Roboute Guilliman</title>
    <revision>
      <timestamp>2024-01-23T14:00:00Z</timestamp>
      <text>==Roboute Guilliman==
[[Roboute Guilliman]] was the [[Primarch]] of the [[Ultramarines]].

He authored the [[Codex Astartes]] and ruled the [[Ultramar]] realm.
He was resurrected in the 41st millennium to lead the [[Imperium]].</text>
    </revision>
  </page>
  <page>
    <id>2579</id>
    <ns>0</ns>
    <title>Baal</title>
    <revision>
      <timestamp>2024-01-24T16:00:00Z</timestamp>
      <text>==Baal==
[[Baal]] is the homeworld of the [[Blood Angels]] [[Space Marine]] chapter.

It is a radioactive death world with two moons: [[Baal Primus]] and [[Baal Secundus]].
The chapter fortress-monastery is located here.</text>
    </revision>
  </page>
</mediawiki>
"""

    output_path.write_text(xml_content)
    print(f"Created synthetic XML at {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    """Main entry point for XML extraction script."""
    # Determine source XML path
    source_xml = Path("data/warhammer40k_pages_current.xml")
    output_path = Path("test-data/e2e-test-wiki.xml")

    # Check if source exists
    if source_xml.exists():
        print("Found source XML, extracting real pages...")
        extract_minimal_xml(source_xml, TARGET_PAGE_IDS, output_path)
    else:
        print(f"Source XML not found at {source_xml}")
        print("Creating synthetic XML instead...")
        create_synthetic_xml(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
