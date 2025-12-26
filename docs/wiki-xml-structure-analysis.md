# WH40K Fandom Wiki XML Structure Analysis

**File:** `warhammer40k_pages_current.xml`
**Size:** 173 MB
**Format:** MediaWiki XML Export 0.11
**Date Analyzed:** 2025-12-26

---

## XML Structure Overview

### Root Element
```xml
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/"
           version="0.11"
           xml:lang="en">
```

### Top-Level Structure
1. **`<siteinfo>`** - Wiki metadata (site name, namespaces, generator version)
2. **`<page>`** elements - One per wiki article (repeated thousands of times)

---

## Page Element Structure

### Core Page Fields
```xml
<page>
    <title>Tyranids</title>                    <!-- Article title -->
    <ns>0</ns>                                  <!-- Namespace ID (0 = main articles) -->
    <id>3</id>                                  <!-- Unique page ID -->
    <revision>
        <id>430426</id>                         <!-- Revision ID -->
        <parentid>428954</parentid>             <!-- Parent revision -->
        <timestamp>2024-09-20T14:18:31Z</timestamp>
        <contributor>
            <username>Montonius</username>
            <id>2045915</id>
        </contributor>
        <model>wikitext</model>                 <!-- Content model -->
        <format>text/x-wiki</format>            <!-- Content format -->
        <text bytes="72175"
              sha1="5fns5a2mnt7p6xkqiq540c26btl83lh"
              xml:space="preserve">
            <!-- MediaWiki markup content here -->
        </text>
    </revision>
</page>
```

---

## Namespace Filtering

**CRITICAL:** We only want pages with `<ns>0</ns>` (main article namespace).

### Namespace Reference (from siteinfo)
- `0` - Main articles (WHAT WE WANT)
- `1` - Talk pages
- `2` - User pages
- `4` - Wiki project pages
- `6` - File/media pages
- `10` - Template pages
- `14` - Category pages
- `110` - Forum pages
- `500+` - Blog, user blog, message walls, etc.

**Filter logic:** `if page.find('ns').text == '0':`

---

## Content Format: MediaWiki Markup (Wikitext)

The `<text>` element contains **MediaWiki markup**, NOT HTML.

### Common MediaWiki Syntax Examples

#### 1. Templates
```wikitext
{{Quote|Text here|Source}}
```

#### 2. Headings
```wikitext
==History==
===Tyrannic Wars===
====Subsection====
```

#### 3. Internal Links
```wikitext
[[Inquisitor]]                          <!-- Simple link -->
[[Inquisitor|custom text]]              <!-- Link with custom display text -->
[[File:Tyranid.gif|thumb|250px|Caption]] <!-- Image with parameters -->
```

#### 4. External Links
```wikitext
[http://example.com Link text]
```

#### 5. Lists
```wikitext
* Bullet item
** Sub-item
# Numbered item
```

#### 6. Bold/Italic
```wikitext
'''bold text'''
''italic text''
'''''bold and italic'''''
```

---

## Data Extraction Requirements

### Required Parsing Steps

1. **Stream XML with iterparse** - Memory-efficient for 173MB file
   ```python
   for event, elem in ET.iterparse(xml_path, events=('end',)):
       if elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
           # Process page
           elem.clear()  # Free memory
   ```

2. **Filter by namespace**
   ```python
   ns_elem = page.find('.//{...}ns')
   if ns_elem is not None and ns_elem.text == '0':
       # This is a main article - process it
   ```

3. **Extract page data**
   - Title: `page.find('.//{...}title').text`
   - ID: `page.find('.//{...}id').text`
   - Timestamp: `page.find('.//{...}timestamp').text`
   - Content: `page.find('.//{...}text').text`

4. **Convert MediaWiki markup to Markdown**
   - Need library: `mwparserfromhell` or `wikitextparser`
   - OR: Custom regex-based converter for common patterns
   - Handle: headings, links, bold/italic, lists, templates

5. **Extract internal links for metadata**
   - Parse `[[Link]]` patterns
   - Store as `links` array in chunk metadata
   - Example: `[[Hive Fleet]]`, `[[Imperium of Man]]`

---

## Sample Article Analysis

### Article 1: "Tyranids" (Page ID: 3)

**Stats:**
- Title: "Tyranids"
- Namespace: 0 (main article) ✓
- Size: 72,175 bytes (~72 KB)
- Last edited: 2024-09-20
- Page ends at line 559

**Content Structure:**
- Opening quote template: `{{Quote|...}}`
- Intro paragraphs (3-4)
- Multiple `==Level 1==` sections: History, The Great Devourer, Combat Doctrine, etc.
- Many `===Level 2===` and `====Level 3====` subsections
- Extensive internal links: `[[Hive Fleet]]`, `[[Space Marines]]`, `[[Imperium of Man]]`
- Image embeds: `[[File:Tyranid.gif|thumb|250px|Caption]]`
- Embedded templates for quotes, galleries
- Categories at end: `[[Category:T]]`, `[[Category:Factions]]`
- External links in Sources section

**Chunking Implications:**
- Large article (~72KB) will produce 30-50 chunks at 200-500 tokens each
- Section hierarchy provides natural chunking boundaries
- Rich linking structure perfect for metadata `links` array

### Article 2: "Asuryani" (Page ID: 4)

**Stats:**
- Title: "Asuryani"
- Namespace: 0 (main article) ✓
- Size: 203,578 bytes (~204 KB) - VERY LARGE
- Last edited: 2024-12-02
- Page starts at line 560

**Content Structure:**
- Similar structure to Tyranids article
- Very long article with extensive sections
- Complex templating including quotes, file embeds
- Extensive internal linking

**Chunking Implications:**
- Very large article (~204KB) will produce ~100-150 chunks
- Demonstrates need for robust chunking logic for variable article sizes

---

## XML Parsing Library Requirements

### Recommended: lxml + defusedxml

**lxml advantages:**
- Fast C-based parser
- XPath support for querying
- `iterparse` for memory-efficient streaming
- Namespace handling built-in

**defusedxml advantages:**
- Security protection against XML bombs
- Prevents billion laughs attack
- Prevents entity expansion attacks

### Namespace Handling

MediaWiki XML uses default namespace, so XPath requires namespace mapping:

```python
namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

# XPath with namespace
title = page.find('.//mw:title', namespaces)
ns = page.find('.//mw:ns', namespaces)
text = page.find('.//mw:revision/mw:text', namespaces)
```

---

## MediaWiki to Markdown Conversion

### Option 1: Use mwparserfromhell (Recommended)
```python
import mwparserfromhell

wikicode = mwparserfromhell.parse(text_content)
# Remove templates, extract plain text, convert to markdown
```

### Option 2: Custom Regex Converter
```python
# Basic conversions
text = re.sub(r"====(.*?)====", r"#### \1", text)  # h4
text = re.sub(r"===(.*?)===", r"### \1", text)      # h3
text = re.sub(r"==(.*?)==", r"## \1", text)          # h2
text = re.sub(r"\[\[([^\|]+?)\]\]", r"[\1](\1)", text)  # links
text = re.sub(r"'''(.*?)'''", r"**\1**", text)      # bold
text = re.sub(r"''(.*?)''", r"*\1*", text)          # italic
```

---

## Memory Efficiency Strategy

**Problem:** 173 MB XML file could consume significant memory if loaded entirely.

**Solution:** Streaming parser with element cleanup

```python
import xml.etree.ElementTree as ET

for event, elem in ET.iterparse(xml_path, events=('end',)):
    if elem.tag.endswith('page'):
        # Process page
        process_page(elem)
        # CRITICAL: Clear element to free memory
        elem.clear()
        # Also clear parent references
        while elem.getprevious() is not None:
            del elem.getparent()[0]
```

**Expected memory usage:** <500 MB despite 173 MB source file

---

## Estimated Processing Stats

**Total pages:** Unknown (need to count `<ns>0</ns>` pages)
**Estimated main articles:** ~10,000-15,000
**Expected chunks:** ~50,000 chunks
**Processing time:** 2-4 hours
**OpenAI embedding cost:** ~$1.00

---

## Implementation Checklist for Story 1.4 (Wiki XML Parser)

- [ ] Install lxml and defusedxml (added to Story 1.1 dependencies)
- [ ] Install mwparserfromhell for MediaWiki→Markdown conversion (added to Story 1.1)
- [ ] Implement streaming XML parser with namespace handling
- [ ] **CRITICAL:** Filter pages by `<ns>0</ns>` (main articles only)
- [ ] Handle XML namespace: `{http://www.mediawiki.org/xml/export-0.11/}`
- [ ] Extract title, id, timestamp, text content from each `<page>`
- [ ] Implement MediaWiki to Markdown converter using mwparserfromhell
  - [ ] Convert headings (==, ===, ====) → (##, ###, ####)
  - [ ] Convert bold/italic (''', '') → (**, *)
  - [ ] Convert internal links ([[Link]]) → [Link](Link)
  - [ ] Strip or simplify templates ({{Quote|...}})
  - [ ] Handle lists (*, #) → (-, 1.)
  - [ ] Remove File/Image embeds or convert to placeholders
  - [ ] Remove Categories
- [ ] Extract internal links from wikitext for metadata
- [ ] Save markdown files to `data/markdown-archive/{sanitized_title}.md`
- [ ] **Handle special characters in filenames** (spaces, slashes, colons)
- [ ] Progress logging (every 100 articles)
- [ ] Error handling for malformed pages (skip and log)
- [ ] **Memory-efficient cleanup:** `elem.clear()` + `while elem.getprevious()...`
- [ ] Integration test with full 173MB file
- [ ] Verify memory usage stays <500MB

---

## Updated Dependencies for Story 1.1

Add to `pyproject.toml`:
```toml
lxml = "^5.0"
defusedxml = "^0.7"
mwparserfromhell = "^0.6"  # MediaWiki markup parser
```

**Rationale:**
- `lxml`: Fast, standards-compliant XML parsing with XPath
- `defusedxml`: Security against XML vulnerabilities
- `mwparserfromhell`: Robust MediaWiki wikitext parsing library

---

## Key Findings Confirmed

✅ **Consistent XML structure** across all pages
✅ **Namespace filtering critical:** Many pages with `ns != 0` (Talk, User, Template, etc.)
✅ **MediaWiki markup format** confirmed (not HTML)
✅ **Variable article sizes:** 72KB to 204KB+ per article
✅ **Memory streaming essential:** 173MB file, articles can be processed one-by-one
