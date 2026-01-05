# Chunk Schema Refactoring Plan

## Summary

Refactor the ChromaDB chunk schema to remove unused fields and add useful data that's currently discarded.

## Changes Overview

| Change | Type | Rationale |
|--------|------|-----------|
| Remove `spoiler_flag` | Delete | No source data in wiki XML; handle at LLM response time |
| Remove `faction` | Delete | Never populated; infobox data was discarded |
| Remove `era` | Delete | Never populated; infobox data was discarded |
| Remove `content_type` | Delete | Always defaults to "lore"; no meaningful distinction |
| Add `links` | Add | Extract internal wiki links per-chunk for graph traversal/context |
| Extract infobox as chunk | Add | Store structured infobox data instead of discarding |

---

## Detailed Changes

### 1. Remove Unused Metadata Fields and Dead Code

**Files to modify:**
- `src/rag/vector_store.py` - Remove from `ChunkMetadata` TypedDict and `_chunk_to_metadata()`/`_metadata_to_chunk()` methods
- `src/ingestion/metadata_extractor.py` - **DELETE entirely** or strip to minimal (only character_names and source_books if needed)
- `src/common/constants.py` - **DELETE** FACTIONS, ERAS, CONTENT_TYPE_KEYWORDS, SPOILER_KEYWORDS (dead code)

**Fields to remove:**
```python
# Current ChunkMetadata (lines 54-75 in vector_store.py)
spoiler_flag: bool      # DELETE - no source data, always False
content_type: str       # DELETE - always "lore"
faction: str            # DELETE - never populated
era: str                # DELETE - never populated
```

**Dead code to remove from metadata_extractor.py:**
- `_detect_faction()` - DELETE
- `_detect_eras()` - DELETE
- `_classify_content_type()` - DELETE
- `_detect_spoiler()` - DELETE
- Related constants in `src/common/constants.py`

**New ChunkMetadata:**
```python
class ChunkMetadata(TypedDict, total=False):
    wiki_page_id: str
    article_title: str
    section_path: str  # "Infobox" for infobox chunks
    chunk_index: int
    article_last_updated: str
    links: list[str]  # NEW - chunk-level links
```

---

### 2. Add Links Field (Chunk-Level)

**Files to modify:**
- `src/rag/vector_store.py` - Add `links` to `ChunkMetadata`
- `src/ingestion/text_chunker.py` - Extract links from each chunk's text
- `src/ingestion/pipeline.py` - Pass chunk-level links to metadata
- `src/ingestion/models.py` - Add `links` field to `Chunk` model

**Implementation:**
1. Use `extract_internal_links()` on each chunk's text (not article-level)
2. Store links in `Chunk.links: list[str]`
3. Each chunk stores only the links found within its own text
4. Infobox chunk will have its own links extracted from infobox fields

**Note:** Links are chunk-level (each chunk has its own links from its text content).

---

### 3. Extract Infobox as Separate Chunk

**Files to modify:**
- `src/ingestion/wiki_xml_parser.py` - Extract infobox before removing templates
- `src/ingestion/text_chunker.py` - Handle infobox as special first chunk
- `src/ingestion/models.py` - Add `infobox` field to `WikiArticle`

**Implementation:**

#### Step 3.1: Extract infobox in wiki_xml_parser.py

```python
def _extract_infobox(self, parsed: mwparserfromhell.wikicode.Wikicode) -> tuple[str | None, list[str]]:
    """Extract infobox template and its links.

    Returns:
        Tuple of (infobox_text, infobox_links)
    """
    for template in parsed.filter_templates():
        template_name = str(template.name).strip()
        if template_name.startswith("Infobox"):
            # Convert to readable format
            infobox_text = self._format_infobox(template)
            # Extract links from infobox
            infobox_links = self._extract_links_from_template(template)
            return infobox_text, infobox_links
    return None, []
```

#### Step 3.2: Format infobox as readable text

Convert from:
```
{{Infobox Chapter
| name = Black Templars
| Primarch = [[Rogal Dorn]]
| Founding = [[Second Founding]]
}}
```

To:
```
## Infobox: Chapter

- **Name**: Black Templars
- **Primarch**: Rogal Dorn
- **Founding**: Second Founding
```

#### Step 3.3: Store as chunk_index=0 with section_path="Infobox"

**Infobox detection strategy:** Use `section_path = "Infobox"` to identify infobox chunks.

- **With infobox:** chunk 0 has `section_path = "Infobox"`, chunks 1+ have normal section paths
- **Without infobox:** chunk 0 has normal `section_path`, no "Infobox" exists

**To find all infoboxes:** `WHERE section_path = "Infobox"`
**To check if article has infobox:** Query for chunks with `wiki_page_id = X AND section_path = "Infobox"`

All chunk indices remain sequential starting from 0.

---

## Files Changed Summary

| File | Changes | Status |
|------|---------|--------|
| `src/rag/vector_store.py` | Remove 4 metadata fields, add `links` field | ✅ Done |
| `src/ingestion/wiki_xml_parser.py` | Add infobox extraction before template removal | ✅ Done |
| `src/ingestion/text_chunker.py` | Handle infobox as first chunk, extract per-chunk links | ✅ Done |
| `src/ingestion/models.py` | Add `links` to `Chunk`, add `infobox` to `WikiArticle` | ✅ Done |
| `src/ingestion/pipeline.py` | Pass links and infobox through pipeline | ✅ Done |
| `src/ingestion/metadata_extractor.py` | Simplified (removed faction/era/spoiler/content_type) | ✅ Done |
| `src/common/constants.py` | Deleted dead constants (FACTIONS, ERAS, etc.) | ✅ Done |
| `tests/unit/test_metadata_extractor.py` | Updated tests for simplified functionality | ✅ Done |
| `tests/unit/test_vector_store.py` | Updated tests for new schema | ✅ Done |
| `tests/integration/test_metadata_extraction.py` | Updated tests for simplified functionality | ✅ Done |
| `tests/unit/test_text_chunker.py` | Added tests for link extraction and infobox handling | ✅ Done |
| `tests/unit/test_wiki_xml_parser.py` | Added tests for infobox extraction | ✅ Done |

---

## Migration Considerations

- **Existing data**: Will need re-ingestion after schema change
- **ChromaDB**: No formal schema migration; old fields will be ignored on new writes
- **Backwards compatibility**: Not required (development phase)

---

## Testing

1. Unit tests for infobox extraction (various infobox types)
2. Unit tests for link extraction from infobox fields
3. Unit tests for chunk-level link extraction
4. Integration test: verify infobox appears as chunk_index=0
5. Verify removed fields no longer appear in stored chunks
6. Verify links field populated correctly per-chunk

---

## Documentation Updates Required

The following documents reference the old schema and need updating:

### Stories to Update

| Document | Changes Needed |
|----------|---------------|
| `docs/stories/1.8.chroma-vector-database.md` | Update ChunkData schema, remove faction/era/spoiler filtering examples, add links field |
| `docs/stories/1.9.metadata-extraction.md` | Remove faction/era/spoiler/content_type extraction (entire story may become obsolete or simplified) |
| `docs/stories/1.4.wiki-xml-parser.md` | Add infobox extraction to acceptance criteria |

### Other Docs to Update

| Document | Changes Needed |
|----------|---------------|
| `docs/wiki-xml-structure-analysis.md` | Update "Extract internal links for metadata" section, add infobox extraction notes |
| `docs/architecture.md` | Update chunk metadata description if present |
| `CLAUDE.md` | Verify chunk schema description is current |

### New Documentation to Create

| Document | Purpose |
|----------|---------|
| `docs/stories/1.15-chunk-schema-refactoring.md` | New story for this refactoring work |

---

## Implementation Order

1. **Phase 1: Remove unused fields and dead code** ✅ COMPLETED
   - [x] Delete fields from `ChunkMetadata` TypedDict (`spoiler_flag`, `faction`, `era`, `content_type`)
   - [x] Update `_chunk_to_metadata()` and `_metadata_to_chunk()` methods
   - [x] Simplify `metadata_extractor.py` (removed `_detect_faction`, `_detect_eras`, `_classify_content_type`, `_detect_spoiler`)
   - [x] Delete dead constants from `src/common/constants.py` (FACTIONS, ERAS, CONTENT_TYPE_KEYWORDS, SPOILER_KEYWORDS)
   - [x] Update/remove related tests (`test_metadata_extractor.py`, `test_vector_store.py`, `test_metadata_extraction.py`)
   - [x] Add `links` field to `ChunkMetadata` (stored as JSON string for Chroma compatibility)

2. **Phase 2: Add chunk-level links** ✅ COMPLETED
   - [x] Add `links` field to `Chunk` model
   - [x] Extract links during chunking in `text_chunker.py`
   - [x] Add links storage in ChromaDB metadata (vector_store.py done)
   - [x] Update pipeline.py to pass links to ChunkData
   - [x] Update tests

3. **Phase 3: Extract infobox as chunk** ✅ COMPLETED
   - [x] Add infobox extraction to `wiki_xml_parser.py`
   - [x] Format infobox as readable markdown
   - [x] Extract infobox links
   - [x] Add `infobox` and `infobox_links` fields to `WikiArticle` model
   - [x] Handle infobox as first chunk in `text_chunker.py`
   - [x] Store as chunk_index=0 with `section_path="Infobox"`
   - [x] Content chunks follow at indices 1, 2, 3... (if infobox) or 0, 1, 2... (if no infobox)
   - [x] Update tests

4. **Phase 4: Update documentation** ⏳ PENDING
   - [ ] Update story files
   - [ ] Update architecture docs
   - [ ] Create new story document

5. **Phase 5: Re-ingestion** ⏳ PENDING
   - [ ] Purge existing ChromaDB data
   - [ ] Re-run ingestion pipeline with new schema
