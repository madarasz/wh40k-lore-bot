# Epic 1: Foundation & Data Pipeline

**Epic Goal:** Establish the foundational infrastructure and implement a complete data ingestion pipeline that can parse the WH40K Fandom Wiki XML export, convert it to markdown, chunk the content, generate embeddings, and store everything in a vector database.

**Value Proposition:** Enables the RAG system to have access to searchable WH40K lore data, which is the core requirement for all subsequent features.

**Dependencies:** None (foundational epic)

**Estimated Effort:** 18-25 hours (12 stories)

---

## Story 1.1: Project Setup & Core Infrastructure

**Story:** As a developer, I want the project scaffolding, dependencies, and core infrastructure components set up so that I can begin building features on a solid foundation.

**Acceptance Criteria:**
1. Repository initialized with Poetry for dependency management
2. Core dependencies installed: **lxml, defusedxml, mwparserfromhell**, SQLAlchemy 2.0+, Alembic, structlog, python-dotenv, pytest
3. Development tools installed: **ruff** (linter and formatter), **mypy** (static type checking)
4. Project structure created following architecture:
   ```
   wh40k-lore-bot/
   ├── src/
   │   ├── ingestion/       # Wiki parsing & chunking
   │   ├── rag/             # RAG query engine
   │   ├── discord_bot/     # Discord integration
   │   ├── trivia/          # Trivia system
   │   └── common/          # Shared utilities
   ├── data/
   │   ├── markdown-archive/
   │   └── chroma-db/
   ├── tests/
   ├── docs/
   └── pyproject.toml
   ```
5. Environment configuration with `.env.template` file containing:
   - `DISCORD_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `DATABASE_URL`
   - `LOG_LEVEL`
6. Logging configured using structlog with JSON formatting
7. SQLite database initialized with Alembic migrations
8. `README.md` created with setup instructions
9. Pre-commit hooks configured (ruff, mypy)

**Technical Notes:**
- Use Poetry 1.7+ for dependency management
- Python 3.11+ required
- SQLAlchemy 2.0+ (type-safe ORM, async support)
- SQLite for MVP (migrations support future PostgreSQL upgrade)
- structlog for structured logging with context injection
- ruff replaces black+isort+flake8 for faster linting and formatting

**Estimated Effort:** 2-3 hours

---

## Story 1.2: Database Schema Design & Implementation

**Story:** As a developer, I want the database schema defined and implemented so that I can store wiki chunks, embeddings, and metadata efficiently.

**Acceptance Criteria:**
1. Alembic migration created for initial schema
2. `WikiChunk` table created with columns:
   - `id` (UUID primary key)
   - `wiki_page_id` (string, indexed - from wiki XML export)
   - `article_title` (string, indexed)
   - `section_path` (string, e.g., "History > Horus Heresy")
   - `chunk_text` (text)
   - `chunk_index` (integer)
   - `metadata_json` (JSON)
   - `created_at` (timestamp)
   - `updated_at` (timestamp)
3. Metadata JSON schema supports:
   - `faction`: Optional[str]
   - `subfaction`: Optional[str]
   - `character_names`: List[str]
   - `era`: Optional[str]
   - `spoiler_flag`: bool
   - `content_type`: str
   - `links`: List[str]
   - `source_books`: List[str]
4. Indexes created on: `article_title`, `faction`, `era`, `spoiler_flag`
5. SQLAlchemy ORM models created in `src/common/models.py`
6. Repository pattern implemented in `src/common/repositories/chunk_repository.py`
7. Unit tests for CRUD operations

**Technical Notes:**
- Use UUID for chunk IDs (future-proof for distributed systems)
- JSON column for flexible metadata (easy to extend)
- Indexes optimized for common queries (faction filter, spoiler filter)

**Estimated Effort:** 2-3 hours

---

## Story 1.3: Markdown Archive Setup

**Story:** As a developer, I want a structured directory for storing markdown files extracted from the wiki so that I have a persistent, human-readable archive.

**Acceptance Criteria:**
1. `data/markdown-archive/` directory created
2. File naming convention: `{sanitized_article_title}.md`
3. Filename sanitization utility handles:
   - Spaces → underscores
   - Slashes → hyphens
   - Colons → hyphens
   - Special characters removed
   - Unicode normalized
4. Markdown files include frontmatter with metadata:
   ```yaml
   ---
   title: "Tyranids"
   wiki_id: "3"
   last_updated: "2024-09-20T14:18:31Z"
   word_count: 12453
   ---
   ```
5. Utility function `save_markdown_file(article: WikiArticle) -> Path` implemented
6. Unit tests verify filename sanitization edge cases

**Technical Notes:**
- Use `pathlib.Path` for cross-platform compatibility
- YAML frontmatter for metadata (human-readable)
- Git ignore `data/` directory (too large for version control)

**Estimated Effort:** 1-2 hours

---

## Story 1.4: Wiki XML Parser Implementation

**Story:** As a developer, I want to parse the WH40K Fandom Wiki XML export and convert articles to markdown so that I have clean, structured content for chunking and embedding.

**Acceptance Criteria:**
1. `src/ingestion/wiki_xml_parser.py` created with `WikiXMLParser` class
2. Parser uses lxml's iterparse for memory-efficient streaming of large XML files
3. **CRITICAL: Namespace filtering - Only process pages with `<ns>0</ns>` (main articles)**
4. XML namespace handling: `{http://www.mediawiki.org/xml/export-0.11/}`
5. `parse_xml_export(xml_path: str, page_ids: Optional[List[str]] = None)` method extracts article elements using XPath
6. **Page ID filtering:** If `page_ids` list provided, only process pages whose `<id>` is in the list
7. Each article extracts: **wiki page id**, title, timestamp, content (MediaWiki markup)
8. **MediaWiki markup converted to Markdown using mwparserfromhell library:**
   - Headings: `==Text==` → `## Text`, `===Text===` → `### Text`, `====Text====` → `#### Text`
   - Bold: `'''text'''` → `**text**`
   - Italic: `''text''` → `*text*`
   - Internal links: `[[Link]]` → `[Link](Link)` and `[[Link|text]]` → `[text](Link)`
   - External links: `[http://example.com text]` → `[text](http://example.com)`
   - Lists: `* item` → `- item`, `# item` → `1. item`
   - Strip or simplify templates (e.g., `{{Quote|...}}` → plain text or removed)
   - Remove File/Image embeds: `[[File:...]]` removed or converted to `[Image: filename]`
   - Remove Categories: `[[Category:...]]` removed
9. Extract internal links from wikitext for metadata `links` array
10. `save_articles_batch()` writes markdown files to `data/markdown-archive/{sanitized_title}.md`
11. **Sanitize filenames** (handle spaces, slashes, colons, special characters)
12. Parser handles malformed XML gracefully with defusedxml security protection
13. **Memory cleanup: `elem.clear()` after processing each page**
14. Progress logging every 100 articles parsed (e.g., "Parsed 500/10000 articles...")
15. Integration test with full 173MB XML file verifies end-to-end parsing
16. Parses 173MB XML file in <500MB memory footprint
17. CLI command supports: `poetry run parse-wiki data/warhammer40k_pages_current.xml [--page-ids-file pages.txt]`

**Technical Notes:**
- Use `lxml.etree.iterparse()` with `events=('end',)` for streaming
- MediaWiki namespace: `http://www.mediawiki.org/xml/export-0.11/`
- `mwparserfromhell` for robust MediaWiki markup parsing
- `defusedxml` to prevent XML bomb attacks
- Estimated processing time: 2-4 hours for full wiki
- Estimated article count: ~10,000-15,000 main articles

**Sample Code Pattern:**
```python
import xml.etree.ElementTree as ET
from defusedxml import ElementTree as DefusedET
import mwparserfromhell

namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

for event, elem in ET.iterparse(xml_path, events=('end',)):
    if elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
        ns_elem = elem.find('mw:ns', namespaces)
        if ns_elem is not None and ns_elem.text == '0':
            # Process main article
            title = elem.find('mw:title', namespaces).text
            text = elem.find('.//mw:revision/mw:text', namespaces).text

            # Convert MediaWiki to Markdown
            wikicode = mwparserfromhell.parse(text)
            markdown = convert_to_markdown(wikicode)

        # Free memory
        elem.clear()
```

**Estimated Effort:** 4-6 hours

---

## Story 1.5: Test Bed Page Selection (100-Page Dataset)

**Story:** As a developer, I want to create a curated list of ~100 wiki pages starting from Blood Angels and expanding to related pages so that I can fine-tune and test the RAG pipeline before processing the full 10,000+ page dataset.

**Acceptance Criteria:**
1. `src/ingestion/test_bed_builder.py` created with `TestBedBuilder` class
2. Starting page: **Blood Angels** (wiki page id: `58`)
3. Expansion strategy implements breadth-first traversal:
   - Start with seed page (Blood Angels)
   - Extract all internal wiki links from the page
   - Add linked pages to queue
   - Process queue until ~100 pages collected
   - Prioritize pages with most incoming links (hub pages)
4. `build_test_bed(xml_path: str, seed_page_id: str, target_count: int = 100) -> List[str]` method
5. Link extraction from MediaWiki markup:
   - Parse `[[Internal Link]]` and `[[Link|Display Text]]` patterns
   - Resolve link targets to wiki page IDs using XML lookup
   - Track link frequency for prioritization
6. Output saved to: `data/test-bed-pages.txt` (one page ID per line)
7. Include page titles as comments in output file for readability:
   ```
   # Blood Angels (seed)
   58
   # Sanguinius
   142
   # Horus Heresy
   89
   ...
   ```
8. Statistics logged:
   - Total pages selected
   - Depth of traversal from seed
   - Top 10 most-linked pages
   - Coverage by faction/topic
9. CLI command: `poetry run build-test-bed data/warhammer40k_pages_current.xml --seed-id 58 --count 100`
10. Unit tests verify link extraction and traversal logic
11. Integration test with real wiki XML

**Technical Notes:**
- Use BFS (breadth-first search) to ensure diverse topic coverage near seed
- Track visited pages to avoid duplicates
- Page ID resolution requires first-pass XML scan to build title→ID mapping
- Priority queue based on incoming link count ensures important hub pages included
- Test bed enables faster iteration on chunking, embedding, and retrieval strategies

**Estimated Effort:** 2-3 hours

---

## Story 1.6: Intelligent Text Chunking Implementation

**Story:** As a developer, I want to intelligently chunk markdown articles into semantically coherent pieces so that embeddings capture meaningful context without breaking mid-thought.

**Acceptance Criteria:**
1. `src/ingestion/text_chunker.py` created with `MarkdownChunker` class
2. Chunking strategy:
   - **Primary:** Chunk by section headers (##, ###, ####)
   - **Secondary:** If section > 500 tokens, split on paragraph boundaries
   - **Tertiary:** If paragraph > 500 tokens, split on sentence boundaries
   - **Max chunk size:** 500 tokens (optimal for embeddings)
   - **Min chunk size:** 50 tokens (avoid tiny fragments)
3. Section hierarchy preserved in metadata:
   - `section_path`: "History > Horus Heresy > Battle of Terra"
4. `chunk_markdown(markdown: str, article_title: str) -> List[Chunk]` method
5. Each chunk includes:
   - `chunk_text`: The actual text content
   - `article_title`: Source article
   - `section_path`: Hierarchical section path
   - `chunk_index`: 0-based index within article
6. Tokenizer uses `tiktoken` with `cl100k_base` encoding (OpenAI)
7. Unit tests with sample markdown covering:
   - Single section (no splitting)
   - Multiple sections
   - Long section requiring paragraph split
   - Very long paragraph requiring sentence split
8. Integration test with real wiki article (e.g., "Tyranids" - 72KB)

**Technical Notes:**
- Use `tiktoken` for accurate token counting (matches OpenAI)
- Preserve markdown formatting in chunks (headers, lists, bold)
- Overlap strategy: Include section header in every chunk from that section

**Estimated Effort:** 3-4 hours

---

## Story 1.7: OpenAI Embeddings Integration

**Story:** As a developer, I want to generate vector embeddings for text chunks using OpenAI's API so that I can perform semantic search on the lore data.

**Acceptance Criteria:**
1. `src/ingestion/embedding_generator.py` created with `EmbeddingGenerator` class
2. Uses OpenAI API with `text-embedding-3-small` model
3. Batch processing: Up to 100 chunks per API call (OpenAI limit: 8192 tokens/chunk)
4. `generate_embeddings(chunks: List[str]) -> List[np.ndarray]` method
5. Retry logic with exponential backoff:
   - Max 3 retries
   - Backoff: 2^retry seconds
6. Rate limiting: Max 3000 RPM (requests per minute)
7. Cost tracking: Log total tokens processed and estimated cost
8. Error handling:
   - Network failures → retry
   - Rate limit errors → wait and retry
   - Invalid input → log and skip
9. Unit tests with mocked OpenAI API responses
10. Integration test with real API (small batch of 10 chunks)
11. Environment variable: `OPENAI_API_KEY`

**Technical Notes:**
- OpenAI `text-embedding-3-small`: 1536 dimensions, $0.02 per 1M tokens
- Expected cost for full wiki (~50,000 chunks): ~$1.00
- Use `openai` Python library (v1.0+)
- Store embeddings as `numpy.ndarray` (float32)

**Estimated Effort:** 2-3 hours

---

## Story 1.8: Chroma Vector Database Integration

**Story:** As a developer, I want to store chunk embeddings in Chroma so that I can perform fast similarity searches.

**Acceptance Criteria:**
1. `src/rag/vector_store.py` created with `ChromaVectorStore` class
2. Chroma persistent client initialized:
   - Storage path: `data/chroma-db/`
   - Collection name: `wh40k-lore`
3. Collection metadata schema:
   - `article_title`: string
   - `section_path`: string
   - `chunk_index`: int
   - `faction`: string (optional, for filtering)
   - `era`: string (optional, for filtering)
   - `spoiler_flag`: bool
   - `content_type`: string
4. `add_chunks(chunks: List[Chunk], embeddings: List[np.ndarray])` method
5. Batch insertion: 1000 chunks per batch (Chroma performance)
6. `query(query_embedding: np.ndarray, n_results: int, filters: Dict) -> List[Chunk]` method
7. Metadata filtering support:
   - Filter by faction: `{"faction": "Space Marines"}`
   - Filter by era: `{"era": "Horus Heresy"}`
   - Exclude spoilers: `{"spoiler_flag": False}`
8. Distance metric: Cosine similarity
9. Unit tests with mock Chroma client
10. Integration test: Insert 100 chunks, query with filters

**Technical Notes:**
- Chroma embedded mode (no server required for MVP)
- Collection persists to disk automatically
- Future: Upgrade to Chroma server for production

**Estimated Effort:** 2-3 hours

---

## Story 1.9: Metadata Extraction from Content

**Story:** As a developer, I want to extract metadata (faction, era, character names, links) from chunk content so that I can enable rich filtering and enhance retrieval relevance.

**Acceptance Criteria:**
1. `src/ingestion/metadata_extractor.py` created with `MetadataExtractor` class
2. Faction detection:
   - Keyword matching from predefined list: `FACTIONS = ["Space Marines", "Tyranids", "Orks", ...]`
   - Case-insensitive matching
   - Return most frequent faction mentioned in chunk
3. Era detection:
   - Keyword matching: `ERAS = ["Great Crusade", "Horus Heresy", "Age of Apostasy", ...]`
   - Return all eras mentioned
4. Character name extraction:
   - Use internal wiki links: `[[Roboute Guilliman]]` → "Roboute Guilliman"
   - Limit to top 5 most mentioned
5. Content type classification:
   - "lore" (default)
   - "military" (keywords: battle, war, tactics)
   - "technology" (keywords: weapon, armor, vehicle)
   - "character" (if character names dominate)
6. Spoiler detection:
   - Keyword matching: "spoiler", "recent lore", "9th edition", "10th edition"
   - Default: `False` (Fandom wiki is canon, not spoilers)
7. Source book extraction:
   - Regex for book titles in references: "Codex: Space Marines", "Horus Heresy: Betrayal"
8. `extract_metadata(chunk: Chunk) -> Dict` method returns all metadata fields
9. Unit tests with sample chunks covering each detection case
10. Integration test with real wiki chunks

**Technical Notes:**
- Simple keyword-based approach (no LLM needed for MVP)
- Faction/era lists maintained in `src/common/constants.py`
- Character names extracted from internal links (reliable)

**Estimated Effort:** 2-3 hours

---

## Story 1.10: Complete Ingestion Pipeline Orchestration

**Story:** As a developer, I want an orchestrated pipeline that runs all ingestion steps end-to-end so that I can transform the wiki XML export into a searchable vector database with a single command.

**Acceptance Criteria:**
1. `src/ingestion/pipeline.py` created with `IngestionPipeline` class
2. Pipeline executes steps in order:
   1. Parse XML → markdown articles
   2. Save markdown files to archive
   3. Chunk markdown articles
   4. Extract metadata from chunks
   5. Generate embeddings (batched)
   6. Store chunks + embeddings + metadata in Chroma
3. Progress tracking:
   - Log completion of each major step
   - Track articles processed, chunks created, embeddings generated
   - Display estimated time remaining
4. Resumable pipeline:
   - Track progress in SQLite: `ingestion_progress` table
   - If pipeline fails, resume from last completed batch
5. CLI command: `poetry run ingest-wiki data/warhammer40k_pages_current.xml`
6. CLI options:
   - `--batch-size`: Number of articles per batch (default: 100)
   - `--skip-existing`: Skip already processed articles
   - `--dry-run`: Parse and chunk without generating embeddings
   - `--page-ids-file`: Path to file containing page IDs to process (e.g., `data/test-bed-pages.txt`)
7. Summary report at completion:
   - Total articles processed
   - Total chunks created
   - Total embeddings generated
   - Total cost (OpenAI API)
   - Processing time
8. Error handling:
   - Log errors to `logs/ingestion-errors.log`
   - Continue processing on non-fatal errors
   - Rollback batch on fatal errors
9. Integration test: Full pipeline with small XML sample (5 articles)
10. End-to-end test: Full 173MB XML file (optional, takes 2-4 hours)

**Technical Notes:**
- Use `click` library for CLI
- Batch processing to manage memory (process 100 articles at a time)
- Transactional batches: Commit to DB only after successful embedding generation
- Expected total processing time: 2-4 hours

**Estimated Effort:** 3-4 hours

---

## Story 1.11: Data Quality Validation & Monitoring

**Story:** As a developer, I want data quality checks and monitoring during ingestion so that I can detect and fix issues early.

**Acceptance Criteria:**
1. `src/ingestion/validators.py` created with validation functions
2. Validation checks:
   - **XML validation:** Well-formed XML, expected schema
   - **Markdown quality:** Non-empty content, valid frontmatter
   - **Chunk quality:** Min/max token count, non-empty text
   - **Embedding quality:** Correct dimensions (1536), no NaN values
   - **Metadata completeness:** Required fields present
3. Validation metrics logged:
   - Articles skipped (malformed XML)
   - Chunks discarded (too short/long)
   - Embedding failures (API errors)
   - Metadata extraction failures
4. Quality report generated:
   - `data/quality-report.json` with summary statistics
5. Alerting for critical issues:
   - >5% articles skipped → warning
   - >10% embeddings failed → error
6. Unit tests for each validation function
7. Integration test with intentionally malformed data

**Technical Notes:**
- Validation should be fast (don't block pipeline)
- Log detailed errors for debugging
- Quality report useful for monitoring data drift over time

**Estimated Effort:** 2-3 hours

---

## Story 1.12: Ingestion Documentation & Usage Guide

**Story:** As a future developer or user, I want comprehensive documentation on the ingestion pipeline so that I can understand how it works and troubleshoot issues.

**Acceptance Criteria:**
1. `docs/ingestion-guide.md` created with sections:
   - **Overview:** High-level architecture diagram
   - **Prerequisites:** Dependencies, environment setup
   - **Usage:** CLI commands with examples
   - **Configuration:** Environment variables, config files
   - **Troubleshooting:** Common errors and solutions
   - **Performance:** Expected processing times, cost estimates
   - **Architecture:** Component diagram, data flow
2. Inline code documentation:
   - Docstrings for all public classes and methods (Google style)
   - Type hints on all function signatures
3. Example configuration file: `.env.example`
4. FAQ section covering:
   - How to create a test bed for RAG fine-tuning?
   - How to ingest only specific pages (via page ID filtering)?
   - How to resume failed ingestion?
   - How to update existing data?
   - How to handle new wiki exports?
   - How much does ingestion cost?
5. Architecture diagram (Mermaid format) showing data flow:
   ```
   XML → Parser → Markdown → Chunker → Chunks
                                         ↓
                                    Metadata Extractor
                                         ↓
                                    Embedding Generator
                                         ↓
                                    Chroma Vector Store
   ```

**Technical Notes:**
- Documentation should be beginner-friendly
- Include cost estimates and performance metrics
- Link to external resources (OpenAI docs, Chroma docs)

**Estimated Effort:** 2 hours

---

## Epic Completion Criteria

**Epic is complete when:**
1. ✅ All 12 stories completed
2. ✅ Test bed (100 pages) successfully created and ingested for RAG fine-tuning
3. ✅ Full wiki XML (173MB) successfully ingested
4. ✅ Vector database contains ~50,000 chunks with embeddings
5. ✅ Markdown archive contains ~10,000-15,000 articles
6. ✅ WikiChunk table includes wiki_page_id field for traceability
7. ✅ Integration tests pass
8. ✅ Documentation complete
9. ✅ Cost estimate validated (~$1.00 for embeddings)
10. ✅ Processing time validated (2-4 hours for full wiki, <5 minutes for test bed)

**Acceptance Test (Test Bed):**
Run `poetry run build-test-bed data/warhammer40k_pages_current.xml --seed-id 58 --count 100` and verify:
- Test bed file created with ~100 page IDs
- Blood Angels (58) is first page
- Related pages included (Sanguinius, Horus Heresy, etc.)

**Acceptance Test (Full Ingestion):**
Run `poetry run ingest-wiki data/warhammer40k_pages_current.xml` and verify:
- No fatal errors
- Quality report shows <5% data issues
- Chroma database queryable via Python API
- Sample query returns relevant results

---

## Dependencies for Next Epics

**Epic 2 (RAG Query System) depends on:**
- ✅ Vector database populated (Story 1.8)
- ✅ Chunk metadata available (Story 1.9)
- ✅ Query interface defined (Story 1.8)
- ✅ Test bed dataset available for fine-tuning (Story 1.5)

**Epic 3 (Discord Bot) depends on:**
- ✅ Query pipeline functional (Epic 2)
- ✅ Database schema stable with wiki_page_id (Story 1.2)
