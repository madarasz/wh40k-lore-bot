# CLI Commands

## Data Preparation Commands

### build-test-bed

Build a test bed of wiki pages using BFS traversal from a seed page.

```bash
poetry run build-test-bed <xml_path> [OPTIONS]
```

**Arguments:**
- `xml_path` - Path to MediaWiki XML export file (required)

**Options:**
- `--seed-id TEXT` - Seed page ID to start traversal (default: "58")
- `--count INTEGER` - Target number of pages (default: 100)
- `--output PATH` - Output file path (default: "data/test-bed-pages.txt")

### parse-wiki

Parse MediaWiki XML exports and convert articles to markdown archive.

Uses two-pass processing to automatically handle wiki redirects:
1. **First pass**: Builds a map of all redirect pages (source → target)
2. **Second pass**: Processes articles with automatic redirect handling
   - Redirect pages are automatically skipped (not saved to archive)
   - Internal links pointing to redirect sources are automatically resolved to their canonical targets
   - HTML entities in redirect targets (e.g., `&#039;` → `'`) are properly decoded

```bash
poetry run parse-wiki <xml_path> [--page-ids-file PATH]
```

**Arguments:**
- `xml_path` - Path to MediaWiki XML export file (required)

**Options:**
- `--page-ids-file PATH` - File containing page IDs to filter (one per line)

**Redirect Handling:**
- Redirect pages are automatically detected and excluded from processing
- Links in article content are automatically updated to point to canonical targets
- Example: A link to `[[Mankind]]` is automatically converted to `[[Humans]]` if "Mankind" redirects to "Humans"
- Display text in links is preserved: `[[Mankind|humanity]]` becomes `[[Humans|humanity]]`
- Statistics are logged: redirects found, redirects skipped, links resolved

---

## Ingestion Commands (Markdown-Based)

### ingest

Full ingestion pipeline - loads markdown from archive, chunks, extracts metadata, generates embeddings, stores in Chroma, and builds BM25 index.

The pipeline automatically creates both vector embeddings (for semantic search) and a BM25 keyword index (for exact term matching) to enable hybrid retrieval.

```bash
poetry run ingest [OPTIONS]
```

**Options:**
- `--archive-path PATH` - Path to markdown archive directory (default: data/markdown-archive)
- `--batch-size INTEGER` - Number of articles to process per batch (default: 100)
- `--wiki-ids-file PATH` - Path to file containing wiki IDs to process (one per line)
- `--dry-run` - Parse and chunk without generating embeddings (useful for testing)
- `--force` - Re-ingest all articles regardless of last_updated timestamp
- `--chroma-path TEXT` - Path to Chroma vector database (default: data/chroma-db/)

**Examples:**
```bash
# Process entire archive
poetry run ingest

# Process specific wiki IDs
poetry run ingest --wiki-ids-file data/test-bed-pages.txt

# Dry run (no embeddings)
poetry run ingest --dry-run

# Force re-ingest all
poetry run ingest --force
```

### chunk

Step 1: Chunk markdown articles from archive into JSON format.

```bash
poetry run chunk [OPTIONS]
```

**Options:**
- `--archive-path PATH` - Path to markdown archive directory (default: data/markdown-archive)
- `--wiki-ids-file PATH` - Path to file containing wiki IDs to process (one per line)
- `--output PATH` - Output file path for chunks JSON (default: data/chunks.json)

### embed

Step 2: Generate embeddings for chunks JSON file.

```bash
poetry run embed <chunks_file> [OPTIONS]
```

**Arguments:**
- `chunks_file` - Path to chunks JSON file (output of `chunk` command)

**Options:**
- `--output PATH` - Output file path for embeddings JSON (default: data/embeddings.json)
- `--batch-size INTEGER` - Number of chunks to embed per batch (default: 100)

### store

Step 3: Store embeddings in Chroma vector database and build BM25 index.

Stores embeddings in ChromaDB for vector similarity search and automatically builds
a BM25 keyword search index for hybrid retrieval. The BM25 index is built after
vector storage completes successfully.

```bash
poetry run store <embeddings_file> [OPTIONS]
```

**Arguments:**
- `embeddings_file` - Path to embeddings JSON file (output of `embed` command)

**Options:**
- `--chroma-path TEXT` - Path to Chroma vector database (default: data/chroma-db/)
- `--batch-size INTEGER` - Number of chunks to store per batch (default: 1000)
- `--force` - Force re-ingestion even if article hasn't changed

**BM25 Indexing:**
- BM25 index is built automatically after vector storage
- Index saved to path specified by `BM25_INDEX_PATH` env var (default: data/bm25-index/bm25_index.pkl)
- If BM25 indexing fails, vector storage still succeeds (graceful degradation)
- Use `build-bm25` command to rebuild index if needed

### build-bm25

Standalone BM25 index builder (optional/debugging).

Build a BM25 keyword search index from chunks JSON file. Useful for debugging
or rebuilding the BM25 index without re-running the full storage pipeline.

```bash
poetry run build-bm25 <chunks_file> [OPTIONS]
```

**Arguments:**
- `chunks_file` - Path to chunks JSON file (output of `chunk` command)

**Options:**
- `--output PATH` - Output path for BM25 index (default: from BM25_INDEX_PATH env var)

**Examples:**
```bash
# Build BM25 index from chunks
poetry run build-bm25 data/chunks.json

# Custom output path
poetry run build-bm25 data/chunks.json --output data/my-bm25-index.pkl
```

---

## Workflow Examples

### Full Pipeline (Recommended)

```bash
poetry run ingest
```

### Test Bed Workflow

```bash
# Step 1: Build test bed file
poetry run build-test-bed data/warhammer40k_pages_current.xml --seed-id 58 --count 100

# Step 2: Parse XML to markdown archive (if not done)
poetry run parse-wiki data/warhammer40k_pages_current.xml

# Step 3: Ingest test bed
poetry run ingest --wiki-ids-file data/test-bed-pages.txt
```

### Step-by-Step Pipeline (Debugging)

```bash
# Step 1: Chunk markdown articles
poetry run chunk --output data/chunks.json

# Step 2: Generate embeddings
poetry run embed data/chunks.json --output data/embeddings.json

# Step 3: Store embeddings and build BM25 index
poetry run store data/embeddings.json

# Optional: Rebuild BM25 index separately (if needed)
poetry run build-bm25 data/chunks.json
```

---

## Database Analysis & Maintenance Commands

### stats-markdown

Display markdown archive statistics including file counts, word counts, and median.

```bash
poetry run stats-markdown [OPTIONS]
```

**Options:**
- `--archive-path PATH` - Path to markdown archive directory (default: data/markdown-archive)

**Output includes:**
- Total .md file count
- Top 10 smallest files by word count
- Top 10 largest files by word count
- Median word count across all files

### stats-db

Display vector database statistics including chunk counts, token counts, and last updated date.

```bash
poetry run stats-db [OPTIONS]
```

**Options:**
- `--chroma-path PATH` - Path to Chroma vector database (default: data/chroma-db/)

**Output includes:**
- Total chunk count in Chroma
- Top 10 smallest chunks by token count
- Top 10 largest chunks by token count
- Most recent `article_last_updated` date

### show-chunk

Display detailed information about a specific chunk.

```bash
poetry run show-chunk <chunk_id> [OPTIONS]
```

**Arguments:**
- `chunk_id` - The chunk ID to look up (format: {wiki_page_id}_{chunk_index}, e.g., "58_0")

**Options:**
- `--chroma-path PATH` - Path to Chroma vector database (default: data/chroma-db/)

**Output includes:**
- Chunk ID, wiki page ID, chunk index
- Article title and section path
- Token count (calculated with tiktoken)
- Full chunk text content
- All metadata (faction, era, spoiler_flag, content_type, etc.)

### delete-chunk

Delete a specific chunk from both Chroma and SQLite stores.

```bash
poetry run delete-chunk <chunk_id> [OPTIONS]
```

**Arguments:**
- `chunk_id` - The chunk ID to delete (format: {wiki_page_id}_{chunk_index})

**Options:**
- `--chroma-path PATH` - Path to Chroma vector database (default: data/chroma-db/)
- `--force` - Skip confirmation prompt

**Examples:**
```bash
# Delete with confirmation prompt
poetry run delete-chunk 58_0

# Delete without confirmation
poetry run delete-chunk 58_0 --force
```

### db-health

Check health and consistency of both SQLite and Chroma databases.

```bash
poetry run db-health [OPTIONS]
```

**Options:**
- `--chroma-path PATH` - Path to Chroma vector database (default: data/chroma-db/)

**Output includes:**
- SQLite connection status and WikiChunk row count
- Chroma connection status, collection name, and chunk count
- Consistency check (Chroma count vs SQLite count match)

### purge-db

Delete ALL chunks from both Chroma and SQLite stores.

```bash
poetry run purge-db [OPTIONS]
```

**Options:**
- `--chroma-path PATH` - Path to Chroma vector database (default: data/chroma-db/)
- `--force` - Skip confirmation prompt (requires typing "DELETE ALL" otherwise)

**Examples:**
```bash
# Purge with confirmation (type "DELETE ALL")
poetry run purge-db

# Purge without confirmation (dangerous!)
poetry run purge-db --force
```

**Warning:** This is a destructive operation that cannot be undone!
