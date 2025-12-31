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

```bash
poetry run parse-wiki <xml_path> [--page-ids-file PATH]
```

**Arguments:**
- `xml_path` - Path to MediaWiki XML export file (required)

**Options:**
- `--page-ids-file PATH` - File containing page IDs to filter (one per line)

---

## Ingestion Commands (Markdown-Based)

### ingest

Full ingestion pipeline - loads markdown from archive, chunks, extracts metadata, generates embeddings, and stores in Chroma.

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

Step 3: Store embeddings in Chroma vector database.

```bash
poetry run store <embeddings_file> [OPTIONS]
```

**Arguments:**
- `embeddings_file` - Path to embeddings JSON file (output of `embed` command)

**Options:**
- `--chroma-path TEXT` - Path to Chroma vector database (default: data/chroma-db/)
- `--batch-size INTEGER` - Number of chunks to store per batch (default: 1000)

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
poetry run chunk --output data/chunks.json
poetry run embed data/chunks.json --output data/embeddings.json
poetry run store data/embeddings.json
```
