# CLI Commands

## parse-wiki

Parse MediaWiki XML exports and convert articles to markdown.

```bash
poetry run parse-wiki <xml_path> [--page-ids-file PATH]
```

**Arguments:**
- `xml_path` - Path to MediaWiki XML export file (required)

**Options:**
- `--page-ids-file PATH` - File containing page IDs to filter (one per line)

## build-test-bed

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
