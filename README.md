# vault-recommender

[![CI](https://github.com/JoshuaOliphant/vault-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/JoshuaOliphant/vault-recommender/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vault-recommender)](https://pypi.org/project/vault-recommender/)

Semantic recommendation engine for Obsidian vaults. Uses sentence-transformer embeddings + wiki-link graph boosting to surface related notes, forgotten knowledge, and missing connections.

Designed as a tool for LLMs — returns context-rich results with explanations, not just ranked paths.

## How it works

```
Your vault (markdown files)
       │
   Parser ─── extracts frontmatter, body, wiki-links
       │
   Indexer ─── embeds each note as a 384-dim vector (all-MiniLM-L6-v2)
       │
   Link Graph ─── builds bidirectional wiki-link adjacency
       │
   Recommender ─── cosine similarity + graph boost + staleness boost
       │
   Ranked results with reasons
```

**Three scoring signals:**
- **Semantic similarity** — cosine distance between note embeddings. Catches meaning, not just keywords.
- **Link graph boost** — notes connected through wiki-links get a bump. 2-hop neighbors (connected through a shared link) surface "bridge" connections.
- **Staleness boost** — notes untouched for 30+ days get a small boost. Surfaces forgotten-but-relevant knowledge.

## Installation

```bash
# From PyPI
uv tool install vault-recommender

# Or from source
git clone https://github.com/JoshuaOliphant/vault-recommender.git
cd vault-recommender
uv sync
```

## Usage

### CLI

```bash
# Build the index (run once, re-run when vault changes significantly)
vault-recommender --vault /path/to/vault index

# Recommend by topic
vault-recommender --vault /path/to/vault recommend --topic "career transition strategies"

# Recommend notes similar to a specific note
vault-recommender --vault /path/to/vault recommend --note "areas/career/plan.md"

# Find missing connections (similar but not linked)
vault-recommender --vault /path/to/vault recommend --note "areas/career/plan.md" --exclude-linked

# Auto-rebuild stale index before querying
vault-recommender --vault /path/to/vault recommend --topic "python testing" --rebuild

# JSON output (for LLM consumption)
vault-recommender --vault /path/to/vault recommend --topic "python testing" --json
```

The `--rebuild` flag checks whether any vault file is newer than the index. If so, it rebuilds automatically before querying. If the index is fresh, it skips silently.

### HTTP Server (for hooks and fast queries)

The CLI cold-starts the embedding model on topic queries (~13s). For latency-sensitive use cases like Claude Code hooks, run the HTTP server instead:

```bash
# Start the server (loads index once, then serves fast queries)
vault-recommender --vault /path/to/vault serve

# Custom host/port
vault-recommender --vault /path/to/vault serve --host 0.0.0.0 --port 8000
```

Endpoints:

```bash
# Health check
curl localhost:7532/health

# Recommend by topic
curl "localhost:7532/recommend?topic=career+transition&top_k=5"

# Recommend by note
curl "localhost:7532/recommend?note=areas/career/plan.md&top_k=3"

# Find missing connections
curl "localhost:7532/recommend?note=areas/career/plan.md&exclude_linked=true"

# Hot-reload index after re-indexing via CLI
curl -X POST localhost:7532/reload
```

### MCP Server (Claude Code integration)

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "vault-recommender": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/vault-recommender",
        "python",
        "-m",
        "vault_recommender.mcp_server"
      ],
      "env": {
        "VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

This exposes four tools:
- `recommend_by_topic` — open-ended semantic search
- `recommend_by_note` — "notes like this one"
- `find_missing_connections` — similar but unlinked notes
- `reload_index` — force-reload the index after re-indexing via CLI

### Python API

```python
from pathlib import Path
from vault_recommender.recommender import create_recommender

vault = Path("/path/to/vault")
index_dir = Path(".vault-recommender-index")

rec = create_recommender(vault, index_dir)

# By topic
results = rec.similar_to_topic("career transition")

# By note
results = rec.similar_to_note("areas/career/plan.md")

# Each result has: path, title, score, snippet, tags, reason
for r in results:
    print(f"{r.score:.3f} {r.title} — {r.reason}")
```

## Performance

- ~1,500 notes indexed in ~5 seconds (M-series Mac)
- Queries return in <1 second (after model warm-up)
- Index persists as numpy + JSON (~2MB for 1,500 notes)
- Model: `all-MiniLM-L6-v2` (~80MB, runs on CPU)
- `--help` responds instantly (heavy imports deferred until needed)

## Requirements

- Python 3.12+
- An Obsidian vault (or any directory of markdown files with `[[wiki-links]]`)

## License

MIT
