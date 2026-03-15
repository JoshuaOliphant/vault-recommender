# vault-recommender

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
git clone https://github.com/JoshuaOliphant/vault-recommender.git
cd vault-recommender
uv sync
```

## Usage

### CLI

```bash
# Build the index (run once, re-run when vault changes significantly)
uv run vault-recommender --vault /path/to/vault index

# Recommend by topic
uv run vault-recommender --vault /path/to/vault recommend --topic "career transition strategies"

# Recommend notes similar to a specific note
uv run vault-recommender --vault /path/to/vault recommend --note "areas/career/plan.md"

# Find missing connections (similar but not linked)
uv run vault-recommender --vault /path/to/vault recommend --note "areas/career/plan.md" --exclude-linked

# JSON output (for LLM consumption)
uv run vault-recommender --vault /path/to/vault recommend --topic "python testing" --json
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

This exposes three tools:
- `recommend_by_topic` — open-ended semantic search
- `recommend_by_note` — "notes like this one"
- `find_missing_connections` — similar but unlinked notes

### Python API

```python
from pathlib import Path
from vault_recommender.parser import parse_vault
from vault_recommender.indexer import build_index
from vault_recommender.graph import build_graph
from vault_recommender.recommender import VaultRecommender

vault = Path("/path/to/vault")
notes = parse_vault(vault)
index = build_index(notes)
graph = build_graph(notes)

rec = VaultRecommender(index=index, graph=graph, vault_path=vault)

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

## Requirements

- Python 3.12+
- An Obsidian vault (or any directory of markdown files with `[[wiki-links]]`)

## License

MIT
