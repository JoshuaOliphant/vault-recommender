# vault-recommender

Semantic recommendation engine for Obsidian vaults. Embeddings + wiki-link graph boosting with MCP server for Claude Code integration.

## Quick Start

```bash
uv sync              # install deps
uv run pytest -q     # run tests (34 tests, ~3s)
```

## Architecture

```
vault_recommender/
├── parser.py        # Markdown/frontmatter parsing, wiki-link extraction
├── indexer.py       # Sentence-transformer embeddings (all-MiniLM-L6-v2), numpy persistence
├── graph.py         # Bidirectional wiki-link adjacency graph
├── recommender.py   # Core engine: cosine similarity + graph boost + staleness boost
├── cli.py           # CLI entrypoint (vault-recommender command)
├── server.py        # HTTP server (Starlette/uvicorn) for fast hook integration
└── mcp_server.py    # FastMCP server exposing 3 tools
```

**Scoring signals**: semantic similarity, link graph boost (1-hop + 2-hop), staleness boost (30+ day untouched notes).

## Key Commands

```bash
uv run vault-recommender --vault /path index          # build index
uv run vault-recommender --vault /path recommend      # query
uv run vault-recommender --vault /path serve           # HTTP server on :7532
uv run pytest -q                                      # tests
```

## Stack

- Python 3.12+, uv for package management
- sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- scikit-learn (cosine similarity)
- FastMCP (MCP server)
- PyYAML (frontmatter parsing)
- pytest (testing)

## Dos

- Use `uv` for all package management
- Follow TDD — write tests first
- Keep scoring logic in `recommender.py`
- Index files persist in `.vault-recommender-index/`

## Don'ts

- Don't mock the embedding model — tests use real embeddings
- Don't add JavaScript dependencies — this is pure Python
- Don't change the MCP tool signatures without updating README
