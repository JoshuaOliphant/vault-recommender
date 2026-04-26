# vault-recommender

Semantic recommendation engine for Obsidian vaults. Embeddings + wiki-link graph boosting with MCP server for Claude Code integration.

## Quick Start

```bash
uv sync              # install deps
uv run pytest -q     # run tests (~3s)
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
└── mcp_server.py    # FastMCP server exposing 4 tools
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
- numpy (cosine similarity via dot product on normalized embeddings)
- FastMCP (MCP server)
- PyYAML (frontmatter parsing)
- pytest (testing)

## Dos

- Use `uv` for all package management
- Follow TDD — write tests first
- Keep scoring logic in `recommender.py`
- Index files persist in `.vault-recommender-index/`

## Don'ts

- Don't mock the embedding model — prefer hand-crafted vectors or the
  `HashEncoder` stub in `tests/conftest.py` (injected via the `encoder=` kwarg
  on `build_index` / `VaultRecommender`).
- Don't add JavaScript dependencies — this is pure Python
- Don't change the MCP tool signatures without updating README

## Coverage Rule (REQUIRED)

**100% test coverage is required.** `pyproject.toml` configures pytest to fail
under 100% via `--cov-fail-under=100`. Before committing or marking any task
complete, run `uv run pytest` and confirm coverage is 100%.

When a line is genuinely untestable (e.g., the `if __name__ == "__main__":`
guard, or a branch that requires a live network/transport), mark it with
`# pragma: no cover` and explain why in the commit message. Do not weaken
`--cov-fail-under` to bypass the rule.
