# ABOUTME: MCP server exposing vault recommendations as structured tools.
# ABOUTME: Runs as a stdio server — Claude Code connects via .mcp.json config.

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Heavy imports (sentence_transformers, torch, numpy) are deferred to
# _get_recommender() so the MCP handshake completes before model loading.

_recommender = None
_vault_path: Path | None = None

mcp = FastMCP(
    "vault-recommender",
    instructions=(
        "Semantic recommendation engine for an Obsidian vault. "
        "Use these tools to find conceptually related notes, discover "
        "forgotten knowledge, and surface bridging connections."
    ),
)


def _get_recommender():
    """Lazy-init the recommender from the pre-built index."""
    global _recommender, _vault_path

    if _recommender is not None:
        return _recommender

    # Defer heavy imports until first tool call
    from vault_recommender.graph import build_graph
    from vault_recommender.indexer import VaultIndex
    from vault_recommender.parser import parse_vault
    from vault_recommender.recommender import VaultRecommender

    experiment_dir = Path(__file__).parent.parent
    index_dir = experiment_dir / ".vault-recommender-index"

    vault_str = os.environ.get(
        "VAULT_PATH",
        str(experiment_dir.parent),
    )
    _vault_path = Path(vault_str).resolve()

    if not (index_dir / "metadata.json").exists():
        raise RuntimeError(
            f"No index found at {index_dir}. "
            "Run 'uv run python -m vault_recommender.cli index' first."
        )

    index = VaultIndex.load(index_dir)
    notes = parse_vault(_vault_path)
    graph = build_graph(notes)

    _recommender = VaultRecommender(index=index, graph=graph, vault_path=_vault_path)
    return _recommender


@mcp.tool()
def recommend_by_topic(
    topic: str,
    top_k: int = 10,
) -> str:
    """Find vault notes semantically related to a topic or question.

    Use this when you need to surface relevant notes the user may have
    forgotten, or to find conceptual connections across the vault.

    Args:
        topic: Natural language query (e.g., "career transition strategies",
               "what have I written about Python testing?").
        top_k: Number of recommendations to return (default 10).

    Returns:
        JSON array of recommendations with path, title, score, snippet,
        tags, and a reason explaining why each note was recommended.
    """
    rec = _get_recommender()
    results = rec.similar_to_topic(topic, top_k=top_k)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def recommend_by_note(
    note_path: str,
    top_k: int = 10,
    exclude_linked: bool = False,
) -> str:
    """Find vault notes similar to a specific note.

    Use this to discover notes related to one the user is currently
    reading or discussing. Set exclude_linked=True to hide notes
    already connected via wiki-links (surfaces only "hidden" connections).

    Args:
        note_path: Path to the note relative to vault root
                   (e.g., "areas/career/job-search-reference.md").
        top_k: Number of recommendations to return (default 10).
        exclude_linked: If True, exclude notes already linked via wiki-links.
                       Great for finding "missing connections".

    Returns:
        JSON array of recommendations with path, title, score, snippet,
        tags, and a reason explaining why each note was recommended.
    """
    rec = _get_recommender()
    results = rec.similar_to_note(note_path, top_k=top_k, exclude_linked=exclude_linked)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def find_missing_connections(
    note_path: str,
    top_k: int = 5,
) -> str:
    """Find notes that are semantically similar but NOT yet linked.

    This is the "you should probably link these" tool — it surfaces
    notes that are conceptually related but have no wiki-link connection.
    Great for strengthening the vault's link graph.

    Args:
        note_path: Path to the note relative to vault root.
        top_k: Number of unlinked-but-similar notes to return.

    Returns:
        JSON array of recommendations for notes that should probably
        be linked to the given note.
    """
    rec = _get_recommender()
    results = rec.similar_to_note(note_path, top_k=top_k, exclude_linked=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
