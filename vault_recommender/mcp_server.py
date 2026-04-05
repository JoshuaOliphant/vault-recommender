# ABOUTME: MCP server exposing vault recommendations as structured tools.
# ABOUTME: Runs as a stdio server — Claude Code connects via .mcp.json config.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.server.lifespan import lifespan


def _resolve_paths() -> tuple[Path, Path]:
    """Resolve vault and index paths from env vars and project layout."""
    project_dir = Path(__file__).parent.parent
    index_dir = project_dir / ".vault-recommender-index"
    vault_str = os.environ.get("VAULT_PATH", str(project_dir.parent))
    vault_path = Path(vault_str).resolve()
    return vault_path, index_dir


@lifespan
async def load_recommender(server):
    """Load the index and link graph at startup so tools respond quickly."""
    from vault_recommender.recommender import create_recommender

    vault_path, index_dir = _resolve_paths()
    print(f"Loading recommender from {index_dir}...", file=sys.stderr)
    recommender = create_recommender(vault_path, index_dir)
    print(
        f"Recommender ready: {len(recommender.index.entries)} notes indexed.",
        file=sys.stderr,
    )
    yield {"recommender": recommender, "vault_path": vault_path, "index_dir": index_dir}


mcp = FastMCP(
    "vault-recommender",
    instructions=(
        "Semantic recommendation engine for an Obsidian vault. "
        "Use these tools to find conceptually related notes, discover "
        "forgotten knowledge, and surface bridging connections."
    ),
    lifespan=load_recommender,
)


@mcp.tool()
def recommend_by_topic(
    topic: str,
    ctx: Context,
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
    rec = ctx.lifespan_context["recommender"]
    results = rec.similar_to_topic(topic, top_k=top_k)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def recommend_by_note(
    note_path: str,
    ctx: Context,
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
    rec = ctx.lifespan_context["recommender"]
    results = rec.similar_to_note(note_path, top_k=top_k, exclude_linked=exclude_linked)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def find_missing_connections(
    note_path: str,
    ctx: Context,
    top_k: int = 5,
) -> str:
    """Find notes that are semantically similar but NOT yet linked.

    This is the "you should probably link these" tool -- it surfaces
    notes that are conceptually related but have no wiki-link connection.
    Great for strengthening the vault's link graph.

    Args:
        note_path: Path to the note relative to vault root.
        top_k: Number of unlinked-but-similar notes to return.

    Returns:
        JSON array of recommendations for notes that should probably
        be linked to the given note.
    """
    rec = ctx.lifespan_context["recommender"]
    results = rec.similar_to_note(note_path, top_k=top_k, exclude_linked=True)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def reload_index(ctx: Context) -> str:
    """Force the recommender to reload its index from disk.

    Call this after re-indexing the vault (e.g., via the CLI) so that
    subsequent queries use the fresh embeddings without restarting
    the MCP server.

    Returns:
        Confirmation message with the number of entries in the reloaded index.
    """
    from vault_recommender.recommender import create_recommender

    lc = ctx.lifespan_context
    lc["recommender"] = create_recommender(lc["vault_path"], lc["index_dir"])
    return f"Index reloaded. {len(lc['recommender'].index.entries)} notes indexed."


if __name__ == "__main__":
    mcp.run(transport="stdio")
