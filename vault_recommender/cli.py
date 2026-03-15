# ABOUTME: CLI interface for the vault recommender — index and query commands.
# ABOUTME: Designed for both human use and Claude Code integration.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vault_recommender.graph import build_graph
from vault_recommender.indexer import build_index, VaultIndex
from vault_recommender.parser import parse_vault
from vault_recommender.recommender import VaultRecommender


DEFAULT_INDEX_DIR = Path(".vault-recommender-index")


def cmd_index(args: argparse.Namespace) -> None:
    """Build or rebuild the vault index."""
    vault_path = Path(args.vault).resolve()
    index_dir = Path(args.index_dir).resolve()

    print(f"Parsing vault at {vault_path}...")
    notes = parse_vault(vault_path)
    print(f"Found {len(notes)} notes.")

    print(f"Building embeddings with {args.model}...")
    index = build_index(notes, model_name=args.model)

    print(f"Saving index to {index_dir}...")
    index.save(index_dir)
    print("Done.")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Get recommendations for a note or topic."""
    vault_path = Path(args.vault).resolve()
    index_dir = Path(args.index_dir).resolve()

    if not (index_dir / "metadata.json").exists():
        print("No index found. Run 'index' first.", file=sys.stderr)
        sys.exit(1)

    index = VaultIndex.load(index_dir)

    # Rebuild graph from current vault (fast, no model needed)
    notes = parse_vault(vault_path)
    graph = build_graph(notes)

    recommender = VaultRecommender(index=index, graph=graph, vault_path=vault_path)

    if args.note:
        results = recommender.similar_to_note(
            args.note, top_k=args.top_k, exclude_linked=args.exclude_linked
        )
    elif args.topic:
        results = recommender.similar_to_topic(args.topic, top_k=args.top_k)
    else:
        print("Provide --note or --topic", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for i, rec in enumerate(results, 1):
            print(f"\n{'─' * 60}")
            print(f"  {i}. {rec.title or rec.path}")
            print(f"     Path:  {rec.path}")
            print(f"     Score: {rec.score:.4f}")
            if rec.tags:
                print(f"     Tags:  {', '.join(rec.tags)}")
            print(f"     Why:   {rec.reason}")
            print(f"     Preview: {rec.snippet[:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vault Recommender — semantic note recommendations"
    )
    parser.add_argument(
        "--vault", default=".", help="Path to Obsidian vault root (default: .)"
    )
    parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory to store/load the index",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    idx = subparsers.add_parser("index", help="Build the embedding index")
    idx.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="Sentence-transformer model name"
    )

    # Recommend command
    rec = subparsers.add_parser("recommend", help="Get recommendations")
    rec.add_argument("--note", help="Path to a note (relative to vault root)")
    rec.add_argument("--topic", help="Free-text topic query")
    rec.add_argument("--top-k", type=int, default=10, help="Number of results")
    rec.add_argument(
        "--exclude-linked", action="store_true", help="Hide already-linked notes"
    )
    rec.add_argument(
        "--json", action="store_true", help="Output as JSON (for LLM consumption)"
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "recommend":
        cmd_recommend(args)


if __name__ == "__main__":
    main()
