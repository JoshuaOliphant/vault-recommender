# ABOUTME: Core recommendation engine — combines embedding similarity with graph boosting.
# ABOUTME: Returns context-rich results designed for LLM consumption.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from vault_recommender.graph import LinkGraph, build_graph
from vault_recommender.indexer import VaultIndex


def create_recommender(vault_path: Path, index_dir: Path) -> "VaultRecommender":
    """Load index, parse vault, build graph, and return a ready recommender.

    Shared factory used by CLI, HTTP server, and MCP server so init logic
    lives in one place.
    """
    from vault_recommender.parser import parse_vault

    if not (index_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run 'vault-recommender index' first."
        )

    index = VaultIndex.load(index_dir)
    notes = parse_vault(vault_path)
    graph = build_graph(notes)
    return VaultRecommender(index=index, graph=graph, vault_path=vault_path)


@dataclass
class Recommendation:
    """A single recommendation, rich enough for an LLM to act on."""

    path: str
    title: str
    score: float
    snippet: str
    tags: list[str] = field(default_factory=list)
    reason: str = ""  # Human/LLM-readable explanation of why this was recommended

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "title": self.title,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "tags": self.tags,
            "reason": self.reason,
        }


class VaultRecommender:
    """Recommendation engine combining semantic similarity and graph signals."""

    def __init__(
        self,
        index: VaultIndex,
        graph: LinkGraph,
        vault_path: Path | None = None,
        link_boost_1hop: float = 0.50,
        link_boost_2hop: float = 0.05,
        staleness_boost: float = 0.1,
    ):
        self.index = index
        self.graph = graph
        self.vault_path = vault_path
        self.link_boost_1hop = link_boost_1hop
        self.link_boost_2hop = link_boost_2hop
        self.staleness_boost = staleness_boost
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.index.model_name)
        return self._model

    def similar_to_note(
        self,
        note_path: str,
        top_k: int = 10,
        exclude_linked: bool = False,
    ) -> list[Recommendation]:
        """Find notes similar to a given note.

        Args:
            note_path: Path to the query note (relative to vault root).
            top_k: Number of recommendations to return.
            exclude_linked: If True, exclude notes already directly linked.

        Returns:
            Ranked list of Recommendations.
        """
        # Find the query note in the index
        query_idx = None
        for i, entry in enumerate(self.index.entries):
            if entry.path == note_path:
                query_idx = i
                break

        if query_idx is None:
            return []

        query_embedding = self.index.embeddings[query_idx]
        return self._rank(query_embedding, query_idx, note_path, top_k, exclude_linked)

    def similar_to_topic(
        self,
        topic: str,
        top_k: int = 10,
    ) -> list[Recommendation]:
        """Find notes similar to a free-text topic query.

        Args:
            topic: Natural language query (e.g., "career transition strategies").
            top_k: Number of recommendations to return.

        Returns:
            Ranked list of Recommendations.
        """
        model = self._get_model()
        query_embedding = model.encode(topic, normalize_embeddings=True)
        return self._rank(
            query_embedding, exclude_idx=None, query_path=None, top_k=top_k
        )

    def _rank(
        self,
        query_embedding: np.ndarray,
        exclude_idx: int | None,
        query_path: str | None,
        top_k: int,
        exclude_linked: bool = False,
    ) -> list[Recommendation]:
        """Core ranking: cosine similarity + graph boost + staleness boost."""
        # Cosine similarity via dot product (embeddings are already normalized)
        similarities = self.index.embeddings @ query_embedding

        # Get graph neighbors for boosting
        neighbors: dict[str, int] = {}
        if query_path:
            neighbors = self.graph.neighbors(query_path, max_hops=2)

        scored: list[tuple[int, float, str]] = []
        for i, entry in enumerate(self.index.entries):
            if i == exclude_idx:
                continue

            if (
                exclude_linked
                and query_path
                and self.graph.are_linked(query_path, entry.path)
            ):
                continue

            score = float(similarities[i])
            reason_parts = [f"semantic similarity: {score:.3f}"]

            # Boost notes that are graph-adjacent but not directly linked
            entry_normalized = entry.path.removesuffix(".md")
            if entry_normalized in neighbors:
                hop_distance = neighbors[entry_normalized]
                if hop_distance == 1:
                    score += self.link_boost_1hop
                    reason_parts.append("1-hop link neighbor")
                elif hop_distance == 2:
                    score += self.link_boost_2hop
                    reason_parts.append("2-hop bridge connection")

            # Boost stale notes (haven't been modified recently)
            if self.vault_path:
                full_path = self.vault_path / entry.path
                if full_path.exists():
                    mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
                    days_stale = (datetime.now() - mtime).days
                    if days_stale > 30:
                        stale_factor = min(days_stale / 365, 1.0) * self.staleness_boost
                        score += stale_factor
                        reason_parts.append(f"stale ({days_stale}d untouched)")

            scored.append((i, score, " + ".join(reason_parts)))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return [
            Recommendation(
                path=self.index.entries[i].path,
                title=self.index.entries[i].title,
                score=score,
                snippet=self.index.entries[i].snippet,
                tags=self.index.entries[i].tags,
                reason=reason,
            )
            for i, score, reason in top
        ]
