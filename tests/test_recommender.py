# ABOUTME: Integration tests for the recommendation engine.
# ABOUTME: Validates that semantic similarity + graph boosting produce ranked results.

import numpy as np
import pytest

from vault_recommender.graph import LinkGraph
from vault_recommender.indexer import NoteEntry, VaultIndex
from vault_recommender.recommender import VaultRecommender, create_recommender


def _make_test_index() -> tuple[VaultIndex, LinkGraph]:
    """Build a minimal index with hand-crafted embeddings for predictable tests.

    We use 3D vectors so we can reason about similarity geometrically:
    - "python-guide" and "coding-tips" point in similar directions (high cosine)
    - "cooking-recipe" points in an orthogonal direction (low cosine)
    - "career-plan" is between python and cooking (medium cosine to both)
    """
    entries = [
        NoteEntry(
            path="python-guide.md",
            title="Python Guide",
            tags=["python", "coding"],
            snippet="Learn Python basics",
        ),
        NoteEntry(
            path="coding-tips.md",
            title="Coding Tips",
            tags=["coding"],
            snippet="Tips for better code",
        ),
        NoteEntry(
            path="cooking-recipe.md",
            title="Cooking Recipe",
            tags=["cooking"],
            snippet="How to make pasta",
        ),
        NoteEntry(
            path="career-plan.md",
            title="Career Plan",
            tags=["career"],
            snippet="My career strategy",
        ),
    ]

    # Hand-crafted normalized vectors
    embeddings = np.array(
        [
            [0.9, 0.4, 0.1],  # python-guide
            [0.85, 0.5, 0.15],  # coding-tips (similar to python)
            [0.1, 0.1, 0.98],  # cooking-recipe (orthogonal)
            [0.5, 0.7, 0.5],  # career-plan (between)
        ],
        dtype=np.float32,
    )
    # Normalize to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    index = VaultIndex(entries=entries, embeddings=embeddings, model_name="test")

    graph = LinkGraph()
    graph.add_note("python-guide.md", ["coding-tips"])
    graph.add_note("career-plan.md", ["python-guide"])

    return index, graph


class TestVaultRecommender:
    """Test the recommendation engine with controlled data."""

    def test_similar_notes_ranked_by_cosine(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("python-guide.md", top_k=3)
        paths = [r.path for r in results]

        # coding-tips should be most similar to python-guide
        assert paths[0] == "coding-tips.md"
        # cooking-recipe should be least similar
        assert paths[-1] == "cooking-recipe.md"

    def test_graph_boost_increases_score(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph, link_boost_1hop=0.15)

        results = rec.similar_to_note("python-guide.md", top_k=3)
        coding_result = next(r for r in results if r.path == "coding-tips.md")

        # Should have graph boost in the reason
        assert "1-hop" in coding_result.reason

    def test_exclude_linked_hides_direct_links(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("python-guide.md", top_k=3, exclude_linked=True)
        paths = [r.path for r in results]

        # coding-tips is directly linked, should be excluded
        assert "coding-tips.md" not in paths

    def test_results_have_reason_field(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("python-guide.md", top_k=1)
        assert results[0].reason  # non-empty reason string

    def test_results_include_metadata(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("python-guide.md", top_k=1)
        r = results[0]
        assert r.title
        assert r.snippet
        assert isinstance(r.tags, list)
        assert isinstance(r.score, float)

    def test_unknown_note_returns_empty(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("nonexistent.md")
        assert results == []

    def test_to_dict_serializable(self):
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)

        results = rec.similar_to_note("python-guide.md", top_k=1)
        d = results[0].to_dict()
        assert "path" in d
        assert "score" in d
        assert "reason" in d
        assert isinstance(d["score"], float)


class TestCreateRecommender:
    """Tests for the shared create_recommender factory."""

    def test_missing_index_raises_file_not_found(self, tmp_path):
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        index_dir = tmp_path / "nonexistent-index"
        index_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No index found"):
            create_recommender(vault_path, index_dir)

    def test_error_message_is_actionable(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="vault-recommender index"):
            create_recommender(tmp_path, index_dir)
