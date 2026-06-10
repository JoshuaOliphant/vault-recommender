# ABOUTME: Integration tests for the recommendation engine.
# ABOUTME: Validates that semantic similarity + graph boosting produce ranked results.

import os
import time

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

    def test_loads_existing_index(self, real_index):
        _, index_dir, vault_path = real_index
        rec = create_recommender(vault_path, index_dir)
        assert rec.vault_path == vault_path
        assert len(rec.index.entries) == 2


class TestSimilarToTopic:
    def test_topic_query_returns_results(self):
        from tests.conftest import HashEncoder

        # Encoder dimension must match the index's embedding dimension
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph, encoder=HashEncoder(dim=3))

        results = rec.similar_to_topic("python", top_k=2)
        assert len(results) == 2
        for r in results:
            assert "semantic similarity" in r.reason

    def test_get_model_lazy_loads_when_no_encoder_injected(self, monkeypatch):
        from tests.conftest import HashEncoder
        import vault_recommender.recommender as rec_mod

        # Patch the SentenceTransformer constructor so no network is required
        monkeypatch.setattr(
            rec_mod, "SentenceTransformer", lambda name: HashEncoder(dim=3)
        )

        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph)
        assert rec._model is None
        m1 = rec._get_model()
        assert isinstance(m1, HashEncoder)
        # Subsequent calls return the cached instance
        assert rec._get_model() is m1


class TestStalenessBoost:
    def test_stale_file_gets_boost(self, tmp_path):
        # Vault with a real file we can age
        (tmp_path / "python-guide.md").write_text("# Python")
        old = time.time() - 60 * 60 * 24 * 365  # 1 year old
        os.utime(tmp_path / "python-guide.md", (old, old))

        index, graph = _make_test_index()
        rec = VaultRecommender(
            index=index, graph=graph, vault_path=tmp_path, staleness_boost=0.1
        )
        results = rec.similar_to_note("coding-tips.md", top_k=4)
        python_result = next(r for r in results if r.path == "python-guide.md")
        assert "stale" in python_result.reason

    def test_fresh_file_gets_no_staleness_boost(self, tmp_path):
        (tmp_path / "python-guide.md").write_text("# Python")
        # Default mtime is "now" — should not be stale
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph, vault_path=tmp_path)
        results = rec.similar_to_note("coding-tips.md", top_k=4)
        python_result = next(r for r in results if r.path == "python-guide.md")
        assert "stale" not in python_result.reason

    def test_missing_file_skips_staleness_branch(self, tmp_path):
        # vault_path set, but the entry's file doesn't exist on disk
        index, graph = _make_test_index()
        rec = VaultRecommender(index=index, graph=graph, vault_path=tmp_path)
        results = rec.similar_to_note("python-guide.md", top_k=3)
        for r in results:
            assert "stale" not in r.reason


class TestGraphBoost:
    def test_2hop_neighbor_gets_smaller_boost(self):
        index, graph = _make_test_index()
        # career-plan -> python-guide -> coding-tips, so coding-tips is 2 hops from career-plan
        rec = VaultRecommender(index=index, graph=graph)
        results = rec.similar_to_note("career-plan.md", top_k=4)
        coding_result = next(r for r in results if r.path == "coding-tips.md")
        assert "2-hop" in coding_result.reason
