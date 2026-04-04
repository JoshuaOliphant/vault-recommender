# ABOUTME: Tests for the HTTP serve endpoint using Starlette TestClient.
# ABOUTME: Uses real VaultRecommender with hand-crafted embeddings (no mocks).

import pytest
from starlette.testclient import TestClient

from vault_recommender.recommender import VaultRecommender
from tests.test_recommender import _make_test_index


@pytest.fixture
def client():
    """Build a TestClient with a real recommender injected."""
    from vault_recommender.server import create_app

    index, graph = _make_test_index()
    recommender = VaultRecommender(index=index, graph=graph)
    app = create_app(recommender=recommender)
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["notes_indexed"] == 4

    def test_health_method_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


class TestRecommend:
    def test_recommend_by_note(self, client):
        resp = client.get(
            "/recommend", params={"note": "python-guide.md", "top_k": "2"}
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
        # coding-tips is most similar to python-guide
        assert results[0]["path"] == "coding-tips.md"

    def test_recommend_missing_params_returns_400(self, client):
        resp = client.get("/recommend")
        assert resp.status_code == 400
        assert (
            "topic" in resp.json()["error"].lower()
            or "note" in resp.json()["error"].lower()
        )

    def test_recommend_both_topic_and_note_returns_400(self, client):
        resp = client.get(
            "/recommend", params={"topic": "python", "note": "python-guide.md"}
        )
        assert resp.status_code == 400

    def test_recommend_unknown_note_returns_empty(self, client):
        resp = client.get("/recommend", params={"note": "nonexistent.md"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_recommend_top_k_defaults_to_3(self, client):
        resp = client.get("/recommend", params={"note": "python-guide.md"})
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_recommend_exclude_linked(self, client):
        resp = client.get(
            "/recommend",
            params={"note": "python-guide.md", "exclude_linked": "true", "top_k": "3"},
        )
        assert resp.status_code == 200
        paths = [r["path"] for r in resp.json()]
        # coding-tips is directly linked to python-guide, should be excluded
        assert "coding-tips.md" not in paths

    def test_recommend_results_have_expected_fields(self, client):
        resp = client.get(
            "/recommend", params={"note": "python-guide.md", "top_k": "1"}
        )
        result = resp.json()[0]
        assert "path" in result
        assert "title" in result
        assert "score" in result
        assert "snippet" in result
        assert "reason" in result


class TestReload:
    def test_reload_returns_200(self, client):
        """Reload with injected recommender re-stores the same instance."""
        resp = client.post("/reload")
        assert resp.status_code == 200
        assert (
            "reloaded" in resp.json()["message"].lower()
            or "notes" in resp.json()["message"].lower()
        )


class TestRouting:
    def test_unknown_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
