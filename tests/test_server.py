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
    from tests.conftest import HashEncoder

    index, graph = _make_test_index()
    recommender = VaultRecommender(index=index, graph=graph, encoder=HashEncoder(dim=3))
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
    def test_recommend_invalid_top_k_returns_400(self, client):
        resp = client.get(
            "/recommend", params={"note": "python-guide.md", "top_k": "abc"}
        )
        assert resp.status_code == 400
        assert "top_k" in resp.json()["error"].lower()

    def test_recommend_negative_top_k_returns_400(self, client):
        resp = client.get(
            "/recommend", params={"note": "python-guide.md", "top_k": "-1"}
        )
        assert resp.status_code == 400

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
    def test_reload_without_vault_path_returns_503(self, client):
        """Reload on test-injected server (no vault_path) returns 503."""
        resp = client.post("/reload")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["error"].lower()

    def test_reload_method_not_allowed(self, client):
        resp = client.get("/reload")
        assert resp.status_code == 405


class TestRouting:
    def test_unknown_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404


class TestRecommendByTopic:
    def test_topic_query_returns_results(self, client):
        resp = client.get("/recommend", params={"topic": "python", "top_k": "2"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        for r in body:
            assert "score" in r


class TestCreateApp:
    def test_create_app_with_vault_path_loads_recommender(self, real_index):
        from vault_recommender.server import create_app

        _, index_dir, vault_path = real_index
        app = create_app(vault_path=vault_path, index_dir=index_dir)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["notes_indexed"] == 2

    def test_create_app_without_args_raises(self):
        from vault_recommender.server import create_app

        with pytest.raises(ValueError, match="recommender or vault_path"):
            create_app()


class TestReloadWithVaultPath:
    def test_reload_rebuilds_recommender(self, real_index):
        from vault_recommender.server import create_app

        _, index_dir, vault_path = real_index
        app = create_app(vault_path=vault_path, index_dir=index_dir)
        client = TestClient(app)
        resp = client.post("/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert "reloaded" in body["message"].lower()
        assert "2 notes" in body["message"]


class TestRunServer:
    def test_run_server_invokes_uvicorn(self, monkeypatch, real_index):
        import vault_recommender.server as srv_mod

        captured = {}

        def fake_run(app, host, port):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port

        # Patch uvicorn.run to avoid actually starting a server
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)

        _, index_dir, vault_path = real_index
        srv_mod.run_server(
            vault_path=vault_path, index_dir=index_dir, host="0.0.0.0", port=9999
        )
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 9999
        assert captured["app"] is not None
