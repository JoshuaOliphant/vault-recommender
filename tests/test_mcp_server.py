# ABOUTME: Tests for the MCP tool functions and lifespan loader.
# ABOUTME: Calls tool functions directly with a stub Context, no real MCP transport.

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from tests.conftest import HashEncoder


def _ctx_with(recommender, vault_path: Path | None = None, index_dir: Path | None = None):
    """Build a stub fastmcp.Context exposing the lifespan_context dict."""
    return SimpleNamespace(
        lifespan_context={
            "recommender": recommender,
            "vault_path": vault_path,
            "index_dir": index_dir,
        }
    )


@pytest.fixture
def recommender():
    from tests.test_recommender import _make_test_index
    from vault_recommender.recommender import VaultRecommender

    index, graph = _make_test_index()
    return VaultRecommender(
        index=index, graph=graph, encoder=HashEncoder(dim=3)
    )


class TestRecommendByTopic:
    def test_returns_json_array(self, recommender):
        from vault_recommender.mcp_server import recommend_by_topic

        result = recommend_by_topic("python", ctx=_ctx_with(recommender), top_k=2)
        body = json.loads(result)
        assert isinstance(body, list)
        assert len(body) == 2


class TestRecommendByNote:
    def test_returns_json_array(self, recommender):
        from vault_recommender.mcp_server import recommend_by_note

        result = recommend_by_note(
            "python-guide.md", ctx=_ctx_with(recommender), top_k=2
        )
        body = json.loads(result)
        assert any(r["path"] == "coding-tips.md" for r in body)

    def test_exclude_linked_passes_through(self, recommender):
        from vault_recommender.mcp_server import recommend_by_note

        result = recommend_by_note(
            "python-guide.md",
            ctx=_ctx_with(recommender),
            top_k=3,
            exclude_linked=True,
        )
        body = json.loads(result)
        assert all(r["path"] != "coding-tips.md" for r in body)


class TestFindMissingConnections:
    def test_excludes_linked_notes(self, recommender):
        from vault_recommender.mcp_server import find_missing_connections

        result = find_missing_connections(
            "python-guide.md", ctx=_ctx_with(recommender), top_k=5
        )
        body = json.loads(result)
        # coding-tips is linked, so it must not appear
        assert all(r["path"] != "coding-tips.md" for r in body)


class TestReloadIndex:
    def test_reload_creates_new_recommender(self, real_index):
        from vault_recommender.mcp_server import reload_index

        _, index_dir, vault_path = real_index
        ctx = _ctx_with(recommender=None, vault_path=vault_path, index_dir=index_dir)
        msg = reload_index(ctx=ctx)
        assert "2 notes" in msg
        assert ctx.lifespan_context["recommender"] is not None


class TestResolvePaths:
    def test_uses_env_var(self, monkeypatch, tmp_path):
        from vault_recommender import mcp_server

        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        vault_path, index_dir = mcp_server._resolve_paths()
        assert vault_path == tmp_path.resolve()
        assert index_dir.name == ".vault-recommender-index"

    def test_falls_back_to_project_parent(self, monkeypatch):
        from vault_recommender import mcp_server

        monkeypatch.delenv("VAULT_PATH", raising=False)
        vault_path, _ = mcp_server._resolve_paths()
        assert vault_path.is_absolute()


class TestLifespan:
    def test_loads_recommender_into_context(self, real_index, monkeypatch):
        from vault_recommender import mcp_server

        _, index_dir, vault_path = real_index
        # Force _resolve_paths to point at our prepared index
        monkeypatch.setattr(
            mcp_server, "_resolve_paths", lambda: (vault_path, index_dir)
        )

        async def run():
            agen = mcp_server.load_recommender._fn(server=None)
            ctx = await agen.__anext__()
            try:
                assert "recommender" in ctx
                assert ctx["vault_path"] == vault_path
                assert ctx["index_dir"] == index_dir
                assert len(ctx["recommender"].index.entries) == 2
            finally:
                with pytest.raises(StopAsyncIteration):
                    await agen.__anext__()

        asyncio.run(run())


class TestEntrypointGuard:
    def test_main_block_is_excluded_from_coverage(self):
        # The if __name__ == "__main__" block runs mcp.run() which starts a stdio
        # server — not testable without a transport. The guard is marked
        # `# pragma: no cover` in the source, so this test simply documents that.
        import vault_recommender.mcp_server as m
        src = Path(m.__file__).read_text()
        assert "pragma: no cover" in src
