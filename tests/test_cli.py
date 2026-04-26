# ABOUTME: Tests for the CLI entrypoints — index, recommend, serve commands.
# ABOUTME: Patches SentenceTransformer/uvicorn so the CLI runs offline.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from vault_recommender import cli
from tests.conftest import HashEncoder


@pytest.fixture
def patched_transformer(monkeypatch):
    """Replace SentenceTransformer in indexer + recommender with a deterministic stub."""
    from vault_recommender import indexer, recommender

    monkeypatch.setattr(indexer, "SentenceTransformer", lambda name: HashEncoder())
    monkeypatch.setattr(recommender, "SentenceTransformer", lambda name: HashEncoder())


@pytest.fixture
def vault(tmp_path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\n[[b]]")
    (tmp_path / "b.md").write_text("# B\n\nbody")
    return tmp_path


def _build_index(vault: Path, index_dir: Path) -> None:
    """Helper: invoke cmd_index with the canonical defaults."""
    args = argparse.Namespace(
        vault=str(vault),
        index_dir=str(index_dir),
        model="all-MiniLM-L6-v2",
    )
    cli.cmd_index(args)


class TestCmdIndex:
    def test_builds_and_persists_index(self, vault, tmp_path, patched_transformer, capsys):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)

        out = capsys.readouterr().out
        assert "Found 2 notes" in out
        assert (index_dir / "metadata.json").exists()
        assert (index_dir / "embeddings.npy").exists()


class TestCmdRecommend:
    def _make_args(self, vault: Path, index_dir: Path, **overrides) -> argparse.Namespace:
        defaults = dict(
            vault=str(vault),
            index_dir=str(index_dir),
            note=None,
            topic=None,
            top_k=2,
            exclude_linked=False,
            json=False,
            rebuild=False,
            model="all-MiniLM-L6-v2",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_index_exits(self, vault, tmp_path, patched_transformer):
        args = self._make_args(vault, tmp_path / "missing", note="a.md")
        with pytest.raises(SystemExit) as exc:
            cli.cmd_recommend(args)
        assert exc.value.code == 1

    def test_no_query_exits(self, vault, tmp_path, patched_transformer):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        args = self._make_args(vault, index_dir)
        with pytest.raises(SystemExit) as exc:
            cli.cmd_recommend(args)
        assert exc.value.code == 1

    def test_recommend_by_note_human_output(self, vault, tmp_path, patched_transformer, capsys):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        args = self._make_args(vault, index_dir, note="a.md")
        cli.cmd_recommend(args)

        out = capsys.readouterr().out
        assert "Path:" in out
        assert "Score:" in out

    def test_recommend_by_note_with_tags_output(self, vault, tmp_path, patched_transformer, capsys):
        # Add a note with tags so the "Tags:" branch is exercised
        (vault / "tagged.md").write_text(
            "---\ntitle: T\ntags: [py, code]\n---\n\nbody"
        )
        # Add another tagged note so it can show up in results
        (vault / "other.md").write_text(
            "---\ntitle: O\ntags: [py]\n---\n\nbody"
        )
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        args = self._make_args(vault, index_dir, note="tagged.md", top_k=3)
        cli.cmd_recommend(args)
        assert "Tags:" in capsys.readouterr().out

    def test_recommend_by_topic_json_output(self, vault, tmp_path, patched_transformer, capsys):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        capsys.readouterr()  # discard build output
        args = self._make_args(vault, index_dir, topic="python", json=True)
        cli.cmd_recommend(args)

        body = json.loads(capsys.readouterr().out)
        assert isinstance(body, list)
        assert len(body) >= 1
        assert all("path" in r for r in body)

    def test_rebuild_when_stale(self, vault, tmp_path, patched_transformer, capsys):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)

        # Touch a file so the index becomes stale
        import os, time
        future = time.time() + 10_000
        os.utime(vault / "a.md", (future, future))

        args = self._make_args(vault, index_dir, note="a.md", rebuild=True)
        cli.cmd_recommend(args)
        err = capsys.readouterr().err
        assert "stale" in err.lower()

    def test_rebuild_when_fresh_skips(self, vault, tmp_path, patched_transformer, capsys):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        args = self._make_args(vault, index_dir, note="a.md", rebuild=True)
        cli.cmd_recommend(args)
        err = capsys.readouterr().err
        assert "fresh" in err.lower()

    def test_rebuild_with_no_existing_index_uses_default_model(
        self, vault, tmp_path, patched_transformer, capsys
    ):
        index_dir = tmp_path / "missing"
        args = self._make_args(vault, index_dir, note="a.md", rebuild=True)
        # No index exists, so cmd_recommend will rebuild and then try to load
        cli.cmd_recommend(args)
        assert "Path:" in capsys.readouterr().out

    def test_rebuild_with_corrupt_metadata_falls_back(
        self, vault, tmp_path, patched_transformer, capsys
    ):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "metadata.json").write_text("{not json")
        args = self._make_args(vault, index_dir, note="a.md", rebuild=True)
        cli.cmd_recommend(args)
        # Should have rebuilt cleanly with default model
        assert "Path:" in capsys.readouterr().out


class TestCmdServe:
    def test_missing_index_exits(self, vault, tmp_path, capsys):
        args = argparse.Namespace(
            vault=str(vault),
            index_dir=str(tmp_path / "missing"),
            host="127.0.0.1",
            port=7532,
        )
        with pytest.raises(SystemExit) as exc:
            cli.cmd_serve(args)
        assert exc.value.code == 1
        assert "No index" in capsys.readouterr().err

    def test_serve_invokes_run_server(self, vault, tmp_path, monkeypatch, patched_transformer):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)

        captured = {}

        def fake_run_server(*, recommender, vault_path, index_dir, host, port):
            captured.update(host=host, port=port)

        monkeypatch.setattr("vault_recommender.server.run_server", fake_run_server)
        args = argparse.Namespace(
            vault=str(vault),
            index_dir=str(index_dir),
            host="127.0.0.1",
            port=7532,
        )
        cli.cmd_serve(args)
        assert captured == {"host": "127.0.0.1", "port": 7532}


class TestMain:
    def test_main_dispatches_index(self, vault, tmp_path, monkeypatch, patched_transformer):
        index_dir = tmp_path / "idx"
        monkeypatch.setattr(
            sys, "argv", [
                "vault-recommender",
                "--vault", str(vault),
                "--index-dir", str(index_dir),
                "index",
            ]
        )
        cli.main()
        assert (index_dir / "metadata.json").exists()

    def test_main_dispatches_recommend(self, vault, tmp_path, monkeypatch, patched_transformer):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)
        monkeypatch.setattr(
            sys, "argv", [
                "vault-recommender",
                "--vault", str(vault),
                "--index-dir", str(index_dir),
                "recommend",
                "--note", "a.md",
                "--top-k", "1",
                "--json",
            ]
        )
        cli.main()

    def test_main_dispatches_serve(self, vault, tmp_path, monkeypatch, patched_transformer):
        index_dir = tmp_path / "idx"
        _build_index(vault, index_dir)

        called = {}
        monkeypatch.setattr(
            "vault_recommender.server.run_server",
            lambda **kwargs: called.update(kwargs),
        )
        monkeypatch.setattr(
            sys, "argv", [
                "vault-recommender",
                "--vault", str(vault),
                "--index-dir", str(index_dir),
                "serve",
            ]
        )
        cli.main()
        assert called["host"] == "127.0.0.1"
        assert called["port"] == 7532


