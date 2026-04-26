# ABOUTME: Tests for the index — VaultIndex persistence, staleness checks, build_index.
# ABOUTME: Uses a single session-scoped real embedding index to keep test time reasonable.

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from vault_recommender.indexer import (
    DEFAULT_MODEL,
    NoteEntry,
    VaultIndex,
    _prepare_text,
)


class TestVaultIndexValidation:
    def test_mismatched_entries_and_embeddings_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            VaultIndex(
                entries=[NoteEntry(path="a.md", title="A")],
                embeddings=np.zeros((2, 3), dtype=np.float32),
                model_name="test",
            )


class TestSaveAndLoad:
    def test_round_trip_preserves_data(self, tmp_path):
        entries = [
            NoteEntry(path="a.md", title="A", tags=["x"], wiki_links=["b"], snippet="aa"),
            NoteEntry(path="b.md", title="B"),
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = VaultIndex(entries=entries, embeddings=embeddings, model_name="test")
        index.save(tmp_path)

        # save() stamps built_at on the in-memory instance
        assert index.built_at is not None
        assert index.built_at.tzinfo is not None

        loaded = VaultIndex.load(tmp_path)
        assert loaded.model_name == "test"
        assert len(loaded.entries) == 2
        assert loaded.entries[0].path == "a.md"
        assert loaded.entries[0].tags == ["x"]
        np.testing.assert_array_equal(loaded.embeddings, embeddings)
        assert loaded.built_at is not None
        assert loaded.built_at.tzinfo is not None

    def test_load_missing_built_at_returns_none(self, tmp_path):
        # Hand-craft metadata without built_at to exercise the fallback branch
        np.save(tmp_path / "embeddings.npy", np.zeros((1, 2), dtype=np.float32))
        meta = {
            "model_name": "test",
            "entries": [{"path": "a.md", "title": "A", "tags": [], "wiki_links": [], "snippet": ""}],
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        loaded = VaultIndex.load(tmp_path)
        assert loaded.built_at is None

    def test_load_naive_built_at_is_promoted_to_utc(self, tmp_path):
        np.save(tmp_path / "embeddings.npy", np.zeros((1, 2), dtype=np.float32))
        meta = {
            "model_name": "test",
            "built_at": "2024-01-01T00:00:00",  # naive
            "entries": [{"path": "a.md", "title": "A", "tags": [], "wiki_links": [], "snippet": ""}],
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        loaded = VaultIndex.load(tmp_path)
        assert loaded.built_at is not None
        assert loaded.built_at.tzinfo == timezone.utc


class TestIsStale:
    def _stamp_index(self, index_dir: Path, when: datetime | None) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        meta: dict = {"model_name": "test", "entries": []}
        if when is not None:
            meta["built_at"] = when.isoformat()
        (index_dir / "metadata.json").write_text(json.dumps(meta))

    def test_missing_metadata_is_stale(self, tmp_path):
        assert VaultIndex.is_stale(tmp_path / "no-such-dir", tmp_path) is True

    def test_corrupted_metadata_is_stale(self, tmp_path, capsys):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "metadata.json").write_text("{not json")

        assert VaultIndex.is_stale(index_dir, tmp_path) is True
        assert "corrupted" in capsys.readouterr().err

    def test_missing_built_at_is_stale(self, tmp_path):
        index_dir = tmp_path / "idx"
        self._stamp_index(index_dir, when=None)
        assert VaultIndex.is_stale(index_dir, tmp_path) is True

    def test_unparseable_built_at_is_stale(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "metadata.json").write_text(
            json.dumps({"model_name": "t", "built_at": "not-a-date", "entries": []})
        )
        assert VaultIndex.is_stale(index_dir, tmp_path) is True

    def test_naive_built_at_promoted_for_comparison(self, tmp_path):
        # Naive timestamp far in the future — vault file will be older
        future_naive = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=365)
        ).replace(microsecond=0)
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        (index_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": "t",
                    "built_at": future_naive.isoformat(),
                    "entries": [],
                }
            )
        )
        (tmp_path / "old.md").write_text("# Old")
        assert VaultIndex.is_stale(index_dir, tmp_path) is False

    def test_missing_vault_path_is_stale(self, tmp_path, capsys):
        index_dir = tmp_path / "idx"
        self._stamp_index(index_dir, when=datetime.now(timezone.utc))
        assert VaultIndex.is_stale(index_dir, tmp_path / "no-vault") is True
        assert "vault path" in capsys.readouterr().err

    def test_empty_vault_is_not_stale(self, tmp_path):
        index_dir = tmp_path / "idx"
        self._stamp_index(index_dir, when=datetime.now(timezone.utc))
        # Empty vault dir — no .md files
        assert VaultIndex.is_stale(index_dir, tmp_path) is False

    def test_newer_md_file_marks_stale(self, tmp_path):
        index_dir = tmp_path / "idx"
        # Built in the past
        self._stamp_index(index_dir, when=datetime.now(timezone.utc) - timedelta(days=1))
        # Add a fresh .md file
        f = tmp_path / "fresh.md"
        f.write_text("# Fresh")
        # Force mtime to now
        now = time.time()
        import os
        os.utime(f, (now, now))
        assert VaultIndex.is_stale(index_dir, tmp_path) is True

    def test_older_md_file_is_not_stale(self, tmp_path):
        index_dir = tmp_path / "idx"
        self._stamp_index(index_dir, when=datetime.now(timezone.utc))
        f = tmp_path / "old.md"
        f.write_text("# Old")
        old = time.time() - 60 * 60 * 24
        import os
        os.utime(f, (old, old))
        assert VaultIndex.is_stale(index_dir, tmp_path) is False


class TestPrepareText:
    def test_includes_topic_for_nested_path(self):
        text = _prepare_text(
            entry_path="areas/career/job.md",
            title="Job",
            tags=["work"],
            body="long body" * 100,
            wiki_links=["resume"],
        )
        assert "Topic: areas/career" in text
        assert "Title: Job" in text
        assert "Tags: work" in text
        assert "Related: resume" in text
        # Body is truncated to 300 chars
        assert "Body: " in text
        body_section = text.split("Body: ", 1)[1]
        assert len(body_section) == 300

    def test_omits_topic_when_path_is_root(self):
        text = _prepare_text(
            entry_path="root.md", title="Root", tags=[], body="hello", wiki_links=None
        )
        assert "Topic:" not in text
        assert "Tags:" not in text
        assert "Related:" not in text
        assert "Title: Root" in text


class TestBuildIndex:
    def test_index_has_expected_shape(self, real_index, encoder):
        index, _, _ = real_index
        assert len(index.entries) == 2
        assert index.embeddings.shape == (2, encoder.dim)
        assert index.model_name == DEFAULT_MODEL
        for e in index.entries:
            assert len(e.snippet) <= 200

    def test_string_tag_is_coerced_to_list(self, encoder):
        from vault_recommender.indexer import build_index
        from vault_recommender.parser import parse_note

        note = parse_note("---\ntitle: T\ntags: solo\n---\n\nbody", path="t.md")
        index = build_index([note], encoder=encoder)
        assert index.entries[0].tags == ["solo"]

    def test_int_tag_is_coerced_to_string(self, encoder):
        from vault_recommender.indexer import build_index
        from vault_recommender.parser import parse_note

        note = parse_note("---\ntitle: T\ntags: [1, 2]\n---\n\nbody", path="t.md")
        index = build_index([note], encoder=encoder)
        assert index.entries[0].tags == ["1", "2"]

    def test_no_tags_in_frontmatter_yields_empty_list(self, encoder):
        from vault_recommender.indexer import build_index
        from vault_recommender.parser import parse_note

        note = parse_note("# Title\n\nbody", path="t.md")
        index = build_index([note], encoder=encoder)
        assert index.entries[0].tags == []
