# ABOUTME: Shared pytest fixtures and a deterministic stub encoder.
# ABOUTME: The stub keeps tests offline-friendly while preserving real numpy math.

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from vault_recommender.indexer import build_index
from vault_recommender.parser import parse_vault


class HashEncoder:
    """Deterministic stand-in for SentenceTransformer.

    Hashes input strings into reproducible normalized vectors. This is not a
    mock: the embeddings are real numpy arrays produced by deterministic math,
    just not the ones a transformer would produce. Lets us test code paths
    that need an ``encode`` method without downloading model weights.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Map bytes to floats in [-1, 1], cycling as needed
        raw = np.frombuffer(
            (h * ((self.dim // len(h)) + 1))[: self.dim], dtype=np.uint8
        )
        v = (raw.astype(np.float32) / 127.5) - 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture(scope="session")
def encoder() -> HashEncoder:
    return HashEncoder()


@pytest.fixture(scope="session")
def tiny_vault(tmp_path_factory) -> Path:
    """A real on-disk vault with a couple of small notes."""
    vault = tmp_path_factory.mktemp("vault")
    (vault / "alpha.md").write_text(
        "---\ntitle: Alpha\ntags: [python]\n---\n\n# Alpha\n\nPython programming notes.\n"
    )
    (vault / "beta.md").write_text("# Beta\n\nNotes about cooking pasta.\n[[alpha]]\n")
    return vault


@pytest.fixture(scope="session")
def real_index(tiny_vault, tmp_path_factory, encoder):
    """Build an index using the deterministic encoder, persist to disk."""
    notes = parse_vault(tiny_vault)
    index = build_index(notes, encoder=encoder)
    index_dir = tmp_path_factory.mktemp("index")
    index.save(index_dir)
    return index, index_dir, tiny_vault
