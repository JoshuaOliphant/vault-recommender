# ABOUTME: Builds the embedding index — converts parsed notes into semantic vectors.
# ABOUTME: Uses sentence-transformers for local embedding, stores index as numpy + JSON.

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class NoteEntry:
    """A note's identity and metadata, stored alongside its embedding."""

    path: str
    title: str
    tags: list[str] = field(default_factory=list)
    wiki_links: list[str] = field(default_factory=list)
    snippet: str = ""


@dataclass
class VaultIndex:
    """The complete search index: embeddings + metadata for every note."""

    entries: list[NoteEntry]
    embeddings: np.ndarray  # shape: (n_notes, embedding_dim)
    model_name: str

    def save(self, directory: Path) -> None:
        """Persist index to disk as numpy array + JSON metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        metadata = {
            "model_name": self.model_name,
            "entries": [asdict(e) for e in self.entries],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VaultIndex:
        """Load a previously saved index."""
        embeddings = np.load(directory / "embeddings.npy")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        entries = [NoteEntry(**e) for e in metadata["entries"]]
        return cls(
            entries=entries,
            embeddings=embeddings,
            model_name=metadata["model_name"],
        )


# The default model: 384-dim vectors, ~80MB, fast on CPU
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _prepare_text(entry_path: str, title: str, tags: list[str], body: str) -> str:
    """Combine note fields into a single string for embedding.

    We concatenate title, tags, and body so the embedding captures all
    facets of the note's meaning. Title is repeated to give it extra weight
    since it's the strongest signal of what a note is "about".
    """
    parts = []
    if title:
        parts.append(title)
        parts.append(title)  # repeat for emphasis
    if tags:
        parts.append(" ".join(tags))
    # Truncate body to avoid overwhelming the model's context window
    # Most sentence-transformers handle 256-512 tokens well
    truncated_body = body[:2000]
    parts.append(truncated_body)
    return "\n".join(parts)


def build_index(
    parsed_notes: list,
    model_name: str = DEFAULT_MODEL,
    snippet_length: int = 200,
) -> VaultIndex:
    """Build a semantic index from parsed notes.

    Args:
        parsed_notes: List of ParsedNote objects from the parser.
        model_name: Sentence-transformer model to use.
        snippet_length: Characters to include in the snippet for LLM context.

    Returns:
        VaultIndex with embeddings and metadata.
    """
    model = SentenceTransformer(model_name)

    entries = []
    texts = []

    for note in parsed_notes:
        tags = note.frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        # Coerce all tag values to strings (some frontmatter has ints)
        tags = [str(t) for t in tags] if tags else []

        entry = NoteEntry(
            path=note.path,
            title=note.title,
            tags=tags,
            wiki_links=note.wiki_links,
            snippet=note.body[:snippet_length].strip(),
        )
        entries.append(entry)

        text = _prepare_text(note.path, note.title, tags, note.body)
        texts.append(text)

    # Batch encode all notes at once — much faster than one-by-one
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return VaultIndex(
        entries=entries,
        embeddings=np.array(embeddings),
        model_name=model_name,
    )
