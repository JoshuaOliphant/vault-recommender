# ABOUTME: Builds the embedding index — converts parsed notes into semantic vectors.
# ABOUTME: Uses sentence-transformers for local embedding, stores index as numpy + JSON.

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
    built_at: datetime | None = None

    def __post_init__(self) -> None:
        if len(self.entries) != self.embeddings.shape[0]:
            raise ValueError(
                f"entries ({len(self.entries)}) and embeddings "
                f"({self.embeddings.shape[0]}) must have the same length"
            )

    def save(self, directory: Path) -> None:
        """Persist index to disk as numpy array + JSON metadata.

        Always stamps built_at with the current time and updates the instance
        to stay consistent with what was written to disk.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        self.built_at = datetime.now(timezone.utc)
        metadata = {
            "model_name": self.model_name,
            "built_at": self.built_at.isoformat(),
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
        built_at = None
        built_at_str = metadata.get("built_at")
        if built_at_str:
            parsed = datetime.fromisoformat(built_at_str)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            built_at = parsed
        return cls(
            entries=entries,
            embeddings=embeddings,
            model_name=metadata["model_name"],
            built_at=built_at,
        )

    @classmethod
    def is_stale(cls, index_dir: Path, vault_path: Path) -> bool:
        """Check if the index is older than the newest .md file in the vault.

        Returns True (stale) when:
        - The index directory or metadata.json doesn't exist
        - The metadata is corrupted or has no built_at timestamp
        - The vault path doesn't exist
        - Any .md file in the vault is newer than built_at
        """
        meta_path = index_dir / "metadata.json"
        if not meta_path.exists():
            return True

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"Warning: index metadata is corrupted ({exc}). Treating as stale.",
                file=sys.stderr,
            )
            return True

        built_at_str = metadata.get("built_at")
        if not built_at_str:
            return True

        try:
            built_at = datetime.fromisoformat(built_at_str)
        except ValueError:
            return True  # Unrecognizable timestamp — force rebuild

        # Ensure built_at is timezone-aware for comparison
        if built_at.tzinfo is None:
            built_at = built_at.replace(tzinfo=timezone.utc)

        if not vault_path.is_dir():
            print(
                f"Warning: vault path does not exist: {vault_path}",
                file=sys.stderr,
            )
            return True

        # Walk vault for the most recently modified .md file
        max_mtime: float = 0.0
        for md_file in vault_path.rglob("*.md"):
            mtime = md_file.stat().st_mtime
            if mtime > max_mtime:
                max_mtime = mtime

        if max_mtime == 0.0:
            return False  # No .md files in vault — nothing to index

        newest_file_time = datetime.fromtimestamp(max_mtime, tz=timezone.utc)
        return newest_file_time > built_at


# The default model: 384-dim vectors, ~80MB, fast on CPU
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _prepare_text(
    entry_path: str,
    title: str,
    tags: list[str],
    body: str,
    wiki_links: list[str] | None = None,
) -> str:
    """Combine note fields into a labeled string for embedding.

    Assembles directory topic, title, tags, wiki-links, and a short body
    excerpt with section labels. Shorter body (300 chars) concentrates
    signal in structured metadata fields for better recall.
    """
    parts = []
    if entry_path:
        from pathlib import Path as _Path

        directory = str(_Path(entry_path).parent)
        if directory != ".":
            parts.append(f"Topic: {directory}")
    if title:
        parts.append(f"Title: {title}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if wiki_links:
        parts.append(f"Related: {', '.join(wiki_links)}")
    # Truncate body to avoid overwhelming the model's context window
    # Most sentence-transformers handle 256-512 tokens well
    truncated_body = body[:300]
    parts.append(f"Body: {truncated_body}")
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

        text = _prepare_text(note.path, note.title, tags, note.body, note.wiki_links)
        texts.append(text)

    # Batch encode all notes at once — much faster than one-by-one
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return VaultIndex(
        entries=entries,
        embeddings=np.array(embeddings),
        model_name=model_name,
    )
