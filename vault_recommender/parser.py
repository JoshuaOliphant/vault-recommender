# ABOUTME: Parses markdown notes into structured data (frontmatter, body, wiki-links).
# ABOUTME: First stage of the recommender pipeline — extracts features from raw markdown.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ParsedNote:
    """Structured representation of a parsed markdown note."""

    path: str
    body: str
    frontmatter: dict = field(default_factory=dict)
    wiki_links: list[str] = field(default_factory=list)
    title: str = ""

    def __post_init__(self):
        if not self.title and self.frontmatter.get("title"):
            self.title = self.frontmatter["title"]
        elif not self.title:
            # Infer title from first heading or filename
            heading = re.search(r"^#\s+(.+)$", self.body, re.MULTILINE)
            if heading:
                self.title = heading.group(1).strip()
            else:
                self.title = Path(self.path).stem if self.path else ""


# Matches YAML frontmatter between --- delimiters at the start of the file
_FRONTMATTER_RE = re.compile(r"\A---\n(.+?)\n---\n?", re.DOTALL)

# Matches [[target]] and [[target|display text]] wiki-links
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_note(content: str, path: str = "") -> ParsedNote:
    """Parse a markdown note and extract frontmatter, body text, and wiki-links.

    Args:
        content: Raw markdown string.
        path: File path (used for title inference and identity).

    Returns:
        ParsedNote with extracted fields.
    """
    frontmatter = {}
    body = content

    # Extract YAML frontmatter if present
    fm_match = _FRONTMATTER_RE.match(content)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            frontmatter = {}
        body = content[fm_match.end() :]

    # Extract wiki-links, keeping only the target (left of pipe)
    raw_links = _WIKILINK_RE.findall(body)
    wiki_links = [link.split("|")[0].strip() for link in raw_links]

    return ParsedNote(
        path=path,
        body=body.strip(),
        frontmatter=frontmatter,
        wiki_links=wiki_links,
    )


def parse_vault(
    vault_path: Path, exclude_patterns: list[str] | None = None
) -> list[ParsedNote]:
    """Parse all markdown files in a vault directory.

    Args:
        vault_path: Root directory of the Obsidian vault.
        exclude_patterns: Glob patterns to skip (e.g., ['.git', 'node_modules']).

    Returns:
        List of ParsedNote objects for all .md files.
    """
    if exclude_patterns is None:
        exclude_patterns = [
            ".git",
            ".obsidian",
            ".claude",
            "node_modules",
            ".venv",
            "__pycache__",
        ]

    notes = []
    for md_file in sorted(vault_path.rglob("*.md")):
        # Skip excluded directories
        if any(part in md_file.parts for part in exclude_patterns):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        rel_path = str(md_file.relative_to(vault_path))
        notes.append(parse_note(content, path=rel_path))

    return notes
