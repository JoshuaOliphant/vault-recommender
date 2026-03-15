# ABOUTME: Builds the wiki-link graph from parsed notes for relationship-aware boosting.
# ABOUTME: Enables 2-hop discovery — notes connected through shared neighbors.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LinkGraph:
    """Bidirectional graph of wiki-link relationships between notes."""

    # Forward links: note path -> set of link targets
    outgoing: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Backlinks: link target -> set of notes that link to it
    incoming: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_note(self, path: str, wiki_links: list[str]) -> None:
        """Register a note and its outgoing wiki-links."""
        # Normalize path by stripping .md extension for matching
        normalized_path = path.removesuffix(".md")
        for link in wiki_links:
            normalized_link = link.removesuffix(".md")
            self.outgoing[normalized_path].add(normalized_link)
            self.incoming[normalized_link].add(normalized_path)

    def neighbors(self, path: str, max_hops: int = 2) -> dict[str, int]:
        """Find all notes reachable within max_hops, with their distance.

        Returns:
            Dict mapping note path -> shortest hop distance (1 = directly linked).
        """
        normalized = path.removesuffix(".md")
        visited: dict[str, int] = {}
        frontier = {normalized}

        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for node in frontier:
                # Follow both directions — outgoing links and backlinks
                connected = self.outgoing.get(node, set()) | self.incoming.get(
                    node, set()
                )
                for neighbor in connected:
                    if neighbor not in visited and neighbor != normalized:
                        visited[neighbor] = hop
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return visited

    def are_linked(self, path_a: str, path_b: str) -> bool:
        """Check if two notes are directly linked in either direction."""
        a = path_a.removesuffix(".md")
        b = path_b.removesuffix(".md")
        return b in self.outgoing.get(a, set()) or a in self.outgoing.get(b, set())


def build_graph(parsed_notes: list) -> LinkGraph:
    """Build a link graph from parsed notes.

    Args:
        parsed_notes: List of ParsedNote objects from the parser.

    Returns:
        LinkGraph with bidirectional link relationships.
    """
    graph = LinkGraph()
    for note in parsed_notes:
        graph.add_note(note.path, note.wiki_links)
    return graph
