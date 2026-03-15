# ABOUTME: Tests for the wiki-link graph that enables relationship-aware boosting.
# ABOUTME: Validates bidirectional links and multi-hop neighbor discovery.

from vault_recommender.graph import LinkGraph, build_graph
from vault_recommender.parser import ParsedNote


class TestLinkGraph:
    """Test the wiki-link graph structure and traversal."""

    def _make_graph(self) -> LinkGraph:
        """Build a small test graph: A -> B -> C, A -> D."""
        graph = LinkGraph()
        graph.add_note("notes/a.md", ["notes/b", "notes/d"])
        graph.add_note("notes/b.md", ["notes/c"])
        graph.add_note("notes/c.md", [])
        graph.add_note("notes/d.md", [])
        return graph

    def test_direct_link_detected(self):
        graph = self._make_graph()
        assert graph.are_linked("notes/a.md", "notes/b.md")

    def test_reverse_link_detected(self):
        graph = self._make_graph()
        # B doesn't link to A, but A links to B — should still be "linked"
        assert graph.are_linked("notes/b.md", "notes/a.md")

    def test_unlinked_notes(self):
        graph = self._make_graph()
        assert not graph.are_linked("notes/c.md", "notes/d.md")

    def test_1hop_neighbors(self):
        graph = self._make_graph()
        neighbors = graph.neighbors("notes/a.md", max_hops=1)
        assert "notes/b" in neighbors
        assert "notes/d" in neighbors
        assert neighbors["notes/b"] == 1

    def test_2hop_neighbors(self):
        graph = self._make_graph()
        neighbors = graph.neighbors("notes/a.md", max_hops=2)
        # C is 2 hops from A (through B)
        assert "notes/c" in neighbors
        assert neighbors["notes/c"] == 2

    def test_build_graph_from_parsed_notes(self):
        notes = [
            ParsedNote(path="x.md", body="", wiki_links=["y"]),
            ParsedNote(path="y.md", body="", wiki_links=["z"]),
        ]
        graph = build_graph(notes)
        assert graph.are_linked("x.md", "y.md")
        assert not graph.are_linked("x.md", "z.md")
