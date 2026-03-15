# ABOUTME: Tests for the markdown parser that extracts frontmatter, body, and wiki-links.
# ABOUTME: First component in the recommender pipeline.

import pytest
from vault_recommender.parser import parse_note


class TestParseNote:
    """Test parsing markdown notes into structured data."""

    def test_extracts_body_text(self):
        content = "# My Note\n\nThis is the body of my note."
        result = parse_note(content)
        assert "This is the body of my note." in result.body

    def test_extracts_yaml_frontmatter(self):
        content = "---\ntitle: Test Note\ntags: [python, ai]\n---\n\n# Content here"
        result = parse_note(content)
        assert result.frontmatter["title"] == "Test Note"
        assert result.frontmatter["tags"] == ["python", "ai"]

    def test_no_frontmatter_returns_empty_dict(self):
        content = "# Just a heading\n\nNo frontmatter here."
        result = parse_note(content)
        assert result.frontmatter == {}

    def test_extracts_wiki_links(self):
        content = "See [[project-alpha]] and also [[areas/career/README]]."
        result = parse_note(content)
        assert "project-alpha" in result.wiki_links
        assert "areas/career/README" in result.wiki_links

    def test_wiki_links_with_display_text(self):
        content = "Check out [[project-alpha|Project Alpha]] for details."
        result = parse_note(content)
        assert "project-alpha" in result.wiki_links

    def test_no_wiki_links_returns_empty_list(self):
        content = "# Plain note\n\nNo links here."
        result = parse_note(content)
        assert result.wiki_links == []

    def test_body_excludes_frontmatter(self):
        content = "---\ntitle: Secret\n---\n\n# Public Content\n\nVisible body."
        result = parse_note(content)
        assert "title: Secret" not in result.body
        assert "Visible body." in result.body

    def test_title_from_frontmatter(self):
        content = "---\ntitle: My Title\n---\n\n# Heading"
        result = parse_note(content)
        assert result.title == "My Title"

    def test_title_from_heading(self):
        content = "# Heading Title\n\nBody text."
        result = parse_note(content)
        assert result.title == "Heading Title"

    def test_title_from_path(self):
        content = "No heading, no frontmatter."
        result = parse_note(content, path="areas/career/job-search.md")
        assert result.title == "job-search"
