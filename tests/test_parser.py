# ABOUTME: Tests for the markdown parser that extracts frontmatter, body, and wiki-links.
# ABOUTME: First component in the recommender pipeline.

from vault_recommender.parser import parse_note, parse_vault


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

    def test_malformed_frontmatter_returns_empty_dict(self):
        # Unbalanced quotes / bad YAML inside the --- block must not crash
        content = '---\ntitle: "unterminated\ntags: [a, b\n---\n\nbody'
        result = parse_note(content)
        assert result.frontmatter == {}
        assert "body" in result.body

    def test_title_falls_back_to_empty_when_no_path_no_heading(self):
        result = parse_note("plain body")
        assert result.title == ""


class TestParseVault:
    """Test the directory walker that turns a vault into ParsedNote objects."""

    def test_parses_all_markdown_files(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n[[b]]")
        (tmp_path / "b.md").write_text("# B\nbody")
        (tmp_path / "ignored.txt").write_text("not markdown")

        notes = parse_vault(tmp_path)
        paths = sorted(n.path for n in notes)
        assert paths == ["a.md", "b.md"]

    def test_skips_default_exclude_patterns(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "x.md").write_text("# x")
        (tmp_path / ".obsidian").mkdir()
        (tmp_path / ".obsidian" / "y.md").write_text("# y")
        (tmp_path / "real.md").write_text("# real")

        notes = parse_vault(tmp_path)
        assert [n.path for n in notes] == ["real.md"]

    def test_custom_exclude_patterns(self, tmp_path):
        (tmp_path / "drafts").mkdir()
        (tmp_path / "drafts" / "wip.md").write_text("# wip")
        (tmp_path / "kept.md").write_text("# kept")

        notes = parse_vault(tmp_path, exclude_patterns=["drafts"])
        assert [n.path for n in notes] == ["kept.md"]

    def test_skips_files_with_undecodable_bytes(self, tmp_path):
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"\xff\xfe\x00not valid utf-8")
        good = tmp_path / "good.md"
        good.write_text("# good")

        notes = parse_vault(tmp_path)
        assert [n.path for n in notes] == ["good.md"]
