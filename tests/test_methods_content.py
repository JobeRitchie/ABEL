"""Tests for the Methods tab content (references + formulas).

Guards the publication documentation: every reference must have a link, every
formula must name an implementing function, cross-references must resolve, and the
rendered HTML must be well-formed.
"""

from __future__ import annotations

from html.parser import HTMLParser

import abel.ui.methods_content as mc


# Void/self-closing HTML elements that need no closing tag.
_VOID = {"br", "hr", "img", "input", "meta", "link"}


class _WellFormed(HTMLParser):
    """Minimal balance checker: every non-void open tag gets a matching close."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.ok = True

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.ok = False
        else:
            self.stack.pop()


def _assert_balanced(html: str) -> None:
    p = _WellFormed()
    p.feed(html)
    assert p.ok, "unbalanced tags"
    assert not p.stack, f"unclosed tags: {p.stack}"


def test_references_nonempty_and_linked() -> None:
    assert len(mc.REFERENCES) >= 15
    keys = [r.key for r in mc.REFERENCES]
    assert len(keys) == len(set(keys)), "duplicate reference keys"
    for r in mc.REFERENCES:
        assert r.url.startswith("http"), f"{r.key} has no resolvable link"
        assert r.authors and r.year and r.title and r.venue
        assert r.used_for, f"{r.key} does not say what it is used for"


def test_formulas_reference_existing_sources_and_refs() -> None:
    assert len(mc.FORMULAS) >= 20
    ref_keys = {r.key for r in mc.REFERENCES}
    for f in mc.FORMULAS:
        assert f.name and f.formula_html and f.description
        assert f.source.startswith("abel."), f"{f.name} source not an abel path"
        for key in f.refs:
            assert key in ref_keys, f"{f.name} cites unknown reference '{key}'"


def test_every_reference_is_used() -> None:
    """No orphan citations: each reference backs at least one formula or is a
    study-design/agreement source surfaced only in the References list."""
    cited = {k for f in mc.FORMULAS for k in f.refs}
    # These appear in the References list for context but need not tag a formula.
    context_only = {"chicco2020", "stone1974", "mcinnes2018", "wilcoxon1945"}
    for r in mc.REFERENCES:
        assert r.key in cited or r.key in context_only, f"orphan reference {r.key}"


def test_render_references_html_well_formed() -> None:
    html = mc.render_references_html()
    assert "http" in html
    # Every reference title should appear.
    for r in mc.REFERENCES:
        assert r.url in html
    _assert_balanced(html)


def test_render_formulas_html_well_formed() -> None:
    html = mc.render_formulas_html()
    for f in mc.FORMULAS:
        assert f.source in html
    _assert_balanced(html)
