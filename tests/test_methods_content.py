"""Tests for the Methods tab content (references + formulas).

Guards the publication documentation: every reference must have a link, every
formula must name an implementing function, cross-references must resolve, and the
rendered HTML must be well-formed.
"""

from __future__ import annotations

import importlib
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
        assert r.url.startswith("http"), f"{r.key} has no well-formed URL"
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


def test_formula_sources_resolve_to_real_code() -> None:
    """Every formula's ``source`` must import.

    A prefix check ("starts with abel.") cannot tell a real function from a
    fictional one, and three formulas once named functions that did not exist.
    Resolve the path for real so the Methods tab cannot drift from the code.
    """
    for f in mc.FORMULAS:
        parts = f.source.split(".")
        for split in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            for attr in parts[split:]:
                assert hasattr(obj, attr), (
                    f"{f.name}: '{f.source}' does not resolve, no attribute "
                    f"'{attr}' on {obj!r}"
                )
                obj = getattr(obj, attr)
            break
        else:  # pragma: no cover - only fires on a wholly bogus module path
            raise AssertionError(f"{f.name}: no importable module in '{f.source}'")


def test_every_reference_is_used() -> None:
    """No orphan citations: each reference backs at least one formula, or names a
    pose format / library ABEL consumes rather than a procedure it computes.

    Keep ``context_only`` short. It is an exemption from "this citation justifies
    something we compute", not a parking space for related work.
    """
    cited = {k for f in mc.FORMULAS for k in f.refs}
    # Input formats and libraries, not procedures: DeepLabCut and SLEAP are the
    # pose formats ABEL reads; UMAP is the embedding library behind the motif
    # presets and the Active Learning separation plots.
    context_only = {
        "mathis2018",
        "pereira2022",
        "mcinnes2018",
    }
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
