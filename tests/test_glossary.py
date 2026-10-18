"""Glossary content rules and the Info > Glossary search."""

from __future__ import annotations

import re

import pytest

from abel.ui.glossary import ENTRIES, entry, search

# Phrasing the user strips out of any text by hand.
_BANNED = re.compile(
    r"\b(delve|seamless(ly)?|robust|leverage|comprehensive|utili[sz]e)\b", re.IGNORECASE
)


def test_terms_unique() -> None:
    seen: set[str] = set()
    for e in ENTRIES:
        key = e.term.lower()
        assert key not in seen, e.term
        seen.add(key)


def test_see_also_targets_exist() -> None:
    for e in ENTRIES:
        for t in e.see:
            assert entry(t) is not None, f"{e.term} -> {t}"


def test_text_style() -> None:
    for e in ENTRIES:
        for s in (e.term, e.where, e.text, *e.aliases):
            assert "—" not in s and "–" not in s, e.term
        assert e.text.strip(), e.term
        assert e.where.strip(), e.term
        assert not _BANNED.search(e.text), e.term


def test_every_tab_has_an_entry() -> None:
    tabs = [
        "Home", "Data Import", "Behaviors", "ROI", "Features", "Active Learning",
        "Temporal", "Analytics", "Validation", "Export", "Direct Use",
        "Model Refinement", "Settings", "Info", "Seeds", "Learning", "Clips",
        "Refinement", "Graphs", "Spatial Heatmap", "Density Analysis",
        "Behavior Relationships", "Session Sections", "Velocity",
        "Social Interaction", "Validation Quiz", "Behavior Grid", "Feature Audit",
        "Review Effort", "Apply Models", "Transfer Feedback", "Preferences",
        "Dependencies", "Logs", "Glossary", "Methods", "Version History",
    ]
    for t in tabs:
        assert search(t), t


def test_exact_term_ranks_first() -> None:
    assert search("stride")[0].term == "Stride"
    assert search("Retrain")[0].term == "Retrain"
    assert search("merge gap")[0].term == "Merge gap"


def test_alias_and_punctuation_insensitive() -> None:
    assert search("LOSO")[0].term == "Leave-one-mouse-out CV"
    assert search("pr auc")[0].term == "PR-AUC"
    assert search("prauc")[0].term == "PR-AUC"
    assert search("px mm")[0].term == "px/mm"
    assert search("review")[0].term == "Review (Active Learning)"


def test_falls_back_to_text_and_handles_misses() -> None:
    assert any(e.term == "Hard negative" for e in search("confident mistakes"))
    assert search("zzzz not a word") == []
    assert len(search("")) == len(ENTRIES)


pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.glossary_tab import GlossaryTab  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_tab_filters_and_shows_first_match(app) -> None:
    tab = GlossaryTab()
    assert tab._list.count() == len(ENTRIES)
    tab._search.setText("inhibition")
    assert tab._list.currentItem().text() == "Inhibition weight"
    assert "competing" in tab._detail.toPlainText() or "compete" in tab._detail.toPlainText()
    tab._search.setText("zzzz not a word")
    assert tab._list.count() == 0
    assert "No match" in tab._detail.toPlainText()
