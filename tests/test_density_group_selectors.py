"""Density Analysis group drop-downs follow factor assignments made in-session.

On a fresh project the Density tab was populated once at load, before any
factors existed, and was left out of the factor-change refresh. Factors typed
into the Summary table then reached the Graphs tab but never the Density
tab, whose Group A/B and specific-group lists stayed at "(all sessions)".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.tabs.behavior_analytics_tab import FACET_SPLIT, BehaviorAnalyticsTab  # noqa: E402

LABELS = ["m1", "m2", "m3", "m4"]


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _items(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def test_factors_assigned_after_load_reach_density_tab(_app, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    tab = BehaviorAnalyticsTab()
    tab.set_project(tmp_path)
    tab._load_group_state()
    tab._density_tab.refresh_selectors()  # load-time population, no factors yet
    density = tab._density_tab
    assert _items(density._diff_group_a_combo) == ["(all sessions)"]

    # User adds a factor and assigns levels, as the Summary table does.
    tab._factor_definitions.append("Drug")
    for label, level in zip(LABELS, ["Saline", "Saline", "Fent", "Fent"]):
        tab._session_factors[label] = {"Drug": level}
    tab._sync_session_groups()
    tab._refresh_group_selectors()

    assert density._diff_factor_combo.currentData() == "Drug"
    assert _items(density._diff_group_a_combo) == ["Fent", "Saline"]
    assert density._diff_group_a_combo.currentText() != density._diff_group_b_combo.currentText()
    assert _items(density._dm_specific_group_combo) == ["Fent", "Saline"]


def test_explicit_top_level_choice_is_kept(_app, tmp_path: Path) -> None:
    (tmp_path / "derived").mkdir()
    tab = BehaviorAnalyticsTab()
    tab.set_project(tmp_path)
    tab._load_group_state()
    tab._factor_definitions.append("Drug")
    for label, level in zip(LABELS, ["Saline", "Saline", "Fent", "Fent"]):
        tab._session_factors[label] = {"Drug": level}
    # Graphs tab splits on Drug, so the top-level groups are already usable.
    tab._facet_controls = {"Drug": FACET_SPLIT}
    tab._sync_session_groups()
    tab._density_tab.refresh_selectors()

    density = tab._density_tab
    assert density._diff_factor_combo.currentData() == ""
    assert _items(density._diff_group_a_combo) == ["Fent", "Saline"]
