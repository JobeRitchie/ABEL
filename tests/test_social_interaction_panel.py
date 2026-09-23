"""Headless smoke test: every dominance chart view draws from a real fit."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.services.social_analysis_service import SocialAnalysisService  # noqa: E402
from test_social_analysis_service import _synthetic_cohort  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Host:
    _project_root = None
    _session_label_by_session = {"s0": "M1", "s1": "M2", "s2": "M3"}
    # Keyed by session *label*, as the Analytics tab stores it.
    _session_groups = {"M1": "ctrl", "M2": "drug", "M3": "ctrl"}
    _graph_settings = {"fig_bg": "#ffffff", "dpi": 100}

    def _project_fps(self) -> float:
        return 30.0

    def _group_color(self, group: str, index: int) -> str:
        return ["#5b9bd5", "#ed7d31"][index % 2]


def test_every_view_draws_and_tables_export(qapp):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("matplotlib")
    from abel.ui.tabs.social_interaction_panel import _VIEWS, SocialInteractionWidget

    res = SocialAnalysisService().fit_dominance_hmm(
        _synthetic_cohort(), fps=30.0, n_states=3, n_restarts=1, bin_s=5.0,
    )
    assert res["error"] is None
    w = SocialInteractionWidget(_Host())
    w._show_result(res, restored=False)

    # Groups come from the label-keyed host map, not the raw session id.
    groups = {r["session_label"]: r["group"] for r in w._current_dominance()}
    assert groups == {"M1": "ctrl", "M2": "drug", "M3": "ctrl"}
    assert w._dom_table.rowCount() == 6

    for i, (key, _label) in enumerate(_VIEWS):
        w._view_combo.setCurrentIndex(i)
        axes = w._fig.get_axes()
        assert axes, key
        texts = [t.get_text() for ax in axes for t in ax.texts]
        assert "Could not draw this view; see the log." not in texts, key

    tables = w.result_tables()
    for stem in (
        "social_dominance", "social_dominance_bins", "social_displacement_events",
        "social_state_profiles", "social_state_occupancy", "social_state_switching",
    ):
        assert stem in tables and not tables[stem].empty, stem
    assert "Ranked first by track id: track_0 3" in w._stats_text.toPlainText()
    w.deleteLater()


class _FacetHost(_Host):
    """Host with Graphs-style factors: Drug (ctrl/drug) x Sex (M/F)."""

    _factor_definitions = ["Drug", "Sex"]
    _session_factors = {
        "M1": {"Drug": "ctrl", "Sex": "M"},
        "M2": {"Drug": "drug", "Sex": "F"},
        "M3": {"Drug": "ctrl", "Sex": "F"},
    }

    def _session_groups_for_controls(self, controls):
        from abel.ui.tabs.behavior_analytics_tab import facet_session_labels
        return facet_session_labels(self._session_factors, self._factor_definitions, controls)[0]

    def _split_factors_for_controls(self, controls):
        from abel.ui.tabs.behavior_analytics_tab import FACET_SPLIT
        return [f for f in self._factor_definitions if controls.get(f) == FACET_SPLIT]

    def _ordered_group_list(self, available, split_factors=None):
        # Saved level order puts "drug" first, to prove it is not alphabetical.
        return sorted(available, key=lambda g: (0 if g.startswith("drug") else 1, g))

    def _levels_by_factor(self):
        return {"Drug": ["drug", "ctrl"], "Sex": ["M", "F"]}

    def _default_facet_controls(self):
        from abel.ui.tabs.behavior_analytics_tab import FACET_COMBINE, FACET_SPLIT
        return {"Drug": FACET_SPLIT, "Sex": FACET_COMBINE}


def _fit():
    return SocialAnalysisService().fit_dominance_hmm(
        _synthetic_cohort(), fps=30.0, n_states=3, n_restarts=1, bin_s=5.0,
    )


def test_facets_split_filter_and_compare_groups(qapp):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("matplotlib")
    from abel.ui.tabs.behavior_analytics_tab import FACET_SPLIT
    from abel.ui.tabs.social_interaction_panel import SocialInteractionWidget

    w = SocialInteractionWidget(_FacetHost())
    w._show_result(_fit(), restored=False)
    # Default: split on the first factor, in the host's saved level order.
    assert {r["session_label"]: r["group"] for r in w._current_dominance()} == {
        "M1": "ctrl", "M2": "drug", "M3": "ctrl"}
    assert w._order_groups(["ctrl", "drug"]) == ["drug", "ctrl"]

    # Split both factors -> interaction groups.
    w._facet.set_state({"Drug": FACET_SPLIT, "Sex": FACET_SPLIT})
    w._on_facets_changed()
    assert {r["group"] for r in w._current_dominance()} == {"ctrl × M", "drug × F", "ctrl × F"}

    # A specific level filters sessions out of the table, charts and metrics.
    w._facet.set_state({"Drug": "ctrl", "Sex": FACET_SPLIT})
    w._on_facets_changed()
    assert {r["session_label"] for r in w._current_dominance()} == {"M1", "M3"}
    assert w._dom_table.rowCount() == 4
    dm = w.dyad_metrics()
    assert set(dm["session_label"]) == {"M1", "M3"}
    assert {"steepness", "n_disp", "disp_rate", "inter_min", "occ_0"} <= set(dm.columns)

    w._view_combo.setCurrentIndex(w._view_combo.findData("compare"))
    for i in range(w._metric_combo.count()):
        w._metric_combo.setCurrentIndex(i)
        texts = [t.get_text() for ax in w._fig.get_axes() for t in ax.texts]
        assert "Could not draw this view; see the log." not in texts, w._metric_combo.currentText()
    assert "social_dyad_metrics" in w.result_tables()
    w.deleteLater()


def test_saved_fit_restores_on_reopen_even_when_stale(qapp, tmp_path):
    pytest.importorskip("hmmlearn")
    from abel.ui.tabs.social_interaction_panel import SocialInteractionWidget

    svc = SocialAnalysisService()
    res = _fit()
    svc.save_result(tmp_path, res, "old-fingerprint")

    host = _Host()
    host._project_root = tmp_path
    w = SocialInteractionWidget(host)
    w.on_project_reloaded()
    assert w._res and w._res["n_states"] == res["n_states"]
    assert w._dom_table.rowCount() == 6
    assert "changed since this fit" in w._status.text()
    w.deleteLater()


def test_dominance_by_group_and_prism_table(qapp):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("matplotlib")
    from abel.ui.tabs.social_interaction_panel import SocialInteractionWidget

    w = SocialInteractionWidget(_FacetHost())
    w._show_result(_fit(), restored=False)
    w._view_combo.setCurrentIndex(w._view_combo.findData("dominance"))
    assert w._fig.get_axes()
    wide, key = w.prism_table()
    assert key == "steepness"
    # One column per group in saved level order, one row per session.
    assert list(wide.columns) == ["drug", "ctrl"]
    assert wide["ctrl"].notna().sum() == 2 and wide["drug"].notna().sum() == 1
    w.deleteLater()


def test_pct_dominant_view_and_contingency_prism(qapp):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("matplotlib")
    from abel.ui.tabs.social_interaction_panel import _NO_HIERARCHY, SocialInteractionWidget

    w = SocialInteractionWidget(_FacetHost())
    w._show_result(_fit(), restored=False)
    dm = w.dyad_metrics()
    assert dm["track0_di"].between(-1, 1).all()
    w._view_combo.setCurrentIndex(w._view_combo.findData("pct_dominant"))
    texts = [t.get_text() for ax in w._fig.get_axes() for t in ax.texts]
    assert "Could not draw this view; see the log." not in texts
    wide, key = w.prism_table()
    assert key == "outcome"
    assert list(wide["Group"]) == ["drug", "ctrl"]
    assert wide.columns[-1] == _NO_HIERARCHY
    assert int(wide.drop(columns="Group").to_numpy().sum()) == 3
    w.deleteLater()


def test_prechop_trims_frames_before_the_fit():
    pytest.importorskip("hmmlearn")
    df = _synthetic_cohort()
    svc = SocialAnalysisService()
    trimmed = svc.trim_prechop(df, {"s0": 100})
    assert trimmed.loc[trimmed.session_id == "s0", "frame"].min() >= 100
    assert len(trimmed[trimmed.session_id == "s1"]) == len(df[df.session_id == "s1"])
    res = svc.fit_dominance_hmm(df, fps=30.0, n_states=3, n_restarts=1, bin_s=5.0,
                                prechop_frames={"s0": 100, "s1": 0})
    assert res["settings"]["prechop_frames"] == {"s0": 100}
    ev = [e for e in res["displacement_events"] if e["session_id"] == "s0"]
    assert all(e["start_frame"] >= 100 for e in ev)
