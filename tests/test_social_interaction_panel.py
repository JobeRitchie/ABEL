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


class _Table:
    def __init__(self, n: int) -> None:
        self._n = n

    def rowCount(self) -> int:
        return self._n


class _Summary:
    """Stands in for the Analytics subject list with a tick set."""

    def __init__(self, ticked: set[str]) -> None:
        self.ticked = set(ticked)
        self._session_table = _Table(3)

    def _checked_subjects(self) -> set[str]:
        return set(self.ticked)


def test_unticked_subjects_are_left_out_of_fit_views_and_exports(qapp, monkeypatch):
    pytest.importorskip("hmmlearn")
    pytest.importorskip("matplotlib")
    import abel.ui.tabs.social_interaction_panel as mod

    class _SyncPool:
        @staticmethod
        def globalInstance():
            return _SyncPool()

        def start(self, worker):
            worker.run()

    monkeypatch.setattr(mod, "QThreadPool", _SyncPool)
    fitted: list[set[str]] = []
    real_fit = SocialAnalysisService.fit_dominance_hmm

    def _fit_spy(self, df, **kw):
        fitted.append(set(df["session_id"].astype(str)))
        return real_fit(self, df, **{**kw, "n_states": 3, "n_restarts": 1, "bin_s": 5.0})

    monkeypatch.setattr(SocialAnalysisService, "fit_dominance_hmm", _fit_spy)

    host = _Host()
    host._summary_tab = _Summary({"M1", "M3"})
    w = mod.SocialInteractionWidget(host)
    monkeypatch.setattr(w, "_load_frames", _synthetic_cohort)
    w._run_dominance_hmm()

    # M2 (s1) never reaches the fit, so it cannot shape the pooled states.
    assert fitted == [{"s0", "s2"}]
    assert w._res["unticked_sessions"] == ["s1"]
    assert {r["session_label"] for r in w._current_dominance()} == {"M1", "M3"}
    assert "This fit" not in w._status.text()

    # Unticking a fitted subject hides it everywhere and asks for a refit.
    host._summary_tab.ticked = {"M1"}
    w.on_subjects_changed()
    assert {r["session_label"] for r in w._current_dominance()} == {"M1"}
    tables = w.result_tables()
    for stem in ("social_dominance", "social_dominance_bins", "social_state_occupancy"):
        assert set(tables[stem]["session_id"]) <= {"s0"}, stem
    assert [w._session_combo.itemData(i) for i in range(w._session_combo.count())] == ["s0"]
    assert "includes unticked subject(s) M3" in w._status.text()

    # Re-ticking the subject left out of the fit is flagged too; the note is
    # replaced, not stacked.
    host._summary_tab.ticked = {"M1", "M2", "M3"}
    w.on_subjects_changed()
    status = w._status.text()
    assert "left out now-ticked subject(s) M2" in status
    assert status.count("This fit") == 1
    w.deleteLater()


def test_fit_reports_every_em_iteration_with_rising_fraction():
    pytest.importorskip("hmmlearn")
    calls: list[tuple[str, float]] = []
    res = SocialAnalysisService().fit_dominance_hmm(
        _synthetic_cohort(), fps=30.0, n_states=3, n_restarts=2, bin_s=5.0,
        progress_cb=lambda msg, frac: calls.append((msg, frac)),
    )
    assert res["error"] is None
    fracs = [f for _m, f in calls]
    assert fracs == sorted(fracs) and 0.0 <= fracs[0] and fracs[-1] <= 1.0
    iters = [m for m, _f in calls if "EM iteration" in m]
    assert any("restart 1 of 2" in m for m in iters) and any("restart 2 of 2" in m for m in iters)
    # Once a restart has finished, the second one carries a time estimate.
    assert all(("left" in m or "almost done" in m) for m in iters if "restart 2 of 2" in m)


def test_cancel_stops_the_fit_and_keeps_the_previous_result(qapp, monkeypatch):
    pytest.importorskip("hmmlearn")
    import abel.ui.tabs.social_interaction_panel as mod

    class _SyncPool:
        @staticmethod
        def globalInstance():
            return _SyncPool()

        def start(self, worker):
            worker.run()

    monkeypatch.setattr(mod, "QThreadPool", _SyncPool)
    w = mod.SocialInteractionWidget(_Host())
    w._res = {"sentinel": True}
    monkeypatch.setattr(w, "_load_frames", _synthetic_cohort)
    # Press Cancel on the first progress report.
    real = w._on_progress

    def _press_cancel(msg, frac):
        real(msg, frac)
        w._cancel_hmm()

    w._progress.update.disconnect()
    w._progress.update.connect(_press_cancel)
    w._run_dominance_hmm()
    assert "cancelled" in w._status.text()
    assert w._res == {"sentinel": True}
    assert not w._progress_bar.isVisible() and w._worker is None
    w.deleteLater()
