"""Label-needs advisor: leak-free ablation verdicts per behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import abel.utils.gpu_feature_ops as gpu_ops
from abel.services.label_needs_service import (
    BehaviorNeeds,
    LabelNeedsConfig,
    _verdict,
    analyze_label_needs,
)
from abel.utils.cancellation import OperationCancelled

NAMES = {"a": "Attack", "b": "Boxing", "r": "Rear"}


def _needs(**kw) -> BehaviorNeeds:
    base = dict(behavior_id="b", behavior_name="Boxing", positives=120, sessions_with_positives=28,
                sessions_total=30, top3_session_share=0.2, prauc_full=0.7, gain_positives=0.0,
                gain_spread=0.0, gain_hard=0.0, gain_no_behavior=0.0, confusers=["a"])
    base.update(kw)
    return BehaviorNeeds(**base)


def test_positives_from_new_subjects_when_concentrated() -> None:
    n = _needs(gain_positives=0.06, sessions_with_positives=6, top3_session_share=0.75)
    _verdict(n, NAMES)
    assert n.verdict == "Label more"
    assert "subjects that have none yet" in n.actions[0]


def test_well_covered_behavior_just_needs_more_positives() -> None:
    n = _needs(gain_positives=0.05)
    _verdict(n, NAMES)
    assert n.actions == ["More positives"]


def test_hard_negatives_name_the_confusers() -> None:
    n = _needs(gain_hard=0.07, confusers=["a", "r"])
    _verdict(n, NAMES)
    assert any("Attack, Rear" in a for a in n.actions)


def test_flat_weak_model_is_not_told_to_label_more() -> None:
    # Allogroom on COACIECNO: 0.33 PR-AUC at 25% and 100% of positives alike.
    n = _needs(prauc_full=0.33, gain_positives=0.01)
    _verdict(n, NAMES)
    assert n.verdict == "Flat"
    assert "merging" in n.actions[0]


def test_strong_model_with_no_gains_is_saturated() -> None:
    n = _needs(prauc_full=0.9)
    _verdict(n, NAMES)
    assert n.verdict == "Saturated"


def _toy(n_sessions: int = 8, per_session: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for i in range(per_session):
            lab = ["no_behavior", "a", "b", "r"][i % 4]
            x = rng.normal(0, 1, 6)
            x[0] += 2.5 if lab == "b" else 0.0
            x[1] += 1.5 if lab in ("a", "b") else 0.0  # Attack looks a bit like Boxing
            rows.append({"segment_id": f"s{s}_{i}", "session_id": f"s{s}", "animal_id": "track_0",
                         "label": lab, **{f"f{k}": float(v) for k, v in enumerate(x)}})
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(gpu_ops, "gpu_available", lambda: False)


def test_end_to_end_runs_and_reports_progress(tmp_path) -> None:
    seen = []
    out = analyze_label_needs(
        tmp_path, [("b", "Boxing"), ("zz", "Never labeled")],
        LabelNeedsConfig(folds=4, repeats=1, n_estimators=20),
        progress_cb=lambda name, done, total: seen.append((done, total)),
        training_set=_toy(),
    )
    boxing, missing = out
    assert boxing.prauc_full is not None and boxing.prauc_full > 0.5
    assert boxing.verdict in {"Label more", "Saturated", "Flat"}
    assert boxing.sessions_with_positives == 8
    assert missing.verdict == "Too few"
    assert seen[-1][0] == seen[-1][1]  # skipped behaviors still advance the bar to 100%


def test_cancel_stops_the_run(tmp_path) -> None:
    with pytest.raises(OperationCancelled):
        analyze_label_needs(tmp_path, [("b", "Boxing")], LabelNeedsConfig(folds=4, repeats=1, n_estimators=5),
                            cancel_flag=[True], training_set=_toy())


# -- dialog tab --------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.clip_mining_dialog import ClipMiningDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _dialog(tmp_path) -> ClipMiningDialog:
    dlg = ClipMiningDialog(project_root=tmp_path, exemplar_provider=list, scope_label="test",
                           on_apply=lambda refs, scores: None, reviewed_provider=set)
    dlg._gap_behavior.clear()
    dlg._gap_behavior.addItem("Boxing", "b")
    dlg._gap_chk.setEnabled(True)
    return dlg


def test_results_fill_the_table_and_save_a_csv(_app, tmp_path) -> None:
    dlg = _dialog(tmp_path)
    tabs = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
    assert "Label Needs" in tabs
    n = _needs(gain_positives=0.06, sessions_with_positives=6, top3_session_share=0.75)
    _verdict(n, NAMES)
    dlg._on_label_needs_ready([n])
    assert dlg._needs_table.rowCount() == 1
    assert dlg._needs_table.item(0, 1).text() == "Label more"
    assert dlg._needs_table.item(0, 5).text() == "+0.060"  # +Pos
    assert list((tmp_path / "derived" / "label_needs").glob("label_needs_*.csv"))


def test_hunt_gaps_arms_the_coverage_filter(_app, tmp_path) -> None:
    dlg = _dialog(tmp_path)
    dlg._on_label_needs_ready([_needs()])
    dlg._needs_table.selectRow(0)
    dlg._needs_hunt_gaps()
    assert dlg._gap_chk.isChecked()
    assert dlg._gap_behavior.currentData() == "b"
    assert dlg._tabs.currentIndex() == 0


def test_missing_training_set_reports_instead_of_crashing(_app, tmp_path) -> None:
    dlg = _dialog(tmp_path)
    try:
        dlg._label_needs_job([("b", "Boxing")], [("b", "Boxing")], LabelNeedsConfig(), [False])
    except FileNotFoundError:
        import traceback
        dlg._on_label_needs_failed(traceback.format_exc())
    assert "Train a model first" in dlg._needs_status.text()


def test_flat_model_is_not_sent_hunting_for_new_subjects() -> None:
    # Allogroom: spread +0.039 and concentrated, but 0.33 PR-AUC that more positives don't lift.
    n = _needs(prauc_full=0.327, gain_positives=0.012, gain_spread=0.039, top3_session_share=0.58)
    _verdict(n, NAMES)
    assert n.verdict == "Flat"
