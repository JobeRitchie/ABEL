"""Manual corrections: hand-scored values that replace ABEL's Analytics values.

Covers the stored file, the summary rows, the 'Bouts until behavior' cutoff,
the latency fallback, and the project flag (hidden unless corrections exist).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from abel.services.manual_corrections_service import (
    CORRECTIONS_FILE,
    ManualCorrections,
    flag_text,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEMPLATES = {
    "s1": {"session_id": "s1", "subject": "m1", "session_label": "m1", "session_type": ""},
    "s2": {"session_id": "s2", "subject": "m2", "session_label": "m2", "session_type": ""},
}


def _row(sid: str, bid: str, **metrics: float) -> dict:
    base = {"n_bouts": 2.0, "time_spent_s": 10.0, "mean_bout_s": 5.0, "latency_s": 30.0}
    return {**TEMPLATES[sid], "behavior_id": bid, "behavior": bid.title(),
            "distance_cm": 0.0, **base, **metrics}


def test_round_trip_and_counts(tmp_path: Path) -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 120.0, 30.0)
    mc.set("eat", "s2", "latency_s", 90.0, None)
    mc.set("eat", "s2", "n_bouts", 4, 2.0)
    mc.set("rear", "s1", "n_bouts", 7, 2.0)
    mc.save(tmp_path)

    loaded = ManualCorrections.load(tmp_path)
    assert loaded.counts() == {"eat": 3, "rear": 1}
    assert loaded.get("eat", "s1", "latency_s") == 120.0
    assert loaded.abel_value("eat", "s1", "latency_s") == 30.0
    assert loaded.abel_value("eat", "s2", "latency_s") is None
    assert loaded.latency_by_session("eat") == {"s1": 120.0, "s2": 90.0}


def test_rejects_bad_values() -> None:
    mc = ManualCorrections()
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mc.set("eat", "s1", "latency_s", bad, 1.0)
    with pytest.raises(ValueError):
        mc.set("eat", "s1", "distance_cm", 1.0, 1.0)
    assert not mc


def test_clearing_everything_removes_file(tmp_path: Path) -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 120.0, 30.0)
    mc.save(tmp_path)
    assert (tmp_path / CORRECTIONS_FILE).exists()
    mc.clear("eat", "s1", "latency_s")
    assert not mc
    mc.save(tmp_path)
    assert not (tmp_path / CORRECTIONS_FILE).exists()


def test_apply_replaces_only_corrected_metrics() -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 120.0, 30.0)
    rows = [_row("s1", "eat"), _row("s2", "eat"), _row("s1", "rear")]
    out = mc.apply_to_rows(rows, {"eat", "rear"}, TEMPLATES, {"eat": "Eat"})

    assert out[0]["latency_s"] == 120.0
    assert out[0]["n_bouts"] == 2.0
    assert out[1] is rows[1] and out[2] is rows[2]
    assert rows[0]["latency_s"] == 30.0  # input not mutated


def test_apply_adds_row_where_abel_found_no_bouts() -> None:
    mc = ManualCorrections()
    mc.set("eat", "s2", "latency_s", 200.0, None)
    out = mc.apply_to_rows([_row("s1", "eat")], {"eat"}, TEMPLATES, {"eat": "Eat"})
    added = [r for r in out if r["session_id"] == "s2"]
    assert len(added) == 1
    assert added[0]["latency_s"] == 200.0
    assert added[0]["n_bouts"] == 0.0
    assert added[0]["behavior"] == "Eat"
    assert added[0]["session_label"] == "m2"


def test_apply_skips_unselected_behaviors_and_unknown_sessions() -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 120.0, 30.0)
    mc.set("eat", "gone", "latency_s", 5.0, None)
    rows = [_row("s1", "rear")]
    assert mc.apply_to_rows(rows, {"rear"}, TEMPLATES, {}) is rows
    out = mc.apply_to_rows([_row("s1", "eat")], {"eat"}, TEMPLATES, {})
    assert [r["session_id"] for r in out] == ["s1"]


def test_flag_text_empty_without_corrections(tmp_path: Path) -> None:
    assert flag_text(tmp_path) == ""
    assert flag_text(None) == ""


def test_flag_text_counts_per_behavior_name(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "behavior_definitions.yaml").write_text(
        "behaviors:\n  - behavior_id: eat\n    name: Eat\n", encoding="utf-8",
    )
    mc = ManualCorrections(behavior_names={"old": "Old Behavior"})
    mc.set("eat", "s1", "latency_s", 120.0, 30.0)
    mc.set("eat", "s2", "latency_s", 90.0, 40.0)
    mc.set("old", "s1", "n_bouts", 1, 0.0)
    mc.save(tmp_path)
    text = flag_text(tmp_path)
    assert "Eat: 2" in text and "Old Behavior: 1" in text
    saved = json.loads((tmp_path / CORRECTIONS_FILE).read_text("utf-8"))
    assert saved["corrections"]["eat"]["s1"]["latency_s"]["abel_value"] == 30.0


# ---------------------------------------------------------------- Qt-level

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {exc}")


def _analytics_tab(root: Path):
    from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab

    tab = BehaviorAnalyticsTab()
    tab.set_project(root)
    tab._session_label_by_session = {"s1": "m1", "s2": "m2"}
    tab._fps_by_session = lambda: {"s1": 10.0, "s2": 10.0}  # type: ignore[method-assign]
    tab._session_fps = lambda sid: 10.0  # type: ignore[method-assign]
    tab._selected_behavior_ids = lambda: {"eat", "rear"}  # type: ignore[method-assign]
    tab._summary_rows = [
        _row("s1", "eat"), _row("s1", "rear"), _row("s2", "rear", n_bouts=3.0),
    ]
    tab._raw_bouts = {
        "eat": pd.DataFrame({"session_id": ["s1"], "start_frame": [300], "end_frame": [400],
                             "behavior": ["Eat"]}),
        "rear": pd.DataFrame({
            "session_id": ["s1", "s1", "s2", "s2", "s2"],
            "start_frame": [100, 1500, 100, 1000, 2500],
            "end_frame": [150, 1550, 150, 1050, 2550],
            "behavior": ["Rear"] * 5,
        }),
    }
    return tab


def test_analytics_rows_and_until_cutoff_use_manual_latency(_app, tmp_path: Path) -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 100.0, 30.0)   # ABEL said frame 300 = 30 s
    mc.set("eat", "s2", "latency_s", 200.0, None)   # ABEL found no eating
    mc.save(tmp_path)
    tab = _analytics_tab(tmp_path)
    graphs = tab._graphs_tab

    by_key = {(r["session_id"], r["behavior_id"]): r for r in tab._filtered_rows()}
    assert by_key[("s1", "eat")]["latency_s"] == 100.0
    assert by_key[("s2", "eat")]["latency_s"] == 200.0
    assert by_key[("s1", "rear")]["latency_s"] == 30.0

    # The fallback must not overwrite a manual latency on a zero-bout row.
    kept = graphs._apply_latency_fallbacks(list(by_key.values()), "latency_s")
    assert {(r["session_id"], r["behavior_id"]): r["latency_s"] for r in kept}[("s2", "eat")] == 200.0

    graphs._get_until_behavior_id = lambda: "eat"  # type: ignore[method-assign]
    assert graphs._cutoff_frames_for_until_behavior() == {"s1": 1000.0, "s2": 2000.0}

    graphs._get_first_n_bouts = lambda: 0  # type: ignore[method-assign]
    graphs._is_until_scaling_active = lambda: False  # type: ignore[method-assign]
    pre = {
        (r["session_id"], r["behavior_id"]): r["n_bouts"]
        for r in graphs._recompute_rows_for_first_n(
            [_row("s1", "rear"), _row("s2", "rear")]
        )
    }
    # s1: only the bout at frame 100 precedes 1000; s2: frames 100 and 1000 precede 2000.
    assert pre == {("s1", "rear"): 1.0, ("s2", "rear"): 2.0}


def test_roi_scope_switches_manual_rows_off(_app, tmp_path: Path) -> None:
    mc = ManualCorrections()
    mc.set("eat", "s1", "latency_s", 100.0, 30.0)
    mc.save(tmp_path)
    tab = _analytics_tab(tmp_path)
    tab._roi_scope_zone = 1
    assert {r["latency_s"] for r in tab._filtered_rows() if r["behavior_id"] == "eat"} == {30.0}


def test_dialog_edit_clear_and_reject_bad_input(_app) -> None:
    from abel.ui.manual_corrections_dialog import ManualCorrectionsDialog

    original = ManualCorrections()
    dlg = ManualCorrectionsDialog(
        None, {"eat": "Eat"}, list(TEMPLATES.values()), [_row("s1", "eat")], original,
    )
    lat_col = 4
    assert dlg._table.item(0, lat_col).text() == "30.00"
    assert dlg._table.item(1, lat_col).text() == "-"       # no ABEL bouts for s2

    dlg._table.item(1, lat_col).setText("212.5")
    assert dlg.corrections().get("eat", "s2", "latency_s") == 212.5
    assert dlg._table.item(1, lat_col).font().bold()

    dlg._table.item(0, lat_col).setText("abc")
    assert dlg.corrections().get("eat", "s1", "latency_s") is None
    assert dlg._table.item(0, lat_col).text() == "30.00"

    dlg._table.item(1, lat_col).setText("")
    assert not dlg.corrections()
    assert not original                                     # working copy only


def test_flags_hidden_until_corrections_exist(_app) -> None:
    from abel.ui.tabs.home_tab import HomeTab

    home = HomeTab()
    assert home._manual_flag.isHidden()
    home.set_manual_corrections_flag("Manual corrections in Analytics: Eat: 2")
    assert not home._manual_flag.isHidden()
    home.set_manual_corrections_flag("")
    assert home._manual_flag.isHidden()
