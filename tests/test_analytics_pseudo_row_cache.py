"""The distance/ROI rows must survive a round-trip through their disk cache.

Those two pseudo-behaviors are derived from pose files, which the behaviour
analytics cache never stored, so every refresh re-read every pose file on the UI
thread -- about 7 s of frozen window on a 69-session project, even when the
behaviour rows themselves came straight from cache.  They now have their own
cache, and these tests pin the two properties that make it safe: a faithful
round-trip (NaN latency included, since JSON has no NaN literal) and a
fingerprint that misses whenever an input changes.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab


def _make_host(project_root: Path) -> SimpleNamespace:
    host = SimpleNamespace(
        _project_root=project_root,
        _analytics_cache_dir=lambda root: root / "derived" / "analytics_cache",
    )
    host._pseudo_rows_cache_path = lambda: (
        BehaviorAnalyticsTab._pseudo_rows_cache_path(host)
    )
    return host


ROWS = [
    {"session_id": "s1", "subject": "M01", "behavior_id": "__distance__",
     "behavior": "Distance travelled", "n_bouts": 0.0, "time_spent_s": 0.0,
     "mean_bout_s": 0.0, "latency_s": float("nan"), "distance_cm": 1234.5},
    {"session_id": "s1", "subject": "M01", "behavior_id": "__roi_1__",
     "behavior": "Time in ROI 1", "n_bouts": 4.0, "time_spent_s": 31.25,
     "mean_bout_s": 7.8125, "latency_s": 12.5, "distance_cm": 0.0},
]


def test_pseudo_rows_round_trip(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    BehaviorAnalyticsTab._save_pseudo_rows(host, "fp1", ROWS)

    loaded = BehaviorAnalyticsTab._try_load_pseudo_rows(host, "fp1")

    assert loaded is not None
    assert [r["behavior_id"] for r in loaded] == ["__distance__", "__roi_1__"]
    assert loaded[0]["distance_cm"] == 1234.5
    # A missing latency must come back missing, not as a zero that would be
    # averaged into the real latencies.
    assert math.isnan(loaded[0]["latency_s"])
    assert loaded[1]["latency_s"] == 12.5


def test_pseudo_rows_miss_on_a_different_fingerprint(tmp_path: Path) -> None:
    host = _make_host(tmp_path)
    BehaviorAnalyticsTab._save_pseudo_rows(host, "fp1", ROWS)

    assert BehaviorAnalyticsTab._try_load_pseudo_rows(host, "fp2") is None


def test_pseudo_rows_miss_when_no_cache_written(tmp_path: Path) -> None:
    host = _make_host(tmp_path)

    assert BehaviorAnalyticsTab._try_load_pseudo_rows(host, "fp1") is None


def test_fingerprint_tracks_pose_files_rois_and_prechop(tmp_path: Path) -> None:
    pose = tmp_path / "s1.csv"
    pose.write_text("x,y\n1,1\n", encoding="utf-8")
    roi_path = tmp_path / "config" / "environment_rois.yaml"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text("roi_count: 1\n", encoding="utf-8")

    prechop = {"s1": 0}
    host = SimpleNamespace(
        _project_root=tmp_path,
        _project_fps=lambda: 30.0,
        _pose_path_for_session=lambda sid: pose,
        _analysis_prechop_for_session=lambda sid: prechop[sid],
    )
    fp = lambda: BehaviorAnalyticsTab._pseudo_rows_fingerprint(host, ["s1"])  # noqa: E731

    baseline = fp()
    assert fp() == baseline, "fingerprint must be stable when nothing changes"

    prechop["s1"] = 300
    assert fp() != baseline, "a prechop change must invalidate the rows"

    prechop["s1"] = 0
    assert fp() == baseline, "restoring the prechop must restore the fingerprint"

    roi_path.write_text(
        "roi_count: 2\nproject_rois:\n  target_zones:\n    - {x: 1, y: 1, w: 9, h: 9}\n",
        encoding="utf-8",
    )
    assert fp() != baseline, "an ROI definition change must invalidate the rows"


def test_fingerprint_survives_a_session_without_pose(tmp_path: Path) -> None:
    """An unreachable pose file must hash, not raise."""
    host = SimpleNamespace(
        _project_root=tmp_path,
        _project_fps=lambda: 30.0,
        _pose_path_for_session=lambda sid: None,
        _analysis_prechop_for_session=lambda sid: 0,
    )

    assert BehaviorAnalyticsTab._pseudo_rows_fingerprint(host, ["s1"])
