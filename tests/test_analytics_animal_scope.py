"""Analytics can be scoped to one animal of a multi-animal session.

A multi-animal session holds two subjects, and its session trace is the
per-frame maximum across them, so without a scope every chart silently
describes "either animal".  These tests pin how a selected animal picks its
own trace, its own pose track and its own cache.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab


def _host(scope: str) -> SimpleNamespace:
    return SimpleNamespace(_animal_scope=scope, _safe_name=BehaviorAnalyticsTab._safe_name)


TRACES = {"multi": "multi_max.parquet", "single": "single.parquet", "other": "other_max.parquet"}
ANIMAL_TRACES = {
    "multi": {"track_0": "multi_t0.parquet", "track_1": "multi_t1.parquet"},
    # A session where track_1 was never tracked.
    "other": {"track_0": "other_t0.parquet"},
}


def test_selected_animal_uses_its_own_trace() -> None:
    out = BehaviorAnalyticsTab._scope_trace_paths_to_animal(
        _host("track_1"), TRACES, ANIMAL_TRACES,
    )
    assert out["multi"] == "multi_t1.parquet"
    # Single-animal sessions keep their only trace.
    assert out["single"] == "single.parquet"
    # The max trace must not stand in for an animal that is not there.
    assert "other" not in out


def test_display_name_numbers_tracks_from_one() -> None:
    assert BehaviorAnalyticsTab._animal_display_name("track_0") == "Mouse 1 (track_0)"
    assert BehaviorAnalyticsTab._animal_display_name("track_1") == "Mouse 2 (track_1)"
    assert BehaviorAnalyticsTab._animal_display_name("Mouse1") == "Mouse1"


def test_cache_dir_is_separate_per_animal(tmp_path: Path) -> None:
    base = BehaviorAnalyticsTab._analytics_cache_dir(_host(""), tmp_path)
    t0 = BehaviorAnalyticsTab._analytics_cache_dir(_host("track_0"), tmp_path)
    t1 = BehaviorAnalyticsTab._analytics_cache_dir(_host("track_1"), tmp_path)
    assert len({base, t0, t1}) == 3


def _pose_host(scope: str, individuals: list[str]) -> SimpleNamespace:
    multi = SimpleNamespace(
        individuals=individuals,
        per_individual={i: f"pose:{i}" for i in individuals},
    )
    pose = SimpleNamespace(load_multi=lambda p: multi, load=lambda p: "pose:first")
    return SimpleNamespace(_animal_scope=scope, _pose=pose)


def test_pose_load_picks_the_selected_track() -> None:
    host = _pose_host("track_1", ["track_0", "track_1"])
    assert BehaviorAnalyticsTab._load_pose_file(host, Path("x.h5")) == "pose:track_1"


def test_pose_load_unscoped_keeps_the_default_loader() -> None:
    host = _pose_host("", ["track_0", "track_1"])
    assert BehaviorAnalyticsTab._load_pose_file(host, Path("x.h5")) == "pose:first"


def test_pose_load_never_substitutes_the_other_animal() -> None:
    host = _pose_host("track_1", ["track_0", "track_2"])
    with pytest.raises(ValueError):
        BehaviorAnalyticsTab._load_pose_file(host, Path("x.h5"))
    # A single-animal file has only one track to use.
    host = _pose_host("track_1", ["individual0"])
    assert BehaviorAnalyticsTab._load_pose_file(host, Path("x.h5")) == "pose:individual0"
