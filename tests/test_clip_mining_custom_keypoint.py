"""Clip Mining with features from an arbitrary custom keypoint.

Two things a newly added keypoint must get for free once Feature Extraction has
written its columns:

* the mining search bar finds its features, by raw column name or readable label;
* seed examples (``seed_...`` ids with their own frame spans) reach those
  features through the feature window they overlap most, in the same recording
  and for the same animal, and those windows stay out of the background.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.clip_metrics_service import ClipMetricsService, ClipRef

FEATURE = "whisker_pad_jerk_energy"   # no such keypoint ships with ABEL
RICH_ID = f"feat:{FEATURE}"
WIN = 15
N = 300


def _write_features(root, animals=("a1",)):
    """Non-overlapping 15-frame windows; windows 0..29 are high on FEATURE."""
    rng = np.random.default_rng(0)
    frames = []
    for animal in animals:
        df = pd.DataFrame({
            "segment_id": [f"seg_{animal}_{i}" for i in range(N)],
            "session_id": "s1",
            "animal_id": animal,
            "start_frame": np.arange(N) * WIN,
            "end_frame": np.arange(N) * WIN + WIN - 1,
            FEATURE: rng.normal(0.0, 1.0, N),
            "nose_speed_mean": rng.normal(0.0, 1.0, N),
        })
        if animal == animals[0]:
            df.loc[:29, FEATURE] += 8.0
        frames.append(df)
    path = root / "derived" / "representations" / "segment_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path)


def _seeds(n=20, animal=None):
    """Seeds straddling window ``i`` (12 frames) and ``i + 1`` (6 frames)."""
    return [
        ClipRef(f"seed_{i}", "s1", i * WIN + 3, i * WIN + 20, animal)
        for i in range(n)
    ]


def _service(root):
    svc = ClipMetricsService()
    svc.set_project(root)
    return svc


def test_seed_maps_to_the_window_with_the_greatest_overlap(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    clips = _seeds(3) + [ClipRef("seg_a1_7", "s1", 105, 119)]
    assert svc.feature_window_ids(clips) == {
        "seed_0": "seg_a1_0", "seed_1": "seg_a1_1", "seed_2": "seg_a1_2",
        "seg_a1_7": "seg_a1_7",
    }


def test_seed_outside_any_window_or_recording_is_left_unmatched(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    clips = [ClipRef("seed_far", "s1", N * WIN + 50, N * WIN + 80),
             ClipRef("seed_other", "s2", 3, 20)]
    assert svc.feature_window_ids(clips) == {}


def test_seed_matches_its_own_animal_in_a_multi_animal_recording(tmp_path):
    _write_features(tmp_path, animals=("a1", "a2"))
    svc = _service(tmp_path)
    got = svc.feature_window_ids(_seeds(2, animal="a2"))
    assert got == {"seed_0": "seg_a2_0", "seed_1": "seg_a2_1"}


def test_seed_essence_reaches_the_custom_keypoint_feature(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    seeds = _seeds(20)
    ex_ids = [c.window_id for c in seeds]
    bg_ids = [f"seg_a1_{i}" for i in range(40, N)]

    # The old exact-id lookup: no seed is in the feature table.
    before = svc.essence_frames(ex_ids, bg_ids)
    assert "extracted features" not in before.sources

    fmap = svc.feature_window_ids(seeds)
    frames = svc.essence_frames(ex_ids, bg_ids, feature_ids=fmap)
    assert frames.sources == ["extracted features"]
    assert list(frames.exemplars.index) == ex_ids
    assert not frames.notes
    crits = svc.extract_similar_essence(frames.exemplars, frames.background, k=3)
    assert RICH_ID in [c.metric_id for c in crits]


def test_seeds_without_features_are_reported(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    seeds = _seeds(6) + [ClipRef("seed_far", "s1", N * WIN + 50, N * WIN + 80)]
    frames = svc.essence_frames(
        [c.window_id for c in seeds], [f"seg_a1_{i}" for i in range(40, N)],
        feature_ids=svc.feature_window_ids(seeds),
    )
    assert any(n.startswith("1 of 7 selected clip(s)") for n in frames.notes)


# ── dialog: search bar + seed-based Extract essence ────────────────────────

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    try:
        return QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")


def _dialog(root, exemplars):
    from abel.ui.clip_mining_dialog import ClipMiningDialog

    return ClipMiningDialog(
        project_root=root,
        exemplar_provider=lambda: exemplars,
        scope_label="test",
        on_apply=lambda refs, scores: None,
    )


@pytest.mark.parametrize("query", [
    FEATURE, FEATURE.upper(), "whisker pad jerk", "Whisker pad jerk (energy)",
])
def test_search_finds_the_custom_keypoint_feature(_app, tmp_path, query):
    _write_features(tmp_path)
    dlg = _dialog(tmp_path, [])
    assert dlg._resolve_feature(query) == RICH_ID
    assert any(FEATURE in shown for shown in dlg._feature_display)
    n = len(dlg._rows)
    dlg._add_searched_feature(query)
    assert len(dlg._rows) == n + 1
    assert dlg._rows[-1].to_criterion().metric_id == RICH_ID


def test_seed_based_essence_in_the_dialog(_app, tmp_path):
    _write_features(tmp_path)
    seeds = _seeds(20)
    dlg = _dialog(tmp_path, seeds)
    out = dlg._essence_job(seeds, 3, 0.9)
    frames = out["frames"]
    assert "extracted features" in frames.sources
    assert RICH_ID in [c.metric_id for c in out["crits"]]
    matched = {f"seg_a1_{i}" for i in range(20)}
    assert not matched & set(frames.background.index)
    assert not set(frames.exemplars.index) & set(frames.background.index)
