"""Essence extraction over the shipped per-window feature table.

The Essence Miner used to see only the ~22 interpretable pose/ROI clip metrics,
which have no oscillation, periodicity or angular-velocity terms — exactly where
behaviours like a wet-dog-shake separate.  It now *also* ranges over
``derived/representations/segment_features.parquet``, the same features the
classifier is trained on.  These tests guard the three things that has to get
right: the two metric spaces coexist without colliding, the feature space is
capped so the greedy search stays interactive, and every missing-input path
degrades to a clear, weaker-but-working result instead of a crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.clip_metrics_service import (
    ESSENCE_MAX_FEATURES,
    METRIC_BY_ID,
    ClipMetricsService,
    is_rich_metric,
    metric_def_for,
    metric_label,
    rich_column,
    rich_metric_def,
    rich_metric_id,
)

N_ROWS = 400


def _write_features(root, n=N_ROWS, seed=0):
    """A miniature ``segment_features.parquet``: signal, noise, degenerate, meta.

    Rows ``seg_0…seg_29`` are the "positives" — the only ones high on
    ``ear_right_acceleration_median``, a column the clip-metric space has no
    equivalent of.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "segment_id": [f"seg_{i}" for i in range(n)],
        "session_id": ["s1"] * n,
        "animal_id": ["a1"] * n,
        "start_frame": np.arange(n),
        "end_frame": np.arange(n) + 30,
        "ear_right_acceleration_median": rng.normal(0.0, 1.0, n),
        "oscillation_power_nose_x": rng.normal(0.0, 1.0, n),
        "body_orientation_std": rng.normal(0.0, 1.0, n),
        "flat_feature": np.zeros(n),                       # constant → unusable
        "sparse_feature": np.where(np.arange(n) < 20, rng.normal(0, 1, n), np.nan),
        "prediction_prob": rng.random(n),                  # model output → excluded
        "uncertainty_score": rng.random(n),                # model output → excluded
    })
    df.loc[:29, "ear_right_acceleration_median"] += 6.0
    path = root / "derived" / "representations" / "segment_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _service(root):
    svc = ClipMetricsService()
    svc.set_project(root)
    return svc


def _clip_metrics(ids, seed=1):
    """Interpretable clip metrics that carry no signal about the positives."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"centroid_speed_mean": rng.normal(50, 5, len(ids)),
         "body_length_mean": rng.normal(60, 4, len(ids)),
         "duration_sec": np.full(len(ids), 0.5)},
        index=list(ids),
    )


# ── naming / namespacing ───────────────────────────────────────────────────


def test_rich_ids_are_namespaced_and_never_collide():
    # ``nose_speed_mean`` exists in BOTH spaces in different units; the namespace
    # is what keeps a criterion on one from being evaluated against the other.
    rid = rich_metric_id("nose_speed_mean")
    assert rid != "nose_speed_mean" and is_rich_metric(rid)
    assert not is_rich_metric("nose_speed_mean")
    assert rich_column(rid) == "nose_speed_mean"
    assert rid not in METRIC_BY_ID  # the interpretable registry is untouched


@pytest.mark.parametrize("col,label", [
    ("ear_right_acceleration_median", "Ear right acceleration (median)"),
    ("body_orientation_std", "Body orientation (variability)"),
    ("oscillation_power_nose_x", "Oscillation power nose x"),
])
def test_rich_labels_are_humanised(col, label):
    d = rich_metric_def(rich_metric_id(col))
    assert d.label == label
    assert d.group.startswith("Feature")
    assert col in d.description  # the raw column stays traceable
    assert metric_label(rich_metric_id(col)) == label


def test_metric_def_for_resolves_both_spaces():
    assert metric_def_for("centroid_speed_mean").label == "Centroid speed (mean)"
    assert metric_def_for(rich_metric_id("nose_jerk_mean")) is not None
    assert metric_def_for("not_a_metric") is None


# ── loading ────────────────────────────────────────────────────────────────


def test_rich_feature_ids_exclude_meta_and_model_outputs(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ids = set(svc.rich_feature_ids())
    assert svc.has_rich_features()
    assert rich_metric_id("ear_right_acceleration_median") in ids
    for excluded in ("segment_id", "session_id", "start_frame",
                     "prediction_prob", "uncertainty_score"):
        assert rich_metric_id(excluded) not in ids


def test_missing_feature_table_is_not_an_error(tmp_path):
    svc = _service(tmp_path)
    assert svc.rich_feature_ids() == []
    assert not svc.has_rich_features()
    assert svc.load_rich_features().empty
    assert svc.rich_essence_frame(["seg_0"]).empty


def test_load_rich_features_filters_rows_and_columns(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    want = rich_metric_id("oscillation_power_nose_x")
    df = svc.load_rich_features(metric_ids=[want, "centroid_speed_mean"],
                                segment_ids={"seg_1", "seg_2"})
    assert list(df.columns) == [want]          # non-rich ids ignored
    assert sorted(df.index) == ["seg_1", "seg_2"]


def test_rich_essence_frame_drops_constant_and_sparse_columns(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    frame = svc.rich_essence_frame([f"seg_{i}" for i in range(N_ROWS)])
    assert rich_metric_id("flat_feature") not in frame.columns
    assert rich_metric_id("sparse_feature") not in frame.columns
    assert rich_metric_id("ear_right_acceleration_median") in frame.columns


def test_attach_rich_columns_joins_only_what_is_missing(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ids = [f"seg_{i}" for i in range(10)]
    df = _clip_metrics(ids)
    want = rich_metric_id("body_orientation_std")
    out = svc.attach_rich_columns(df, [want, "centroid_speed_mean"])
    assert want in out.columns and len(out) == len(df)
    assert list(out.index) == ids
    # Idempotent: a second call adds nothing and keeps the same frame.
    assert svc.attach_rich_columns(out, [want]).shape == out.shape


# ── frame assembly / graceful degradation ──────────────────────────────────


def test_essence_frames_union_both_spaces(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ex_ids = [f"seg_{i}" for i in range(8)]
    bg_ids = [f"seg_{i}" for i in range(30, 230)]
    frames = svc.essence_frames(ex_ids, bg_ids,
                                _clip_metrics(ex_ids), _clip_metrics(bg_ids))
    assert frames.usable()
    assert set(frames.sources) == {"pose metrics", "extracted features"}
    assert "centroid_speed_mean" in frames.exemplars.columns
    assert rich_metric_id("ear_right_acceleration_median") in frames.exemplars.columns
    assert len(frames.exemplars) == len(ex_ids)
    assert not frames.exemplars.columns.duplicated().any()


def test_essence_frames_without_pose_metrics(tmp_path):
    """Unreadable pose (all-NaN metrics) must leave the feature half working."""
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ex_ids = [f"seg_{i}" for i in range(8)]
    bg_ids = [f"seg_{i}" for i in range(30, 230)]
    dead = _clip_metrics(ex_ids) * np.nan
    frames = svc.essence_frames(ex_ids, bg_ids, dead, _clip_metrics(bg_ids) * np.nan)
    assert frames.usable()
    assert frames.sources == ["extracted features"]
    assert all(is_rich_metric(c) for c in frames.exemplars.columns)


def test_essence_frames_without_feature_table_falls_back_with_a_note(tmp_path):
    svc = _service(tmp_path)  # no parquet written
    ex_ids = [f"seg_{i}" for i in range(8)]
    bg_ids = [f"seg_{i}" for i in range(30, 230)]
    frames = svc.essence_frames(ex_ids, bg_ids,
                                _clip_metrics(ex_ids), _clip_metrics(bg_ids))
    assert frames.usable()
    assert frames.sources == ["pose metrics"]
    assert frames.notes and "Feature Extraction" in frames.notes[0]


def test_essence_frames_with_nothing_usable(tmp_path):
    svc = _service(tmp_path)
    frames = svc.essence_frames(["seg_0"], ["seg_1"], None, None)
    assert not frames.usable()
    assert frames.notes  # the caller always has something to tell the user


def test_essence_frames_note_when_selection_is_not_in_the_feature_table(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ex_ids = ["unknown_a", "unknown_b"]
    bg_ids = [f"seg_{i}" for i in range(30, 230)]
    frames = svc.essence_frames(ex_ids, bg_ids,
                                _clip_metrics(ex_ids), _clip_metrics(bg_ids))
    assert frames.sources == ["pose metrics"]
    assert frames.notes and "Feature Extraction" in frames.notes[0]


# ── search cost + end-to-end discrimination ────────────────────────────────


def test_feature_search_is_capped():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(300)]
    ex = pd.DataFrame(rng.normal(4, 1, (30, len(cols))), columns=cols)
    bg = pd.DataFrame(rng.normal(0, 1, (500, len(cols))), columns=cols)
    feats = ClipMetricsService._usable_essence_metrics(ex, bg)
    assert len(feats) == ESSENCE_MAX_FEATURES
    assert len(ClipMetricsService._usable_essence_metrics(
        ex, bg, max_features=5)) == 5


def test_essence_finds_a_signal_only_the_rich_features_carry(tmp_path):
    """The end-to-end win: the discriminating column exists ONLY in the feature
    table, so the clip-metric-only essence cannot express it and the unioned one
    must."""
    _write_features(tmp_path)
    svc = _service(tmp_path)
    ex_ids = [f"seg_{i}" for i in range(8)]                 # positives
    bg_ids = [f"seg_{i}" for i in range(30, 330)]           # background
    frames = svc.essence_frames(ex_ids, bg_ids,
                                _clip_metrics(ex_ids), _clip_metrics(bg_ids))
    crits = ClipMetricsService.extract_similar_essence(
        frames.exemplars, frames.background, k=5, recall_target=0.8)
    assert crits
    assert rich_metric_id("ear_right_acceleration_median") in {c.metric_id for c in crits}

    # And the resulting box, evaluated over the whole pool, is dominated by the
    # planted positives — with the feature columns joined on by the service.
    all_ids = [f"seg_{i}" for i in range(N_ROWS)]
    pool = svc.attach_rich_columns(_clip_metrics(all_ids),
                                   [c.metric_id for c in crits])
    matched = ClipMetricsService.mine(pool, crits, match_all=True).matched_ids
    assert matched
    assert sum(1 for w in matched if int(w.split("_")[1]) < 30) >= 0.8 * len(matched)


def test_rich_criteria_survive_a_save_load_round_trip(tmp_path):
    _write_features(tmp_path)
    svc = _service(tmp_path)
    from abel.services.clip_metrics_service import Criterion

    keep = rich_metric_id("body_orientation_std")
    svc.save_criteria(
        [Criterion(keep, 1.0, 2.0),
         Criterion("centroid_speed_mean", None, 40.0),
         Criterion(rich_metric_id("not_a_real_column"), 0.0, 1.0)],
        match_all=True,
    )
    crits, match_all = svc.load_criteria()
    assert match_all is True
    assert [c.metric_id for c in crits] == [keep, "centroid_speed_mean"]
    assert (crits[0].low, crits[0].high) == (1.0, 2.0)
