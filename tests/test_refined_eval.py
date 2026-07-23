"""Tests for the shared post-refinement evaluation engine.

Covers the refinement math, per-behavior settings resolution, held-out metric
computation (including the label-polarity trap where the target is encoded as
class 0), and the "no held-out probability -> None" contract that makes the
Validation tab show "—" for models trained before the column existed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from abel.temporal_refinement.refined_eval import (
    _extract_bouts,
    _frames_from_segment_ids,
    _match_bouts,
    _refined_bout_counts,
    apply_temporal_refinement,
    load_temporal_settings,
    refined_holdout_metrics,
)


def test_frames_parsed_from_segment_ids() -> None:
    s, e = _frames_from_segment_ids(
        ["seg_MS2_session_a7623464_6_20", "seg_MS1_session_53e6a394_300_314", "garbage"]
    )
    assert list(s) == [6, 300, -1]
    assert list(e) == [20, 314, -1]


def test_apply_refinement_drops_short_bout_and_keeps_long() -> None:
    # 8 contiguous 15-frame windows in one session. One isolated high window
    # (15 frames) must be dropped by a 30-frame min-bout; a 3-window run
    # (45 frames) must survive.
    starts = np.arange(0, 8 * 15, 15)
    ends = starts + 14
    sessions = np.array(["s"] * 8)
    p = np.array([0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1])
    probs = np.column_stack([1.0 - p, p])

    refined = apply_temporal_refinement(
        probs, target_col=1, session_ids=sessions,
        start_frames=starts, end_frames=ends,
        onset_threshold=0.5, min_bout_duration_frames=30, merge_gap_frames=0,
        smooth_window=1,
    )
    # Raw argmax would call window 0 positive; refinement drops it (too short).
    assert np.argmax(probs[0]) == 1
    assert refined[0] == 0
    # The 3-window run (indices 3,4,5) survives.
    assert list(refined[3:6]) == [1, 1, 1]


def test_extract_bouts_finds_contiguous_runs() -> None:
    binary = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert _extract_bouts(binary) == [(1, 2), (5, 5), (7, 9)]
    assert _extract_bouts(np.zeros(5)) == []
    assert _extract_bouts(np.ones(3)) == [(0, 2)]


def test_match_bouts_iou_greedy() -> None:
    # One predicted bout squarely overlaps one true bout -> 1 TP.
    tp, fp, fp_fn = _match_bouts([(0, 9)], [(2, 8)], 0.2)
    assert (tp, fp, fp_fn) == (1, 0, 0)
    # Predicted bout misses the true bout entirely -> 1 FP + 1 FN.
    tp, fp, fn = _match_bouts([(0, 3)], [(50, 60)], 0.2)
    assert (tp, fp, fn) == (0, 1, 1)
    # Two predictions, one true bout: best match wins, the other is an FP.
    tp, fp, fn = _match_bouts([(0, 9), (0, 1)], [(0, 9)], 0.2)
    assert (tp, fp, fn) == (1, 1, 0)


def test_bout_counts_collapse_window_boundary_slop() -> None:
    # One real bout spans windows 2..6; the model fires a hair wide (windows 1..7)
    # plus one isolated stray window far away. Window-level scoring would rack up
    # false positives on the boundary/stray windows, but bout matching sees one
    # correct detection (+1 TP) and one stray bout (+1 FP) — no phantom FN.
    n = 12
    starts = np.arange(0, n * 10, 10)
    ends = starts + 9
    sessions = np.array(["s"] * n)
    y_true = np.zeros(n, dtype=int)
    y_true[2:7] = 1
    prob = np.full(n, 0.05)
    prob[1:8] = 0.9          # slightly wider than the true bout
    prob[10] = 0.9           # isolated stray detection (its own bout)
    settings = {"onset_threshold": 0.5, "min_bout_duration_frames": 1, "merge_gap_frames": 2}
    tp, fp, fn = _refined_bout_counts(
        prob=prob, y_true=y_true, session_ids=sessions,
        start_frames=starts, end_frames=ends, settings=settings, smooth_window=1,
    )
    assert (tp, fn) == (1, 0)
    assert fp == 1


def test_holdout_metrics_expose_window_counts(tmp_path: Path) -> None:
    label_true = [0] * 30 + [1] * 90
    prob = [0.85] * 30 + [0.1] * 90
    mdir = _write_val_model(tmp_path, "Appr", label_true, prob, target_index=0)
    res = refined_holdout_metrics(mdir, tmp_path, "Appr", target_behavior_id="appr-id")
    assert res is not None
    for k in ("raw_tp", "raw_fp", "raw_fn", "raw_tn",
              "refined_tp", "refined_fp", "refined_fn", "refined_tn"):
        assert k in res and isinstance(res[k], int)
    # Event-level bout counts are deliberately NOT reported: they are not
    # identifiable from a held-out labeled subset (see _refined_bout_counts).
    for k in ("bout_tp", "bout_fp", "bout_fn"):
        assert k not in res
    # Every scored window lands in exactly one cell of the confusion matrix.
    assert res["raw_tp"] + res["raw_fp"] + res["raw_fn"] + res["raw_tn"] == res["n_val"]


def test_refinement_does_not_bridge_unobserved_gaps() -> None:
    """A wide unlabeled gap must not be interpolated into a predicted bout.

    Two confidently-positive windows sit ~500 frames apart with nothing observed
    between them — the shape of a held-out labeled subset. Interpolating across
    that gap used to raise the whole span above the onset threshold, inventing a
    detection on frames the model never scored.
    """
    from abel.temporal_refinement.refined_eval import (
        apply_temporal_refinement,
        observed_islands,
    )

    starts = np.array([0, 500])
    ends = np.array([14, 514])
    sessions = np.array(["s", "s"])
    p = np.array([0.9, 0.9])
    probs = np.column_stack([1.0 - p, p])

    assert observed_islands(starts, ends, merge_gap_frames=4) == [(0, 0), (1, 1)]

    preds = apply_temporal_refinement(
        probs, target_col=1, session_ids=sessions,
        start_frames=starts, end_frames=ends,
        onset_threshold=0.5, min_bout_duration_frames=1, merge_gap_frames=4,
        smooth_window=1,
    )
    # Both observed windows stay positive; nothing is asserted about the 485
    # unobserved frames between them because nothing was predicted there.
    assert list(preds) == [1, 1]


def test_observed_islands_is_a_noop_when_windows_tile() -> None:
    """At inference, windows tile densely, so refinement is unchanged.

    Real sessions run ~375% coverage with overlapping windows and zero gaps; the
    splitter must return a single island there or deployment behavior would change.
    """
    from abel.temporal_refinement.refined_eval import observed_islands

    starts = np.arange(0, 200, 4)      # 15-frame windows every 4 frames (overlapping)
    ends = starts + 14
    assert observed_islands(starts, ends, merge_gap_frames=4) == [(0, len(starts) - 1)]
    # Abutting windows with no overlap are also one island.
    starts2 = np.arange(0, 200, 15)
    assert observed_islands(starts2, starts2 + 14, merge_gap_frames=0) == [(0, len(starts2) - 1)]


def test_refined_scores_suppressed_when_windows_cannot_support_min_bout() -> None:
    """min_bout longer than the observed island makes refinement unmeasurable.

    Isolated 15-frame labeled windows cannot hold a 30-frame bout, so every
    positive would be forced to a false negative regardless of model quality.
    That must report as "not evaluable", not as a low score.
    """
    from abel.temporal_refinement.refined_eval import (
        refinement_evaluability,
        score_raw_and_refined,
    )

    starts = np.arange(0, 10 * 500, 500)          # 15-frame windows, 485-frame gaps
    ends = starts + 14
    sessions = np.array(["s"] * len(starts))
    y_true = np.ones(len(starts), dtype=int)
    prob = np.full(len(starts), 0.99)             # a perfect model
    settings = {"onset_threshold": 0.5, "min_bout_duration_frames": 30,
                "merge_gap_frames": 4}

    evaluable, frac = refinement_evaluability(y_true, sessions, starts, ends, settings)
    assert evaluable is False
    assert frac == 1.0

    out = score_raw_and_refined(
        y_true=y_true, prob=prob, session_ids=sessions,
        start_frames=starts, end_frames=ends, settings=settings,
    )
    assert out["refined_evaluable"] is False
    assert math.isnan(out["refined_f1"])
    # The raw score is unaffected and still shows the model is perfect.
    assert out["raw_f1"] == pytest.approx(1.0)

    # Same data, a min_bout the windows CAN support -> evaluable again.
    ok_settings = dict(settings, min_bout_duration_frames=10)
    out_ok = score_raw_and_refined(
        y_true=y_true, prob=prob, session_ids=sessions,
        start_frames=starts, end_frames=ends, settings=ok_settings,
    )
    assert out_ok["refined_evaluable"] is True
    assert not math.isnan(out_ok["refined_f1"])


def test_load_settings_resolves_per_behavior_override(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "temporal_review_settings.json").write_text(
        '{"__all__": {"onset_threshold": 0.65, "min_bout_duration_frames": 8, "merge_gap_frames": 4},'
        ' "by_behavior": {"bid-1": {"onset_threshold": 0.3, "min_bout_duration_frames": 15, "merge_gap_frames": 30}}}'
    )
    (tmp_path / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump({"behaviors": [{"behavior_id": "bid-1", "name": "Rear"}]})
    )
    # Resolve by name and by id.
    for key in ("Rear", "bid-1"):
        s = load_temporal_settings(tmp_path, key)
        assert s["onset_threshold"] == 0.3
        assert s["min_bout_duration_frames"] == 15
    # Unknown behavior falls back to __all__ baseline.
    s = load_temporal_settings(tmp_path, "Unknown")
    assert s["onset_threshold"] == 0.65


def _write_val_model(
    root: Path, name: str, label_true: list[int], prob: list[float], target_index: int
) -> Path:
    mdir = root / "derived" / "models" / f"behavior_model_{name}"
    mdir.mkdir(parents=True)
    n = len(label_true)
    starts = list(range(0, 3 * n, 3))
    ids = [f"seg_MS1_session_A_{starts[i]}_{starts[i] + 14}" for i in range(n)]
    pd.DataFrame(
        {
            "segment_id": ids,
            "session_id": ["session_A"] * n,
            "label_true": label_true,
            "label_pred": [1] * n,
            "prediction_prob": prob,
            "target_index": target_index,
        }
    ).to_parquet(mdir / "validation_predictions.parquet")
    (root / "config").mkdir(exist_ok=True)
    return mdir


def test_missing_prob_column_returns_none(tmp_path: Path) -> None:
    mdir = tmp_path / "derived" / "models" / "behavior_model_Old"
    mdir.mkdir(parents=True)
    pd.DataFrame(
        {"segment_id": ["seg_MS1_session_A_0_14"], "session_id": ["session_A"],
         "label_true": [0], "label_pred": [1]}
    ).to_parquet(mdir / "validation_predictions.parquet")
    assert refined_holdout_metrics(mdir, tmp_path, "Old", target_behavior_id="x") is None


def test_target_encoded_as_zero_is_oriented_correctly(tmp_path: Path) -> None:
    # Positive class (target) is encoded as 0 — the Approach case. Probability is
    # P(target); it is HIGH for true positives (label_true==0). If the engine
    # wrongly assumed positive==1 every metric would collapse.
    label_true = [0] * 30 + [1] * 90            # 30 target, 90 negative
    prob = [0.85] * 30 + [0.1] * 90             # clean separation
    mdir = _write_val_model(tmp_path, "Appr", label_true, prob, target_index=0)

    res = refined_holdout_metrics(mdir, tmp_path, "Appr", target_behavior_id="appr-id")
    assert res is not None
    # With clean separation the raw@0.5 macro F1 should be high (not ~0.5).
    assert res["raw_f1"] > 0.9
    assert res["raw_precision"] > 0.9
    # 30 true positives all scored above 0.5.
    assert res["raw_positive_pred"] == 30
    # Positive-class TP/FP/FN oriented to the target (encoded as class 0).
    assert res["raw_tp"] == 30
    assert res["raw_fp"] == 0
    assert res["raw_fn"] == 0


def test_refined_recovers_positives_when_onset_below_half(tmp_path: Path) -> None:
    # Probabilities for true positives sit at 0.4 — argmax@0.5 misses them all,
    # but a tuned onset threshold of 0.3 recovers them. Demonstrates the value of
    # reporting refined metrics for imbalanced behaviors.
    label_true = [0] * 40 + [1] * 80
    prob = [0.4] * 40 + [0.05] * 80
    mdir = _write_val_model(tmp_path, "Rear", label_true, prob, target_index=0)
    (tmp_path / "config" / "temporal_review_settings.json").write_text(
        '{"__all__": {"onset_threshold": 0.3, "min_bout_duration_frames": 1, "merge_gap_frames": 0}}'
    )
    res = refined_holdout_metrics(mdir, tmp_path, "Rear", target_behavior_id="rear-id")
    assert res is not None
    # Raw@0.5 catches none of the positives; refined@0.3 catches them.
    assert res["raw_positive_pred"] == 0
    assert res["refined_positive_pred"] > 0
    assert res["refined_recall"] > res["raw_recall"]
