"""Tests for automatic temporal-refinement settings suggestion.

The load-bearing assertion is *consistency*: the fast grid scorer must reproduce
the shipped event-level bout counts (``refined_eval._refined_bout_counts``) for
any settings, so a suggestion means the same thing the Validation tab reports.
"""

from __future__ import annotations

import numpy as np

from abel.temporal_refinement import auto_settings as A
from abel.temporal_refinement.refined_eval import _refined_bout_counts


def _synth(n_sessions: int = 3, frames_per: int = 300, seed: int = 0):
    """Dense per-window arrays: contiguous 15-frame windows tiling each session.

    Target windows get high probability, negatives low; a handful of negatives
    get borderline probability so onset actually matters.
    """
    rng = np.random.default_rng(seed)
    y, p, sess, sf, ef = [], [], [], [], []
    win = 15
    for s in range(n_sessions):
        n_win = frames_per // win
        # Target present in two contiguous stretches per session.
        pos = set(range(3, 7)) | set(range(12, 15))
        for k in range(n_win):
            is_pos = k in pos
            start = k * win
            y.append(1 if is_pos else 0)
            if is_pos:
                p.append(rng.uniform(0.75, 0.95))
            else:
                # A few borderline negatives near 0.5 to exercise the onset knob.
                p.append(rng.uniform(0.45, 0.6) if k % 5 == 0 else rng.uniform(0.02, 0.3))
            sess.append(f"sub{s}::sess{s}")
            sf.append(start)
            ef.append(start + win - 1)
    return (
        np.array(y, int), np.array(p, float), np.array(sess, object),
        np.array(sf, np.int64), np.array(ef, np.int64),
    )


def test_scorer_matches_refined_bout_counts():
    y, p, sess, sf, ef = _synth()
    traces = A._session_traces(y, p, sess, sf, ef, smooth_window=5)
    for onset in (0.3, 0.5, 0.7, 0.9):
        for min_bout in (3, 8, 15):
            for gap in (2, 8):
                fast = A._score_grid_point(traces, onset, min_bout, gap, 0.2)
                settings = {
                    "onset_threshold": onset,
                    "min_bout_duration_frames": min_bout, "merge_gap_frames": gap,
                }
                ref = _refined_bout_counts(p, y, sess, sf, ef, settings)
                assert fast == ref, f"mismatch at onset={onset} mb={min_bout} gap={gap}: {fast} != {ref}"


def test_suggests_reasonable_onset():
    y, p, sess, sf, ef = _synth()
    res = A.suggest_temporal_settings(
        y_true=y, prob=p, session_ids=sess, start_frames=sf, end_frames=ef
    )
    assert "error" not in res
    # Positives sit at 0.75-0.95, borderline negatives up to 0.6 -> a good onset
    # lives above the negatives; F1 should be strong on this separable synthetic.
    assert res["onset_threshold"] >= 0.6
    assert res["f1"] >= 0.8
    assert res["tp"] > 0
    # Winner must be present in the returned candidate ranking.
    assert res["top_candidates"][0]["f1"] == res["f1"]


def test_flags_sparse_fixed_window_labels():
    # Isolated single 15-frame positive windows (no contiguity) -> sparse flag.
    y, p, sess, sf, ef = [], [], [], [], []
    for s in range(3):
        for k in range(20):
            is_pos = k in (2, 8, 14)  # isolated, non-adjacent
            y.append(1 if is_pos else 0)
            p.append(0.9 if is_pos else 0.05)
            sess.append(f"sub{s}::sess{s}")
            sf.append(k * 15)
            ef.append(k * 15 + 14)
    res = A.suggest_temporal_settings(
        y_true=np.array(y, int), prob=np.array(p, float),
        session_ids=np.array(sess, object),
        start_frames=np.array(sf, np.int64), end_frames=np.array(ef, np.int64),
    )
    assert res["sparse_labels"] is True
    assert res["median_true_bout_frames"] == 15
    assert "dense" in res["note"].lower()


def test_no_scorable_data_returns_error():
    res = A.suggest_temporal_settings(
        y_true=np.array([1]), prob=np.array([0.9]),
        session_ids=np.array(["a::a"], object),
        start_frames=np.array([0], np.int64), end_frames=np.array([14], np.int64),
    )
    assert "error" in res
