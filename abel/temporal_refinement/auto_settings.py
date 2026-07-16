"""Automatic assessment and suggestion of temporal-refinement settings.

Given held-out per-window probabilities and reviewer labels, this searches a grid
of ``(onset_threshold, min_bout_duration_frames, merge_gap_frames)`` and returns
the combination that maximizes event-level bout F1 — the number the reviewer
actually judges in the Temporal Review tab. It reuses the exact bout-matching
primitives from :mod:`abel.temporal_refinement.refined_eval`, so a suggestion
scored here reproduces the same TP/FP/FN the Validation tab and benchmark report.

The expensive per-session smoothed probability trace is built once (it depends
only on ``smooth_window``, not on the grid), so the whole grid is cheap.

Intended input is the pooled leave-one-subject-out held-out predictions (honest,
subject-generalizing) or a densely-inferred, densely-labeled validation session.
When the labels are sparse fixed-length sampling windows (the active-learning
training set), recall is structurally capped and the result flags
``sparse_labels`` so callers can warn that the onset suggestion is trustworthy
but the recall/min-bout choice should be re-validated on dense inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from abel.temporal_refinement.bout_postprocess import (
    merge_close_bouts,
    remove_short_bouts,
    smooth_probabilities,
    threshold_probabilities,
)
from abel.temporal_refinement.refined_eval import (
    BOUT_MATCH_IOU,
    _extract_bouts,
    _match_bouts,
)

# Default search grid. Onset is fine-grained (it is the dominant knob); min-bout
# and merge-gap are coarse because they matter less and over-fine grids invite
# noise-chasing on small held-out sets. Cost is O(grid x sessions x trace-length):
# ~90 points is instant on a single dense held-out session and still tractable on
# pooled multi-subject LOSO traces. Pass explicit grids to widen or narrow.
DEFAULT_ONSET_GRID = [round(x, 2) for x in np.arange(0.20, 0.901, 0.05)]
DEFAULT_MIN_BOUT_GRID = [3, 8, 15]
DEFAULT_MERGE_GAP_GRID = [2, 8]


def _session_traces(
    y_true: np.ndarray,
    prob: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    smooth_window: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grid-invariant per-session ``(smoothed_prob_trace, raw_true_trace)``.

    Mirrors the trace construction in
    :func:`abel.temporal_refinement.refined_eval._refine_binary_trace` /
    ``_refined_bout_counts`` exactly (segment fill -> gap interpolation ->
    smoothing) so the grid search scores identically to the shipped metric.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for sid in np.unique(session_ids):
        idxs = np.where(session_ids == sid)[0]
        if len(idxs) < 2:
            continue
        sf = start_frames[idxs].astype(int)
        ef = end_frames[idxs].astype(int)
        order = np.argsort(sf)
        sf, ef = sf[order], ef[order]
        clip_probs = prob[idxs][order].astype(float)
        yt = y_true[idxs][order].astype(int)

        trace_start = int(sf[0])
        n_frames = int(ef[-1]) - trace_start + 1
        if n_frames <= 0:
            continue

        trace = np.full(n_frames, np.nan, dtype=np.float32)
        true_trace = np.zeros(n_frames, dtype=np.uint8)
        for i in range(len(sf)):
            ls = int(sf[i]) - trace_start
            le = min(int(ef[i]) - trace_start, n_frames - 1)
            trace[ls : le + 1] = clip_probs[i]
            if yt[i] == 1 and le >= max(0, ls):
                true_trace[max(0, ls) : le + 1] = 1

        nans = np.isnan(trace)
        if nans.all():
            continue
        if nans.any():
            known = np.where(~nans)[0]
            trace = np.interp(np.arange(n_frames), known, trace[known]).astype(np.float32)

        smoothed = smooth_probabilities(trace, method="moving_average", window=smooth_window)
        out.append((smoothed, true_trace))
    return out


def _score_grid_point(
    traces: list[tuple[np.ndarray, np.ndarray]],
    onset: float,
    min_bout: int,
    merge_gap: int,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Event-level TP/FP/FN over all sessions for one settings combination."""
    tp = fp = fn = 0
    for smoothed, true_trace in traces:
        pred = threshold_probabilities(smoothed, onset)
        pred = merge_close_bouts(pred, merge_gap)
        pred = remove_short_bouts(pred, min_bout)
        true_binary = merge_close_bouts(true_trace, merge_gap)
        s_tp, s_fp, s_fn = _match_bouts(
            _extract_bouts(pred), _extract_bouts(true_binary), iou_threshold
        )
        tp += s_tp
        fp += s_fp
        fn += s_fn
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def suggest_temporal_settings(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    onset_grid: list[float] | None = None,
    min_bout_grid: list[int] | None = None,
    merge_gap_grid: list[int] | None = None,
    smooth_window: int = 5,
    iou_threshold: float = BOUT_MATCH_IOU,
    top_k: int = 5,
) -> dict[str, Any]:
    """Suggest bout-F1-maximizing temporal-refinement settings for one behavior.

    Inputs are aligned 1-D arrays over held-out windows: ``prob`` = P(target),
    ``y_true`` = 1 for target windows, and ``session_ids``/``start_frames``/
    ``end_frames`` place each window on a frame axis (session ids should already
    be namespaced per fold so two subjects never share one trace).

    Returns a dict with the recommended settings and their held-out bout metrics::

        {
          "onset_threshold", "min_bout_duration_frames", "merge_gap_frames",
          "precision", "recall", "f1", "tp", "fp", "fn",
          "n_true_bouts", "median_true_bout_frames", "min_true_bout_frames",
          "sparse_labels": bool, "note": str,
          "top_candidates": [ {settings + metrics}, ... ],   # up to top_k
        }

    The winner maximizes bout F1, breaking ties toward higher recall and then
    fewer false positives. ``error`` is set instead when there is no scorable
    data (no sessions with >=2 windows, or no target positives).
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    session_ids = np.asarray(session_ids)
    start_frames = np.asarray(start_frames, dtype=np.int64)
    end_frames = np.asarray(end_frames, dtype=np.int64)

    onset_grid = list(onset_grid) if onset_grid is not None else DEFAULT_ONSET_GRID
    min_bout_grid = list(min_bout_grid) if min_bout_grid is not None else DEFAULT_MIN_BOUT_GRID
    merge_gap_grid = list(merge_gap_grid) if merge_gap_grid is not None else DEFAULT_MERGE_GAP_GRID

    traces = _session_traces(
        y_true, prob, session_ids, start_frames, end_frames, smooth_window
    )
    if not traces:
        return {"error": "no scorable sessions (need >=2 windows per session)"}

    # Ground-truth bout-length profile — used to warn about sparse fixed-window
    # labels, where recall is structurally capped and min-bout must be re-checked
    # on dense inference. (True bouts are merge-gap-independent for this profile.)
    bout_lengths: list[int] = []
    for _, true_trace in traces:
        for a, b in _extract_bouts(true_trace):
            bout_lengths.append(b - a + 1)
    if not bout_lengths:
        return {"error": "no target-positive bouts in held-out data"}
    bl = np.asarray(bout_lengths)
    n_true_bouts = int(bl.size)
    median_bout = int(np.median(bl))
    min_bout_len = int(bl.min())
    # Fixed-length sampling windows show up as a (near-)degenerate length
    # distribution: the median equals the minimum and most bouts share it.
    frac_at_min = float((bl == min_bout_len).mean())
    sparse_labels = bool(median_bout == min_bout_len and frac_at_min >= 0.5)

    candidates: list[dict[str, Any]] = []
    for onset in onset_grid:
        for min_bout in min_bout_grid:
            for merge_gap in merge_gap_grid:
                tp, fp, fn = _score_grid_point(
                    traces, float(onset), int(min_bout), int(merge_gap),
                    iou_threshold,
                )
                prec, rec, f1 = _prf(tp, fp, fn)
                candidates.append({
                    "onset_threshold": float(onset),
                    "min_bout_duration_frames": int(min_bout),
                    "merge_gap_frames": int(merge_gap),
                    "precision": prec, "recall": rec, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn,
                })

    # Rank: F1, then recall, then fewer false positives (prefer higher onset last
    # so ties resolve to the more conservative, cheaper-to-review detector).
    candidates.sort(
        key=lambda c: (round(c["f1"], 4), round(c["recall"], 4), -c["fp"], c["onset_threshold"]),
        reverse=True,
    )
    best = candidates[0]

    note = (
        "Labels look like fixed-length sampling windows (median bout == minimum); "
        "recall is capped by the sparse windows, so trust the onset/precision "
        "suggestion but re-validate min-bout on a densely-inferred session."
        if sparse_labels else
        "Suggestion optimizes held-out bout F1; validate on a fresh session before deploying."
    )

    return {
        **best,
        "n_true_bouts": n_true_bouts,
        "median_true_bout_frames": median_bout,
        "min_true_bout_frames": min_bout_len,
        "sparse_labels": sparse_labels,
        "grid_size": len(candidates),
        "note": note,
        "top_candidates": candidates[: max(1, top_k)],
    }
