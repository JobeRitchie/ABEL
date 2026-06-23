"""Frame and bout-level metrics for temporal refinement."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score


def frame_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | list[float]]:
    """Compute frame-level precision/recall/F1/PR metrics."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    ys = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(yt, ys)
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "aucpr": float(average_precision_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan"),
        "pr_curve_precision": precision.astype(float).tolist(),
        "pr_curve_recall": recall.astype(float).tolist(),
    }


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return float(inter) / float(max(1, union))


def bout_level_metrics(
    predicted_intervals: list[tuple[int, int]],
    reviewed_intervals: list[tuple[int, int]],
) -> dict[str, float]:
    """Compute overlap and boundary timing errors between predicted and reviewed bouts."""
    if not predicted_intervals and not reviewed_intervals:
        return {
            "mean_iou": 1.0,
            "onset_error_frames": 0.0,
            "offset_error_frames": 0.0,
            "bout_count_difference": 0.0,
            "mean_bout_duration_error_frames": 0.0,
        }

    ious: list[float] = []
    onset_err: list[float] = []
    offset_err: list[float] = []
    dur_err: list[float] = []

    for pred in predicted_intervals:
        if not reviewed_intervals:
            ious.append(0.0)
            continue
        overlaps = [_interval_iou(pred, gt) for gt in reviewed_intervals]
        best_idx = int(np.argmax(overlaps))
        best_gt = reviewed_intervals[best_idx]
        ious.append(float(overlaps[best_idx]))
        onset_err.append(float(abs(pred[0] - best_gt[0])))
        offset_err.append(float(abs(pred[1] - best_gt[1])))
        dur_err.append(float(abs((pred[1] - pred[0]) - (best_gt[1] - best_gt[0]))))

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "onset_error_frames": float(np.mean(onset_err)) if onset_err else float("nan"),
        "offset_error_frames": float(np.mean(offset_err)) if offset_err else float("nan"),
        "bout_count_difference": float(len(predicted_intervals) - len(reviewed_intervals)),
        "mean_bout_duration_error_frames": float(np.mean(dur_err)) if dur_err else float("nan"),
    }


def probability_histogram(prob_trace: np.ndarray, bins: int = 20) -> dict[str, list[float]]:
    """Simple probability histogram for calibration inspection."""
    vals = np.asarray(prob_trace, dtype=float)
    hist, edges = np.histogram(vals, bins=max(2, int(bins)), range=(0.0, 1.0), density=False)
    return {
        "counts": hist.astype(float).tolist(),
        "bin_edges": edges.astype(float).tolist(),
    }
