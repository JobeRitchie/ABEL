"""Dense sliding-window inference for center-frame temporal models."""

from __future__ import annotations

from typing import Any

import numpy as np

from abel.temporal_refinement.centerframe_model import CenterFrameModel
from abel.temporal_refinement.window_sampler import window_bounds


def _window_from_center(session_features: np.ndarray, center_frame: int, window_frames: int) -> np.ndarray:
    start, end = window_bounds(center_frame=center_frame, window_frames=window_frames)
    n_frames = int(session_features.shape[0])
    s = max(0, int(start))
    e = min(n_frames - 1, int(end))
    window = session_features[s : e + 1]
    pad_left = max(0, -int(start))
    pad_right = max(0, int(end) - (n_frames - 1))
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode="edge")
    if window.shape[0] != window_frames:
        if window.shape[0] < window_frames:
            window = np.pad(window, ((0, window_frames - window.shape[0]), (0, 0)), mode="edge")
        else:
            window = window[:window_frames]
    return np.asarray(window, dtype=np.float32)


def predict_region_dense(
    session_features: np.ndarray,
    start_frame: int,
    end_frame: int,
    model: CenterFrameModel,
    window_frames: int,
    stride: int,
) -> dict[str, Any]:
    """Predict center-frame probabilities over an arbitrary frame region."""
    n_frames = int(session_features.shape[0])
    if n_frames <= 0:
        return {"center_frames": np.asarray([], dtype=int), "probabilities": np.asarray([], dtype=np.float32), "dense_probability": np.asarray([], dtype=np.float32)}

    stride = max(1, int(stride))
    lo = max(0, int(start_frame))
    hi = min(n_frames - 1, int(end_frame))
    centers = np.arange(lo, hi + 1, stride, dtype=int)
    if centers.size == 0:
        return {
            "center_frames": centers,
            "probabilities": np.asarray([], dtype=np.float32),
            "dense_probability": np.full(n_frames, np.nan, dtype=np.float32),
        }

    windows = np.stack([_window_from_center(session_features, int(c), int(window_frames)) for c in centers], axis=0)
    probs = model.predict_proba(windows)

    dense = np.full(n_frames, np.nan, dtype=np.float32)
    dense[centers] = probs
    if stride > 1 and centers.size >= 2:
        known_x = centers.astype(float)
        known_y = probs.astype(float)
        full_x = np.arange(n_frames, dtype=float)
        interp = np.interp(full_x, known_x, known_y, left=known_y[0], right=known_y[-1])
        dense = np.asarray(interp, dtype=np.float32)

    return {
        "center_frames": centers,
        "probabilities": np.asarray(probs, dtype=np.float32),
        "dense_probability": dense,
    }


def predict_session_dense(
    session_features: np.ndarray,
    model: CenterFrameModel,
    window_frames: int,
    stride: int,
) -> dict[str, Any]:
    """Run full-session dense inference (primary mode)."""
    if int(window_frames) != int(model.window_frames):
        raise ValueError(
            "Inference window duration must match training window duration. "
            f"Expected {model.window_frames}, got {window_frames}."
        )
    return predict_region_dense(
        session_features=session_features,
        start_frame=0,
        end_frame=max(0, int(session_features.shape[0]) - 1),
        model=model,
        window_frames=window_frames,
        stride=stride,
    )


def expand_candidate_regions(
    regions: list[tuple[int, int]],
    pad_frames: int,
    n_frames: int | None = None,
) -> list[tuple[int, int]]:
    """Expand regions by +/- pad_frames and merge overlaps."""
    if not regions:
        return []
    pad = max(0, int(pad_frames))
    expanded = sorted((max(0, int(s) - pad), int(e) + pad) for s, e in regions)
    if n_frames is not None and n_frames > 0:
        expanded = [(s, min(n_frames - 1, e)) for s, e in expanded]

    merged: list[tuple[int, int]] = []
    cur_s, cur_e = expanded[0]
    for s, e in expanded[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged
