"""Probability-trace postprocessing into bout intervals."""

from __future__ import annotations

import numpy as np


def smooth_probabilities(
    prob_trace: np.ndarray,
    method: str = "moving_average",
    window: int = 5,
) -> np.ndarray:
    """Apply lightweight smoothing before thresholding."""
    x = np.asarray(prob_trace, dtype=np.float32)
    w = max(1, int(window))
    if w <= 1 or x.size == 0:
        return x

    if method == "moving_average":
        kernel = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(x, kernel, mode="same").astype(np.float32)
    if method == "median":
        half = w // 2
        out = np.zeros_like(x)
        for i in range(len(x)):
            s = max(0, i - half)
            e = min(len(x), i + half + 1)
            out[i] = float(np.median(x[s:e]))
        return out.astype(np.float32)

    raise ValueError("Unsupported smoothing method")


def threshold_probabilities(
    prob_trace: np.ndarray,
    onset_thresh: float,
) -> np.ndarray:
    """Convert probabilities to a binary trace using a single threshold.

    A frame is positive when its (smoothed) probability is at or above
    ``onset_thresh``. This is the only gate in the pipeline, bouts are defined
    by the threshold, min-bout, and merge-gap flags from the Temporal Review tab.
    """
    x = np.asarray(prob_trace, dtype=np.float32)
    return (x >= float(onset_thresh)).astype(np.uint8)


def remove_short_bouts(binary_trace: np.ndarray, min_duration_frames: int) -> np.ndarray:
    """Remove positive runs shorter than ``min_duration_frames``."""
    x = np.asarray(binary_trace, dtype=np.uint8).copy()
    minimum = max(1, int(min_duration_frames))
    i = 0
    while i < len(x):
        if x[i] == 0:
            i += 1
            continue
        j = i
        while j < len(x) and x[j] == 1:
            j += 1
        if (j - i) < minimum:
            x[i:j] = 0
        i = j
    return x


def merge_close_bouts(binary_trace: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill short gaps between positive runs."""
    x = np.asarray(binary_trace, dtype=np.uint8).copy()
    gap = max(0, int(max_gap_frames))
    if gap <= 0 or len(x) == 0:
        return x

    i = 0
    while i < len(x):
        while i < len(x) and x[i] == 0:
            i += 1
        if i >= len(x):
            break
        j = i
        while j < len(x) and x[j] == 1:
            j += 1
        k = j
        while k < len(x) and x[k] == 0:
            k += 1
        if k < len(x) and (k - j) <= gap:
            x[j:k] = 1
            i = k
        else:
            i = k
    return x


def binary_trace_to_intervals(binary_trace: np.ndarray) -> list[tuple[int, int]]:
    """Convert binary trace to inclusive [start, end] bouts."""
    x = np.asarray(binary_trace, dtype=np.uint8)
    intervals: list[tuple[int, int]] = []
    i = 0
    while i < len(x):
        if x[i] == 0:
            i += 1
            continue
        j = i
        while j < len(x) and x[j] == 1:
            j += 1
        intervals.append((int(i), int(j - 1)))
        i = j
    return intervals


# Bout cleanup used when a project has no saved Temporal Review settings.  The
# Temporal Review tab, Analytics and the exports all read these, so a project
# without ``config/temporal_review_settings.json`` gets the same bouts in each.
REVIEW_DEFAULT_ONSET_THRESHOLD = 0.65
REVIEW_DEFAULT_MIN_BOUT_FRAMES = 8
REVIEW_DEFAULT_MERGE_GAP_FRAMES = 4
REVIEW_DEFAULT_SMOOTHING_WINDOW = 5


def review_default_settings() -> dict[str, float | int]:
    """The Temporal Review bout settings a project starts with."""
    return {
        "onset_threshold": REVIEW_DEFAULT_ONSET_THRESHOLD,
        "min_bout_duration_frames": REVIEW_DEFAULT_MIN_BOUT_FRAMES,
        "merge_gap_frames": REVIEW_DEFAULT_MERGE_GAP_FRAMES,
    }


def scaled_cleanup_frames(
    min_bout_duration_frames: int,
    merge_gap_frames: int,
    smoothing_window: int,
    fps_scale: float | None,
) -> tuple[int, int, int]:
    """``(min_bout, merge_gap, smoothing_window)`` in one session's own frames.

    Cleanup settings are stored in project frames.  In a project that mixes
    frame rates, a session recorded at another rate (``fps_scale`` = session
    fps / project fps) gets the same durations in its own frames, so a
    0.4 s minimum bout stays 0.4 s at 60 fps.  ``fps_scale`` of None or 1
    returns the settings unchanged.
    """
    from abel.utils.session_fps import scaled_frames  # noqa: PLC0415

    window = int(smoothing_window)
    if window > 1:  # a window of 1 means smoothing is off and must stay off
        window = scaled_frames(window, fps_scale, minimum=2)
    return (
        scaled_frames(int(min_bout_duration_frames), fps_scale),
        scaled_frames(int(merge_gap_frames), fps_scale, minimum=0),
        window,
    )


def trace_to_intervals(
    prob_trace: np.ndarray,
    *,
    onset_threshold: float,
    min_bout_duration_frames: int,
    merge_gap_frames: int,
    smoothing_method: str = "moving_average",
    smoothing_window: int = REVIEW_DEFAULT_SMOOTHING_WINDOW,
    fps_scale: float | None = None,
) -> list[tuple[int, int]]:
    """Probability trace to inclusive ``(start, end)`` index bouts.

    Smooth, threshold, merge close bouts, drop short bouts, in the order the
    postprocess stage uses.  Every reader that rebuilds bouts from traces
    calls this, so they cannot drift apart.
    """
    min_bout, merge_gap, window = scaled_cleanup_frames(
        min_bout_duration_frames, merge_gap_frames, smoothing_window, fps_scale,
    )
    smoothed = smooth_probabilities(prob_trace, method=smoothing_method, window=window)
    binary = threshold_probabilities(smoothed, onset_thresh=float(onset_threshold))
    binary = merge_close_bouts(binary, max_gap_frames=merge_gap)
    binary = remove_short_bouts(binary, min_duration_frames=min_bout)
    return binary_trace_to_intervals(binary)
