"""GPU-accelerated windowed feature aggregation using PyTorch CUDA.

Falls back to vectorised NumPy when CUDA is unavailable, which is still
significantly faster than the per-window Python loop it replaces.

Both paths eliminate the inner Python loop over sliding windows by computing
all statistics (mean, std, median, max, p10, p90, energy, periodicity) for
every window simultaneously.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Fraction of each window averaged at the start and end when computing the
# clip-wise ``_delta`` (net directional change).  Averaging an edge band makes
# the delta robust to single-frame tracking jitter at the window boundaries
# instead of trusting one endpoint frame each.
_DELTA_EDGE_FRACTION = 0.25

_TORCH_CUDA_CHECKED = False


def _is_roi_spatial_col(col: str) -> bool:
    """Return True for columns that capture distance or angle relative to an ROI or target zone.

    Matches:
      * ``*_to_target_dist``  / ``*_to_roi_N_dist``   — Euclidean distance columns
      * ``*_angle_to_target`` / ``*_angle_to_roi_N``  — heading-angle columns

    These are the columns for which start-to-end delta and linear trend
    statistics are meaningful (e.g. approach vs retreat behaviour).
    """
    return (
        col.endswith("_to_target_dist")
        or col.endswith("_angle_to_target")
        or ("_to_roi_" in col and col.endswith("_dist"))
        or "_angle_to_roi_" in col
    )


def _is_posture_delta_col(col: str) -> bool:
    """Return True for per-frame posture columns whose start-to-end change is meaningful.

    Covers two families produced by the robustness feature extractors:

      * Angle columns — ``joint_angle_*`` (spine/limb flexion etc.),
        ``head_direction_angle``, ``head_angular_velocity`` is excluded (already a
        rate), ``body_orientation``, ``head_pitch`` and ``spine_curvature*``.
      * Proximity columns — pairwise inter-keypoint distances ``dist_<a>_to_<b>``
        and their body-length-normalized variants ``dist_<a>_to_<b>_norm``.

    These are the columns for which a clip-wise ``_delta`` / ``_trend`` captures
    how posture evolves across the window — signal that mean/std aggregates
    discard.  ROI/target columns are handled separately by
    :func:`_is_roi_spatial_col` and are excluded here to avoid duplicate columns.
    """
    if _is_roi_spatial_col(col):
        return False
    # Proximity: inter-keypoint pairwise distances (and normalized variants).
    if col.startswith("dist_") and "_to_" in col:
        return True
    # Social inter-animal proximity / orientation columns.  A start→end delta on
    # ``social_dist_*`` captures net approach or retreat across the clip, and on
    # facing/heading-alignment columns captures turning toward or away.  Rate-like
    # (approach_velocity), overlap, and contact columns are excluded.
    if col.startswith("social_"):
        return "_dist_" in col or "facing" in col or "heading_alignment" in col
    # Angle-like posture descriptors.
    if col.startswith("joint_angle_"):
        return True
    if col.startswith("spine_curvature"):
        return True
    return col in ("head_direction_angle", "body_orientation", "head_pitch")


_TORCH_CUDA_OK = False
_GPU_STATS_DISABLED = False

# The GPU windowed-stats path (torch.quantile / fft / unfold on CUDA) is a pure
# optimization over the vectorised NumPy path, which the CPU code below already
# completes in a few seconds.  On Windows it has proven fragile: torch.quantile
# can hang on large sorts, and a driver/kernel fault surfaces as an *uncatchable*
# native access violation that fast-fails the whole process (exit 0xC0000409),
# taking long-running pipelines down with no Python traceback.  So it is OFF by
# default; set ABEL_GPU_FEATURE_STATS=1 to opt back in.  (This is independent of
# GPU optical flow, which is controlled elsewhere.)
_GPU_STATS_ENABLED = os.environ.get(
    "ABEL_GPU_FEATURE_STATS", "0"
).strip().lower() in ("1", "true", "yes", "on")


def gpu_available() -> bool:
    """Return *True* if PyTorch with CUDA support is importable and a GPU is present."""
    global _TORCH_CUDA_CHECKED, _TORCH_CUDA_OK
    if _TORCH_CUDA_CHECKED:
        return _TORCH_CUDA_OK
    _TORCH_CUDA_CHECKED = True
    try:
        import torch

        _TORCH_CUDA_OK = torch.cuda.is_available()
        if _TORCH_CUDA_OK:
            logger.info(
                "GPU feature acceleration available: %s",
                torch.cuda.get_device_name(0),
            )
        else:
            logger.info(
                "PyTorch found but CUDA not available — using vectorised CPU path."
            )
    except ImportError:
        logger.info("PyTorch not installed — using vectorised CPU path.")
    return _TORCH_CUDA_OK


# ---------------------------------------------------------------------------
# GPU path (PyTorch CUDA)
# ---------------------------------------------------------------------------

def _windowed_stats_gpu(
    data: np.ndarray,
    window_size: int,
    stride: int,
    include_periodicity: bool,
) -> dict[str, np.ndarray]:
    import torch

    device = torch.device("cuda")

    # Probe CUDA health with a tiny allocation first — catches driver-level
    # errors that would otherwise surface as uncatchable access violations
    # when transferring the real (larger) payload.
    _probe = torch.zeros(1, device=device)
    del _probe

    # Transfer via CPU tensor → .to(device) so that OOM or driver faults
    # raise a catchable RuntimeError instead of a process-killing segfault.
    t_cpu = torch.as_tensor(
        np.ascontiguousarray(data), dtype=torch.float32
    )
    t = t_cpu.to(device, non_blocking=False)
    del t_cpu
    torch.cuda.synchronize()

    # unfold along the frame axis → (n_windows, n_features, window_size)
    windows = t.unfold(0, window_size, stride)

    out: dict[str, np.ndarray] = {}
    out["mean"] = windows.mean(dim=2).cpu().numpy()
    # population std (ddof=0) to match np.std default
    out["std"] = windows.std(dim=2, correction=0).cpu().numpy()
    # quantile requires contiguous input; .contiguous() makes the copy
    # explicit and avoids internal re-allocation surprises.
    windows_c = windows.contiguous()
    del windows, t
    out["median"] = torch.quantile(windows_c, 0.5, dim=2).cpu().numpy()
    out["max"] = windows_c.max(dim=2).values.cpu().numpy()
    out["p10"] = torch.quantile(windows_c, 0.1, dim=2).cpu().numpy()
    out["p90"] = torch.quantile(windows_c, 0.9, dim=2).cpu().numpy()
    out["energy"] = (windows_c * windows_c).mean(dim=2).cpu().numpy()

    if include_periodicity:
        centered = windows_c - windows_c.mean(dim=2, keepdim=True)
        variance = centered.var(dim=2, correction=0)
        if window_size >= 8:
            fft = torch.fft.rfft(centered, dim=2)
            mag = torch.abs(fft[:, :, 1:])  # skip DC component
            peak = (
                mag.max(dim=2).values
                if mag.shape[2] > 0
                else torch.zeros_like(variance)
            )
            peak = torch.where(variance > 1e-10, peak, torch.zeros_like(peak))
        else:
            peak = torch.zeros(
                windows_c.shape[0], windows_c.shape[1], device=device
            )
        out["periodicity"] = peak.cpu().numpy()

    del windows_c
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# CPU path (vectorised NumPy — no Python window loop)
# ---------------------------------------------------------------------------

def _windowed_stats_cpu(
    data: np.ndarray,
    window_size: int,
    stride: int,
    include_periodicity: bool,
) -> dict[str, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view

    # (n_valid, n_features, window_size)
    all_windows = sliding_window_view(data, window_size, axis=0)
    windows = all_windows[::stride]  # apply stride → (n_windows, n_features, window_size)

    out: dict[str, np.ndarray] = {}
    out["mean"] = windows.mean(axis=2)
    out["std"] = windows.std(axis=2)  # ddof=0 by default
    out["median"] = np.median(windows, axis=2)
    out["max"] = windows.max(axis=2)
    out["p10"] = np.percentile(windows, 10, axis=2)
    out["p90"] = np.percentile(windows, 90, axis=2)
    out["energy"] = (windows * windows).mean(axis=2)

    if include_periodicity:
        centered = windows - windows.mean(axis=2, keepdims=True)
        variance = centered.var(axis=2)
        if window_size >= 8:
            fft_vals = np.fft.rfft(centered, axis=2)
            mag = np.abs(fft_vals[:, :, 1:])
            peak = (
                mag.max(axis=2) if mag.shape[2] > 0 else np.zeros_like(variance)
            )
            peak = np.where(variance > 1e-10, peak, 0.0)
        else:
            peak = np.zeros((windows.shape[0], windows.shape[1]))
        out["periodicity"] = peak

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def windowed_feature_summary(
    data: np.ndarray,
    window_size: int,
    stride: int,
    include_periodicity: bool = True,
) -> dict[str, np.ndarray]:
    """Compute windowed aggregation statistics, preferring GPU when available.

    Parameters
    ----------
    data : ndarray of shape (n_frames, n_features)
        Contiguous feature matrix for a single group (animal / session).
    window_size : int
        Number of frames per window.
    stride : int
        Step between consecutive window start positions.
    include_periodicity : bool
        Whether to compute FFT-based periodicity per feature.

    Returns
    -------
    dict mapping stat name → ndarray of shape (n_windows, n_features).
    Keys: mean, std, median, max, p10, p90, energy [, periodicity].
    Empty dict if n_frames < window_size.
    """
    n_frames = data.shape[0]
    if n_frames < window_size:
        return {}

    # Pre-compute tensor size to decide GPU vs CPU path.
    # torch.quantile on CUDA hangs for very large numbers of small sorts
    # (e.g. 6 260 windows × 548 features at inference stride=3).
    # Keeping the threshold at 100 M elements covers normal training strides
    # (~41 M elements) on GPU while routing dense temporal-inference passes
    # (~206 M elements) to the CPU path, which completes in a few seconds.
    _MAX_GPU_WINDOW_ELEMENTS = 100_000_000
    n_windows = max(0, (n_frames - window_size) // stride + 1)
    n_features = data.shape[1] if data.ndim == 2 else 1
    total_elements = int(n_windows) * int(n_features) * int(window_size)
    _too_large_for_gpu = total_elements > _MAX_GPU_WINDOW_ELEMENTS

    global _GPU_STATS_DISABLED
    if (
        _GPU_STATS_ENABLED
        and gpu_available()
        and not _GPU_STATS_DISABLED
        and not _too_large_for_gpu
    ):
        try:
            return _windowed_stats_gpu(data, window_size, stride, include_periodicity)
        except Exception as exc:
            logger.warning(
                "GPU windowed aggregation failed (%s); disabling GPU stats for this session and falling back to CPU.", exc
            )
            _GPU_STATS_DISABLED = True

    if _GPU_STATS_ENABLED and _too_large_for_gpu and gpu_available():
        logger.debug(
            "Windowed stats: tensor too large for GPU (%d elements > %d threshold); using CPU path.",
            total_elements,
            _MAX_GPU_WINDOW_ELEMENTS,
        )

    return _windowed_stats_cpu(data, window_size, stride, include_periodicity)


def build_segment_df_fast(
    group_df: "pd.DataFrame",
    feature_cols: list[str],
    animal_id: str,
    session_id: str,
    window_size: int,
    stride: int,
    include_periodicity: bool = True,
    include_posture_deltas: bool = False,
) -> "pd.DataFrame":
    """Build a segment-summary DataFrame for one (animal, session) group.

    This is a drop-in replacement for the per-window Python loop used in both
    ``BehaviorRepresentationService`` and
    ``BehaviorAdaptiveFeatureCacheService``.

    When *include_posture_deltas* is True, angle and proximity columns (see
    :func:`_is_posture_delta_col`) additionally receive clip-wise ``_delta`` and
    ``_trend`` statistics — the same directional-change features always computed
    for ROI/target columns.
    """
    import pandas as pd

    work = group_df.sort_values("frame").reset_index(drop=True)
    # Drop rows with NaN frame indices before integer conversion to avoid
    # ValueError: cannot convert float NaN to integer (occurs when DLC pose
    # files have missing/null frame numbers for dropped or misaligned frames).
    if work["frame"].isna().any():
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "build_segment_df_fast: dropped %d row(s) with NaN frame index "
            "for animal=%s session=%s",
            int(work["frame"].isna().sum()), animal_id, session_id,
        )
        work = work.dropna(subset=["frame"]).reset_index(drop=True)
    n = len(work)
    if n < window_size:
        return pd.DataFrame()

    frames = work["frame"].to_numpy(dtype=int)
    data = work[feature_cols].to_numpy(dtype=np.float64)

    stats = windowed_feature_summary(
        data.astype(np.float32),
        window_size,
        stride,
        include_periodicity=include_periodicity,
    )
    if not stats:
        return pd.DataFrame()

    n_windows = stats["mean"].shape[0]
    window_starts = np.arange(0, n - window_size + 1, stride)[:n_windows]
    window_ends = window_starts + window_size - 1

    # Build metadata columns
    result: dict[str, object] = {
        "segment_id": [
            f"seg_{animal_id}_{session_id}_{int(frames[s])}_{int(frames[e])}"
            for s, e in zip(window_starts, window_ends)
        ],
        "start_frame": frames[window_starts].tolist(),
        "end_frame": frames[window_ends].tolist(),
        "animal_id": [str(animal_id)] * n_windows,
        "session_id": [str(session_id)] * n_windows,
    }

    # Build feature columns directly from the vectorised stat arrays
    stat_names = ["mean", "std", "median", "max", "p10", "p90", "energy"]
    if include_periodicity and "periodicity" in stats:
        stat_names.append("periodicity")

    for j, col in enumerate(feature_cols):
        for stat_name in stat_names:
            result[f"{col}_{stat_name}"] = stats[stat_name][:, j].astype(float)

    # ── Directional trajectory features (delta / trend) ─────────────────────
    # For selected columns we add two extra statistics that capture directional
    # change across the clip duration:
    #
    #   _delta  : mean(last k frames) − mean(first k frames), where k is an
    #             edge band of ~25% of the window.  Averaging each end (rather
    #             than using single endpoint frames) makes the net-displacement
    #             signal robust to per-frame tracking jitter/glitches at the
    #             clip boundaries.
    #             Negative distance delta  → parts/ROI drawing together.
    #             Positive distance delta  → parts/ROI moving apart.
    #
    #   _trend  : slope of the least-squares linear fit (units per frame).
    #             Also noise-robust because it uses all frames.
    #
    # These are the only statistics where aggregating across the window loses
    # the directional signal; mean/std alone cannot distinguish an animal
    # that moves toward a target from one that moves away at the same speed.
    #
    # ROI/target columns always get these (cheap, always meaningful).  Posture
    # angle/proximity columns get them only when clip-wise deltas are enabled.
    _delta_indices = [j for j, col in enumerate(feature_cols) if _is_roi_spatial_col(col)]
    if include_posture_deltas:
        _delta_indices += [
            j for j, col in enumerate(feature_cols) if _is_posture_delta_col(col)
        ]
    if _delta_indices:
        from numpy.lib.stride_tricks import sliding_window_view

        sub_data = data[:, _delta_indices].astype(np.float32)  # (n_frames, n_sub_cols)
        # Shape after sliding_window_view: (n_valid, n_sub_cols, window_size)
        sub_wins_all = sliding_window_view(sub_data, window_size, axis=0)
        sub_wins = sub_wins_all[::stride]  # (n_windows, n_sub_cols, window_size)

        # delta: signed end-minus-start change, averaged over an edge band at
        # each end so a single noisy boundary frame can't dominate.  k is ~25%
        # of the window, clamped so the two bands never overlap and are >= 1.
        _edge_k = max(1, min(window_size // 2, int(round(window_size * _DELTA_EDGE_FRACTION))))
        _start_mean = sub_wins[:, :, :_edge_k].mean(axis=2)
        _end_mean = sub_wins[:, :, -_edge_k:].mean(axis=2)
        sub_delta = _end_mean - _start_mean  # (n_windows, n_sub_cols)

        # trend: linear-regression slope over the window duration
        x_idx = np.arange(window_size, dtype=np.float32)
        x_dev = x_idx - x_idx.mean()
        x_ss = float(np.dot(x_dev, x_dev))  # sum of squared deviations
        # einsum contracts the window axis: result is (n_windows, n_sub_cols)
        sub_trend = np.einsum("ijk,k->ij", sub_wins, x_dev) / x_ss

        for local_idx, feat_idx in enumerate(_delta_indices):
            col = feature_cols[feat_idx]
            result[f"{col}_delta"] = sub_delta[:, local_idx].astype(float)
            result[f"{col}_trend"] = sub_trend[:, local_idx].astype(float)

    return pd.DataFrame(result)
