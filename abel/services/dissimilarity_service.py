"""Dissimilarity analysis for reviewed clips.

Loads segment-level features from the representations parquet, matches them to
reviewed clips for a given behavior, and computes per-clip outlier scores based
on Euclidean distance to the behaviour centroid in standardised feature space.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")


@dataclass
class DissimilarityResult:
    """Per-clip outlier score produced by the dissimilarity analysis."""

    window_id: str
    session_id: str
    start_frame: int
    end_frame: int
    behavior_id: str
    score: float  # 0 = most typical, 1 = most dissimilar
    is_outlier: bool


@dataclass
class DissimilarityReport:
    """Complete output from a dissimilarity analysis run."""

    behavior_id: str
    n_clips: int
    n_matched: int
    n_outliers: int
    results: list[DissimilarityResult] = field(default_factory=list)
    error: str | None = None


def run_dissimilarity_analysis(
    project_root: Path,
    reviewed_clips: list[dict],
    behavior_id: str,
    outlier_percentile: float = 90.0,
) -> DissimilarityReport:
    """Compute dissimilarity scores for reviewed clips of one behavior.

    Parameters
    ----------
    project_root:
        Project root directory.
    reviewed_clips:
        List of dicts with keys: ``window_id``, ``session_id``,
        ``start_frame``, ``end_frame``.
    behavior_id:
        The behavior these clips are labelled as.
    outlier_percentile:
        Clips above this percentile in distance are flagged as outliers.

    Returns
    -------
    DissimilarityReport
    """
    seg_path = project_root / "derived" / "representations" / "segment_features.parquet"
    if not seg_path.exists():
        return DissimilarityReport(
            behavior_id=behavior_id,
            n_clips=len(reviewed_clips),
            n_matched=0,
            n_outliers=0,
            error="Segment features not found. Run feature extraction first.",
        )

    try:
        seg_df = pd.read_parquet(seg_path)
    except Exception as exc:
        return DissimilarityReport(
            behavior_id=behavior_id,
            n_clips=len(reviewed_clips),
            n_matched=0,
            n_outliers=0,
            error=f"Failed to load segment features: {exc}",
        )

    # Identify numeric feature columns (exclude metadata & model output cols).
    meta_cols = {
        "segment_id", "start_frame", "end_frame", "animal_id", "session_id",
        "video_id", "prediction_variance", "density_outlier_score",
        "uncertainty_score", "prediction_prob", "prediction_prob_fused",
    }
    feature_cols = [c for c in seg_df.columns if c not in meta_cols and seg_df[c].dtype.kind == "f"]
    if not feature_cols:
        return DissimilarityReport(
            behavior_id=behavior_id,
            n_clips=len(reviewed_clips),
            n_matched=0,
            n_outliers=0,
            error="No numeric feature columns found in segment features.",
        )

    # Build lookup structures for matching clips to segment feature rows.
    # Segment features use a canonical window grid; reviewed clips may use
    # different frame ranges (random sampling, bout windows) so we match by
    # exact (session, start, end) first, then fall back to the nearest segment
    # centre.
    seg_df = seg_df.reset_index(drop=True)
    seg_lookup: dict[tuple[str, int, int], int] = {}
    for idx, row in seg_df[["session_id", "start_frame", "end_frame"]].iterrows():
        key = (str(row["session_id"]), int(row["start_frame"]), int(row["end_frame"]))
        seg_lookup[key] = int(idx)

    # Match reviewed clips to segment feature rows.
    matched_indices: list[int] = []
    matched_clips: list[dict] = []
    for clip in reviewed_clips:
        sid = str(clip["session_id"])
        sf = int(clip["start_frame"])
        ef = int(clip["end_frame"])

        # Exact match first.
        key = (sid, sf, ef)
        if key in seg_lookup:
            matched_indices.append(seg_lookup[key])
            matched_clips.append(clip)
            continue

        # Nearest-segment match: find the segment whose centre frame is
        # closest to the clip's centre.  This handles clips that fall between
        # grid positions (e.g. randomly sampled or bout windows).
        best_idx = _nearest_segment_match(seg_df, sid, sf, ef)
        if best_idx is not None:
            matched_indices.append(best_idx)
            matched_clips.append(clip)

    n_matched = len(matched_indices)
    if n_matched < 3:
        return DissimilarityReport(
            behavior_id=behavior_id,
            n_clips=len(reviewed_clips),
            n_matched=n_matched,
            n_outliers=0,
            error=(
                f"Only {n_matched} clip(s) matched to segment features — need at "
                f"least 3 for a meaningful analysis."
            ),
        )

    # Extract the feature matrix for matched rows.
    feat_matrix = seg_df.loc[matched_indices, feature_cols].to_numpy(dtype=np.float64)

    # Replace NaN/Inf with column medians so distance calculations are stable.
    col_medians = np.nanmedian(feat_matrix, axis=0)
    for j in range(feat_matrix.shape[1]):
        bad = ~np.isfinite(feat_matrix[:, j])
        feat_matrix[bad, j] = col_medians[j] if np.isfinite(col_medians[j]) else 0.0

    # Standardize features to zero mean / unit variance.
    means = feat_matrix.mean(axis=0)
    stds = feat_matrix.std(axis=0)
    stds[stds < 1e-12] = 1.0
    feat_matrix = (feat_matrix - means) / stds

    # Compute Euclidean distance from each clip to the group centroid in the
    # standardised space.  After standardisation the centroid sits at the origin
    # so the distance is simply the L2 norm of each row.
    distances = np.linalg.norm(feat_matrix, axis=1)

    # Normalize to [0, 1] for interpretability (0 = most typical, 1 = most dissimilar).
    d_min = float(distances.min())
    d_max = float(distances.max())
    if d_max - d_min > 1e-12:
        scores = (distances - d_min) / (d_max - d_min)
    else:
        scores = np.zeros(n_matched)

    threshold = float(np.percentile(scores, outlier_percentile)) if n_matched > 1 else 1.0

    results: list[DissimilarityResult] = []
    n_outliers = 0
    for i, clip in enumerate(matched_clips):
        is_outlier = bool(scores[i] >= threshold)
        if is_outlier:
            n_outliers += 1
        results.append(
            DissimilarityResult(
                window_id=clip["window_id"],
                session_id=clip["session_id"],
                start_frame=int(clip["start_frame"]),
                end_frame=int(clip["end_frame"]),
                behavior_id=behavior_id,
                score=round(float(scores[i]), 4),
                is_outlier=is_outlier,
            )
        )

    # Sort by descending score so the most dissimilar clips appear first.
    results.sort(key=lambda r: r.score, reverse=True)

    return DissimilarityReport(
        behavior_id=behavior_id,
        n_clips=len(reviewed_clips),
        n_matched=n_matched,
        n_outliers=n_outliers,
        results=results,
    )


def _nearest_segment_match(
    seg_df: pd.DataFrame, session_id: str, start: int, end: int
) -> int | None:
    """Find the segment row whose centre is closest to the clip centre."""
    mask = seg_df["session_id"] == session_id
    subset = seg_df.loc[mask]
    if subset.empty:
        return None

    clip_centre = (start + end) / 2.0
    seg_starts = subset["start_frame"].to_numpy(dtype=np.float64)
    seg_ends = subset["end_frame"].to_numpy(dtype=np.float64)
    seg_centres = (seg_starts + seg_ends) / 2.0
    dists = np.abs(seg_centres - clip_centre)
    best_local = int(np.argmin(dists))
    return int(subset.index[best_local])
