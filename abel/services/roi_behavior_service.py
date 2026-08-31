"""Restrict behavior bouts to a region of interest.

The analytics tab already reports *where* the animal was (ROI occupancy) and
*what* it did (bout counts, durations, latencies) as two independent views.
This module joins them: given the frame-level "is the animal inside zone N"
mask for a session, it rewrites a behavior's bout table so every downstream
metric -- summary table, every chart style, statistics, ethograms, exports --
describes only the behavior that happened inside (or outside) that zone.

Attribution
-----------
A bout can straddle the ROI boundary, so "did this bout happen in the zone?"
has more than one defensible answer.  The three supported modes trade off
differently and are exposed to the user rather than hard-coded:

``overlap``
    Split the bout at the boundary and keep the in-zone fragments.  Durations
    are exact (a 10 s groom half inside contributes 5 s), but a single bout
    that crosses the boundary twice becomes two bouts, so counts inflate.
``onset``
    Keep the whole bout when it *starts* in the zone, otherwise drop it.
    Counts and latencies stay comparable to the unscoped analysis; durations
    include time spent after the animal left.
``majority``
    Keep the whole bout when more than half its frames are in the zone.  Like
    ``onset`` for counts, but less sensitive to exactly where a bout began.

Frame indexing
--------------
Bout frames in analytics are *prechop-rebased* (frame 0 = test start) while
pose arrays -- and therefore ROI masks -- stay in raw video frames.  Callers
pass the per-session prechop offset so this module can line the two up; see
``BehaviorAnalyticsTab._unrebase_bout_df_to_video_frames`` for the same
correction applied to spatial views.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from abel.utils import roi_geometry

# Attribution modes (see the module docstring).
ATTRIBUTION_OVERLAP = "overlap"
ATTRIBUTION_ONSET = "onset"
ATTRIBUTION_MAJORITY = "majority"

# Short enough for a toolbar combo; the trade-offs live in this docstring and
# in the control's tooltip.
ATTRIBUTION_LABELS: dict[str, str] = {
    ATTRIBUTION_OVERLAP: "Split at boundary",
    ATTRIBUTION_ONSET: "By bout onset",
    ATTRIBUTION_MAJORITY: "By majority in zone",
}

# Boundary flicker shorter than this is merged into the surrounding run, so a
# jittery tracking frame cannot split one bout into three.  Matches the
# debounce the ROI occupancy metrics already use.
DEFAULT_DEBOUNCE_S: float = 0.2

_BOUT_KEYS = ("session_id", "start_frame", "end_frame")


def roi_inside_mask(
    pose: Any,
    roi: dict | None,
    fps: float,
    *,
    debounce_s: float = DEFAULT_DEBOUNCE_S,
) -> np.ndarray | None:
    """Return a per-video-frame boolean mask of "body centroid inside *roi*".

    Returns None when the ROI has no area or the pose data is unusable, which
    callers treat as "this session cannot be scoped" rather than "the animal
    was never inside".
    """
    if pose is None or not roi or not roi_geometry.roi_has_area(roi):
        return None
    try:
        cx = np.asarray(pose.centroid_x, dtype=np.float64)
        cy = np.asarray(pose.centroid_y, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None
    if cx.size == 0 or cx.size != cy.size:
        return None

    inside = roi_geometry.roi_contains(roi, cx, cy)
    min_run = max(1, int(round(debounce_s * fps))) if fps > 0 else 1
    return roi_geometry.debounce_bool(np.asarray(inside, dtype=bool), min_run)


def _runs(sel: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs for each True run in *sel*."""
    if sel.size == 0 or not sel.any():
        return []
    padded = np.concatenate(([False], sel, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, len(edges), 2)]


def scope_bouts_to_roi(
    bouts: pd.DataFrame,
    masks: Mapping[str, "np.ndarray | None"],
    offsets: Mapping[str, int] | None = None,
    *,
    inside: bool = True,
    attribution: str = ATTRIBUTION_OVERLAP,
    min_frames: int = 1,
) -> tuple[pd.DataFrame, set[str]]:
    """Restrict *bouts* to the frames matching an ROI membership test.

    Args:
        bouts: bout table with ``session_id``/``start_frame``/``end_frame``
            (prechop-rebased). Any other columns are carried through unchanged.
        masks: ``session_id`` -> per-video-frame inside mask (None = unscopable).
        offsets: ``session_id`` -> prechop offset added to bout frames before
            indexing into the mask.
        inside: True keeps in-zone behavior, False keeps out-of-zone behavior.
        attribution: one of the ``ATTRIBUTION_*`` constants.
        min_frames: drop scoped fragments shorter than this (``overlap`` only).

    Returns:
        ``(scoped_bouts, unscopable_session_ids)``.  Sessions with no usable
        mask are dropped from the result and named in the second element so the
        caller can say so rather than silently reporting whole-arena numbers
        under an ROI label.
    """
    if bouts is None or bouts.empty or not set(_BOUT_KEYS).issubset(bouts.columns):
        return (bouts if bouts is not None else pd.DataFrame()), set()

    offsets = offsets or {}
    attribution = str(attribution or ATTRIBUTION_OVERLAP)
    unscopable: set[str] = set()
    kept_index: list[Any] = []            # rows kept whole (onset / majority)
    fragments: list[dict[str, Any]] = []  # rows rewritten (overlap)

    starts_all = pd.to_numeric(bouts["start_frame"], errors="coerce")
    ends_all = pd.to_numeric(bouts["end_frame"], errors="coerce")

    for sid, grp in bouts.groupby(bouts["session_id"].astype(str), sort=False):
        mask = masks.get(str(sid))
        if mask is None or getattr(mask, "size", 0) == 0:
            unscopable.add(str(sid))
            continue
        mask = np.asarray(mask, dtype=bool)
        if not inside:
            mask = ~mask
        off = int(offsets.get(str(sid), 0) or 0)
        n = int(mask.size)

        for idx in grp.index:
            s_raw, e_raw = starts_all.at[idx], ends_all.at[idx]
            if not np.isfinite(s_raw) or not np.isfinite(e_raw):
                continue
            s = int(s_raw) + off
            e = int(e_raw) + off
            if e < s:
                continue
            # Clamp into the mask; a bout entirely past the end of the pose
            # track cannot be judged, so it is dropped rather than assumed.
            lo, hi = max(0, s), min(n - 1, e)
            if hi < lo:
                continue
            sel = mask[lo:hi + 1]
            if not sel.any():
                continue

            if attribution == ATTRIBUTION_ONSET:
                if bool(sel[0]):
                    kept_index.append(idx)
            elif attribution == ATTRIBUTION_MAJORITY:
                if float(sel.mean()) > 0.5:
                    kept_index.append(idx)
            else:  # ATTRIBUTION_OVERLAP
                base = bouts.loc[idx].to_dict()
                for r0, r1 in _runs(sel):
                    if (r1 - r0 + 1) < max(1, int(min_frames)):
                        continue
                    frag = dict(base)
                    frag["start_frame"] = int(lo + r0 - off)
                    frag["end_frame"] = int(lo + r1 - off)
                    fragments.append(frag)

    if attribution in (ATTRIBUTION_ONSET, ATTRIBUTION_MAJORITY):
        out = bouts.loc[kept_index].copy()
    elif fragments:
        out = pd.DataFrame(fragments, columns=list(bouts.columns))
    else:
        out = bouts.iloc[0:0].copy()

    if not out.empty:
        out["start_frame"] = out["start_frame"].astype(np.int64)
        out["end_frame"] = out["end_frame"].astype(np.int64)
        out = out.sort_values(["session_id", "start_frame"], kind="stable")
    return out.reset_index(drop=True), unscopable


def summarize_bouts(
    bouts: pd.DataFrame, fps: float,
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Aggregate a bout table into ``(sid, bid)`` -> (n, time_s, mean_s, latency_s)."""
    stats: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    if bouts is None or bouts.empty or not set(_BOUT_KEYS).issubset(bouts.columns):
        return stats
    if fps <= 0:
        fps = 1.0
    has_bid = "behavior_id" in bouts.columns
    for sid, grp in bouts.groupby(bouts["session_id"].astype(str), sort=False):
        sub_iter = (
            grp.groupby(grp["behavior_id"].astype(str), sort=False)
            if has_bid else [("", grp)]
        )
        for bid, sub in sub_iter:
            n_b = float(len(sub))
            total_f = float((sub["end_frame"] - sub["start_frame"] + 1).sum())
            time_s = total_f / fps
            mean_s = time_s / n_b if n_b > 0 else 0.0
            lat_s = float(sub["start_frame"].min()) / fps if n_b > 0 else float("nan")
            stats[(str(sid), str(bid))] = (n_b, time_s, mean_s, lat_s)
    return stats
