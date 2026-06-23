"""Center-frame label construction for temporal refinement."""

from __future__ import annotations

import pandas as pd


def expand_or_shrink_training_intervals(
    intervals_by_session: dict[str, list[tuple[int, int]]],
    margin_frames: int,
    session_lengths: dict[str, int] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Expand (positive margin) or shrink (negative margin) intervals."""
    margin = int(margin_frames)
    out: dict[str, list[tuple[int, int]]] = {}
    for session_id, intervals in intervals_by_session.items():
        n_frames = int((session_lengths or {}).get(session_id, 0))
        adjusted: list[tuple[int, int]] = []
        for start, end in intervals:
            s = int(start) - margin
            e = int(end) + margin
            if margin < 0:
                s = int(start) + abs(margin)
                e = int(end) - abs(margin)
            if n_frames > 0:
                s = max(0, s)
                e = min(n_frames - 1, e)
            if e >= s:
                adjusted.append((s, e))
        out[session_id] = adjusted
    return out


def center_label_for_frame(
    frame_idx: int,
    reviewed_intervals: list[tuple[int, int]],
) -> int:
    """Return 1 if frame is inside any reviewed positive interval, else 0."""
    f = int(frame_idx)
    for start, end in reviewed_intervals:
        if int(start) <= f <= int(end):
            return 1
    return 0


def _excluded_boundary_mask(
    n_frames: int,
    intervals: list[tuple[int, int]],
    boundary_exclusion_frames: int,
) -> list[bool]:
    mask = [False] * max(0, int(n_frames))
    k = max(0, int(boundary_exclusion_frames))
    if k <= 0:
        return mask

    for start, end in intervals:
        s = int(start)
        e = int(end)
        for idx in range(max(0, s - k), min(n_frames, s + k + 1)):
            mask[idx] = True
        for idx in range(max(0, e - k), min(n_frames, e + k + 1)):
            mask[idx] = True
    return mask


def exclude_uncertain_frames(
    labels_df: pd.DataFrame,
    uncertain_intervals_by_session: dict[str, list[tuple[int, int]]],
) -> pd.DataFrame:
    """Mark uncertain frames as excluded from default training."""
    if labels_df.empty:
        return labels_df

    out = labels_df.copy()
    out["excluded"] = out["excluded"].astype(bool)
    out["exclude_reason"] = out["exclude_reason"].astype(str)

    for session_id, intervals in uncertain_intervals_by_session.items():
        if not intervals:
            continue
        sess_mask = out["session_id"].astype(str) == str(session_id)
        if not bool(sess_mask.any()):
            continue
        frame_series = out.loc[sess_mask, "frame"].astype(int)
        local_idx = out.loc[sess_mask].index
        should_exclude = pd.Series(False, index=local_idx)
        for start, end in intervals:
            should_exclude = should_exclude | ((frame_series >= int(start)) & (frame_series <= int(end)))

        selected = should_exclude[should_exclude].index
        out.loc[selected, "excluded"] = True
        out.loc[selected, "exclude_reason"] = out.loc[selected, "exclude_reason"].replace("", "uncertain")

    return out


def build_centerframe_labels_from_reviews(
    session_lengths: dict[str, int],
    reviewed_intervals_by_session: dict[str, list[tuple[int, int]]],
    boundary_exclusion_frames: int = 2,
    uncertain_intervals_by_session: dict[str, list[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    """Build per-frame labels with explicit ambiguity exclusions near boundaries."""
    rows: list[dict[str, object]] = []
    for session_id, n_frames in session_lengths.items():
        intervals = reviewed_intervals_by_session.get(session_id, [])
        boundary_mask = _excluded_boundary_mask(
            n_frames=int(n_frames),
            intervals=intervals,
            boundary_exclusion_frames=boundary_exclusion_frames,
        )
        for frame_idx in range(int(n_frames)):
            excluded = bool(boundary_mask[frame_idx])
            rows.append(
                {
                    "session_id": str(session_id),
                    "frame": int(frame_idx),
                    "label": int(center_label_for_frame(frame_idx, intervals)),
                    "excluded": excluded,
                    "exclude_reason": "boundary" if excluded else "",
                }
            )

    out = pd.DataFrame(rows)
    if uncertain_intervals_by_session:
        out = exclude_uncertain_frames(out, uncertain_intervals_by_session)
    return out
