"""Window samplers for center-frame temporal refinement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSample:
    """One fixed-length window centered on a labeled frame."""

    session_id: str
    subject_id: str
    concept_id: str
    center_frame: int
    start_frame: int
    end_frame: int
    pad_left: int = 0
    pad_right: int = 0
    label: int = 0
    source: str = "negative"


@dataclass(frozen=True)
class SessionSpec:
    """Minimal session context required by the sampler."""

    session_id: str
    subject_id: str
    n_frames: int


def window_bounds(center_frame: int, window_frames: int) -> tuple[int, int]:
    """Return inclusive [start, end] bounds centered around ``center_frame``."""
    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")
    left = window_frames // 2
    right = window_frames - left - 1
    return center_frame - left, center_frame + right


def _build_sample(
    session: SessionSpec,
    concept_id: str,
    center_frame: int,
    window_frames: int,
    label: int,
    source: str,
    edge_mode: str,
) -> WindowSample | None:
    start, end = window_bounds(center_frame=center_frame, window_frames=window_frames)
    pad_left = max(0, -start)
    pad_right = max(0, end - (session.n_frames - 1))

    if (pad_left > 0 or pad_right > 0) and edge_mode == "skip":
        return None
    if edge_mode not in {"skip", "pad"}:
        raise ValueError("edge_mode must be one of {'skip', 'pad'}")

    return WindowSample(
        session_id=session.session_id,
        subject_id=session.subject_id,
        concept_id=concept_id,
        center_frame=int(center_frame),
        start_frame=max(0, int(start)),
        end_frame=min(session.n_frames - 1, int(end)),
        pad_left=int(pad_left),
        pad_right=int(pad_right),
        label=int(label),
        source=str(source),
    )


def _iter_interval_centers(
    intervals: list[tuple[int, int]],
    stride: int,
) -> list[int]:
    centers: list[int] = []
    stride = max(1, int(stride))
    for start, end in intervals:
        if end < start:
            continue
        centers.extend(range(int(start), int(end) + 1, stride))
    return centers


def sample_positive_windows(
    session: SessionSpec,
    bouts: list[tuple[int, int]],
    window_frames: int,
    stride: int,
    concept_id: str = "target_behavior",
    edge_mode: str = "pad",
) -> list[WindowSample]:
    """Sample positive windows whose center frame lies inside confirmed bouts."""
    centers = _iter_interval_centers(bouts, stride=stride)
    out: list[WindowSample] = []
    for center in centers:
        sample = _build_sample(
            session=session,
            concept_id=concept_id,
            center_frame=center,
            window_frames=window_frames,
            label=1,
            source="positive",
            edge_mode=edge_mode,
        )
        if sample is not None:
            out.append(sample)
    return out


def sample_negative_windows(
    session: SessionSpec,
    bouts: list[tuple[int, int]],
    window_frames: int,
    stride: int,
    concept_id: str = "target_behavior",
    edge_mode: str = "pad",
    include_frames: set[int] | None = None,
) -> list[WindowSample]:
    """Sample negatives from center frames outside confirmed positive bouts."""
    occupied = [False] * max(0, int(session.n_frames))
    for start, end in bouts:
        for idx in range(max(0, int(start)), min(session.n_frames, int(end) + 1)):
            occupied[idx] = True

    if include_frames is None:
        centers = [i for i in range(0, session.n_frames, max(1, int(stride))) if not occupied[i]]
    else:
        centers = [i for i in range(0, session.n_frames, max(1, int(stride))) if (i in include_frames and not occupied[i])]

    out: list[WindowSample] = []
    for center in centers:
        sample = _build_sample(
            session=session,
            concept_id=concept_id,
            center_frame=center,
            window_frames=window_frames,
            label=0,
            source="negative",
            edge_mode=edge_mode,
        )
        if sample is not None:
            out.append(sample)
    return out


def sample_hard_negative_windows(
    session: SessionSpec,
    concept_id: str,
    window_frames: int,
    stride: int,
    hard_negative_intervals: list[tuple[int, int]] | None = None,
    edge_mode: str = "pad",
) -> list[WindowSample]:
    """Sample hard negatives from concept-adjacent or reviewer-rejected intervals."""
    intervals = hard_negative_intervals or []
    centers = _iter_interval_centers(intervals, stride=max(1, int(stride)))
    out: list[WindowSample] = []
    for center in centers:
        sample = _build_sample(
            session=session,
            concept_id=concept_id,
            center_frame=center,
            window_frames=window_frames,
            label=0,
            source="hard_negative",
            edge_mode=edge_mode,
        )
        if sample is not None:
            out.append(sample)
    return out
