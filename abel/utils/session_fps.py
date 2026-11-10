"""Per-session frame rate for projects that mix recording rates.

A project stores one window, stride and "Source FPS".  That is right when every
video shares a frame rate.  When they do not, a fixed frame count covers a
different stretch of real time in each video, and one fps mis-scales every
velocity.  The helpers here give each session a scale relative to the project
rate so windows cover the same seconds everywhere.

Projects whose videos all share one rate get an empty scale map, so nothing
about their extraction, segment ids or caches changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

MIXED_FPS_TOLERANCE = 0.01
"""Relative difference below which two rates count as the same (29.97 vs 30)."""


def _positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def video_fps_by_session(
    project_root: Path | None = None, manifest: Any = None
) -> dict[str, float]:
    """Probed video fps per session id.  Sessions with no known fps are left out."""
    videos: list[tuple[Any, Any]] = []
    sessions: list[tuple[Any, Any]] = []
    if manifest is not None:
        videos = [(v.asset_id, getattr(v, "fps", None)) for v in getattr(manifest, "videos", [])]
        sessions = [
            (s.session_id, getattr(s, "video_asset_id", None))
            for s in getattr(manifest, "linked_sessions", [])
        ]
    elif project_root is not None:
        path = Path(project_root) / "derived" / "review_tables" / "import_manifest.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        videos = [(v.get("asset_id"), v.get("fps")) for v in raw.get("videos") or []]
        sessions = [
            (s.get("session_id"), s.get("video_asset_id"))
            for s in raw.get("linked_sessions") or []
        ]
    fps_by_asset = {aid: _positive(fps) for aid, fps in videos}
    out: dict[str, float] = {}
    for sid, asset_id in sessions:
        fps = fps_by_asset.get(asset_id)
        if sid is not None and fps is not None:
            out[str(sid)] = fps
    return out


def reference_fps(project_root: Path | None) -> float:
    """The project rate the stored window and stride frame counts refer to."""
    if project_root is None:
        return 30.0
    try:
        from abel.storage.file_store import read_yaml  # noqa: PLC0415

        raw = read_yaml(Path(project_root) / "project.yaml", {}) or {}
    except Exception:
        return 30.0
    extraction = raw.get("feature_extraction") or {}
    return _positive(extraction.get("source_fps")) or _positive(raw.get("default_fps")) or 30.0


def is_mixed(fps_values: Iterable[float]) -> bool:
    """True when the rates differ by more than the tolerance."""
    values = [v for v in (_positive(f) for f in fps_values) if v is not None]
    if len(values) < 2:
        return False
    return max(values) / min(values) - 1.0 > MIXED_FPS_TOLERANCE


def session_fps_scales(
    project_root: Path | None = None,
    manifest: Any = None,
    reference: float | None = None,
) -> dict[str, float]:
    """``session fps / project fps`` for each session that needs rescaling.

    Empty unless the project really mixes frame rates.  In a mixed project,
    sessions within the tolerance of the project rate are also left out, so they
    keep the exact frame counts they always had.
    """
    fps = video_fps_by_session(project_root, manifest)
    if not is_mixed(fps.values()):
        return {}
    ref = _positive(reference) or reference_fps(project_root)
    out: dict[str, float] = {}
    for sid, value in fps.items():
        scale = value / ref
        if abs(scale - 1.0) > MIXED_FPS_TOLERANCE:
            out[sid] = scale
    return out


def scaled_frames(frames: int, scale: float | None, minimum: int = 1) -> int:
    """A project frame count converted to one session's frame rate."""
    if not scale or scale == 1.0:
        return int(frames)
    return max(int(minimum), int(round(int(frames) * float(scale))))


def session_fps(reference: float, scale: float | None) -> float:
    """The fps to extract a session at: the project rate unless it is rescaled."""
    return float(reference) * float(scale) if scale else float(reference)


def describe_mixed(fps_by_session: dict[str, float]) -> str:
    """Short "3 at 30 fps, 2 at 60 fps" summary for logs and warnings."""
    counts: dict[float, int] = {}
    for value in fps_by_session.values():
        key = round(float(value), 2)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{n} at {fps:g} fps" for fps, n in sorted(counts.items()))
