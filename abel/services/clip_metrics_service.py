"""Per-clip metric computation, targeted mining, and essence extraction.

This module powers **Targeted Clip Mining**: it turns each candidate window
into a small vector of interpretable, physically-meaningful metrics (time spent
in a zone, distance to a zone, centroid speed, distance travelled, body
elongation, …) and then lets the UI

* **mine** — find every clip whose metrics satisfy a set of user-defined
  criteria (``metric between low/high``), and
* **extract essence** — infer those criteria automatically from a handful of
  exemplar clips the user hand-picked (the overlapping value range per metric).

Metrics are derived from the project's cleaned pose plus the subject/session ROI
so they need no extra feature-extraction pass.  Everything is expressed in real
units (mm, mm/s, fractions, degrees) so criteria transfer across recordings with
different pixel scales.

Pipeline position:
    Pose Features + ROI geometry → **Clip Metrics / Mining** ← here
    → Review (targeted queue)
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.roi_service import MAX_ROIS, ROIService
from abel.storage.file_store import read_yaml, write_yaml
from abel.utils.roi_geometry import (
    roi_contains,
    roi_has_area,
    roi_signed_distance,
)

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDef:
    """One mine-able per-clip metric."""

    id: str
    label: str
    group: str
    unit: str
    description: str


# ── ROI-scoped metric specs ──────────────────────────────────────────────
# One block of these is generated per configured target zone, so a project
# with N ROIs exposes N copies (ROI 1 … ROI N).  (base_id, label, unit, desc).
_ROI_METRIC_SPECS: list[tuple[str, str, str, str]] = [
    ("centroid_in_roi_frac", "Time in zone (fraction)", "0–1",
     "Fraction of the clip the body centroid spends inside the zone (occupancy)."),
    ("centroid_dist_to_roi_mm", "Distance to zone (mean)", "mm",
     "Mean distance from the body centroid to the zone edge (ROI proximity). "
     "Negative = inside the zone, positive = outside — lower is nearer/inside."),
    ("centroid_dist_to_roi_min_mm", "Closest approach to zone", "mm",
     "Nearest the centroid gets to the zone edge over the clip. "
     "Negative = it entered the zone (how deep); positive = closest it came from outside."),
    ("roi_entry_count", "Zone entries (count)", "",
     "Number of times the centroid crosses from outside to inside the zone (visits)."),
    ("nose_in_roi_frac", "Nose in zone (fraction)", "0–1",
     "Fraction of frames the nose is inside the zone."),
    ("nose_dist_to_roi_mm", "Nose distance to zone (mean)", "mm",
     "Mean distance from the nose to the zone edge. Negative = inside, positive = outside."),
    ("nose_off_edge_mm", "Nose past edge (max)", "mm",
     "Furthest the nose reaches beyond the zone edge during the clip. "
     "Positive = crossed the edge; negative = never reached it."),
    ("tail_in_roi_frac", "Tail in zone (fraction)", "0–1",
     "Fraction of frames the tail base is inside the zone."),
]

# Centroid speed (mm/s) below which a frame counts as "immobile" — a rough,
# generic near-still cutoff used by the immobility metric.
IMMOBILE_MM_S = 15.0

# ── ROI-independent metrics (motion, space, posture, timing) ──────────────
_NONROI_METRIC_DEFS: list[MetricDef] = [
    # ── Motion ───────────────────────────────────────────────────────────
    MetricDef("centroid_speed_mean", "Centroid speed (mean)", "Motion", "mm/s",
              "Average body-centroid speed across the clip."),
    MetricDef("centroid_speed_max", "Centroid speed (max)", "Motion", "mm/s",
              "Peak body-centroid speed across the clip."),
    MetricDef("centroid_speed_std", "Centroid speed (variability)", "Motion", "mm/s",
              "Standard deviation of centroid speed — steady vs bursty movement."),
    MetricDef("centroid_accel_mean", "Centroid acceleration (mean)", "Motion", "mm/s²",
              "Average magnitude of frame-to-frame change in centroid speed."),
    MetricDef("immobile_frac", "Immobile time (fraction)", "Motion", "0–1",
              "Fraction of frames the centroid moves slower than ~15 mm/s (near-still)."),
    MetricDef("nose_speed_mean", "Nose speed (mean)", "Motion", "mm/s",
              "Average nose speed across the clip."),
    MetricDef("nose_speed_max", "Nose speed (max)", "Motion", "mm/s",
              "Peak nose speed across the clip."),
    MetricDef("nose_speed_std", "Nose speed (variability)", "Motion", "mm/s",
              "Standard deviation of nose speed."),
    MetricDef("tail_speed_mean", "Tail speed (mean)", "Motion", "mm/s",
              "Average tail-base speed across the clip."),
    MetricDef("nose_travel_mm", "Nose travel (range)", "Motion", "mm",
              "Bounding extent of nose positions (how far the nose ranges)."),
    MetricDef("centroid_path_mm", "Distance travelled", "Motion", "mm",
              "Total path length of the body centroid across the clip (how much it moved)."),
    MetricDef("centroid_displacement_mm", "Net displacement", "Motion", "mm",
              "Straight-line distance from the clip's first to last centroid position."),
    MetricDef("path_tortuosity", "Path tortuosity", "Motion", "×",
              "Centroid path length ÷ net displacement — 1 = straight, higher = more winding."),
    MetricDef("turning_total_deg", "Total turning", "Motion", "°",
              "Sum of absolute heading changes of the centroid across the clip."),
    MetricDef("turning_rate_mean", "Turning rate (mean)", "Motion", "°/s",
              "Average absolute heading change per second (how sharply it turns)."),
    # ── Space ────────────────────────────────────────────────────────────
    MetricDef("centroid_travel_mm", "Centroid travel (range)", "Space", "mm",
              "Bounding extent of centroid positions (how far the body ranges)."),
    MetricDef("explore_area_mm2", "Area explored", "Space", "mm²",
              "Area of the centroid's bounding box (how much ground it covers)."),
    # ── Posture ──────────────────────────────────────────────────────────
    MetricDef("body_length_mean", "Body length (mean)", "Posture", "mm",
              "Average nose-to-tail-base distance (elongation)."),
    MetricDef("body_length_max", "Body length (max)", "Posture", "mm",
              "Peak nose-to-tail-base distance (maximum stretch)."),
    MetricDef("body_length_range_mm", "Body length (range)", "Posture", "mm",
              "Max − min nose-to-tail-base distance (how much it stretches/compresses)."),
    MetricDef("body_stretch_ratio", "Stretch ratio", "Posture", "×",
              "Max body length ÷ median body length — how much the animal elongates."),
    # ── Timing ───────────────────────────────────────────────────────────
    MetricDef("duration_sec", "Duration", "Timing", "s",
              "Clip length in seconds."),
]


def roi_metric_id(base_id: str, roi_index: int) -> str:
    """Metric id for *base_id* on target zone *roi_index* (0-based).

    ROI 1 keeps the bare base id, so saved criteria and single-zone projects are
    untouched; later zones get a ``__roi{N}`` suffix (e.g. ``nose_off_edge_mm__roi2``).
    """
    return base_id if roi_index <= 0 else f"{base_id}__roi{roi_index + 1}"


def build_metric_defs(roi_count: int) -> list[MetricDef]:
    """Return the display-ordered metric list for a project with *roi_count* zones.

    A single-zone project reproduces the original registry exactly (ROI metrics
    grouped under ``"Edge & ROI"``); multi-zone projects emit one ROI-metric block
    per zone, grouped and labelled ``"ROI 1" … "ROI N"`` so each zone is selectable.
    ROI-independent motion/posture/timing metrics always follow, once.
    """
    roi_count = max(1, int(roi_count))
    defs: list[MetricDef] = []
    for i in range(roi_count):
        group = "Edge & ROI" if roi_count == 1 else f"ROI {i + 1}"
        for base_id, label, unit, desc in _ROI_METRIC_SPECS:
            defs.append(MetricDef(roi_metric_id(base_id, i), label, group, unit, desc))
    defs.extend(_NONROI_METRIC_DEFS)
    return defs


# Default (single-zone) registry — preserved verbatim for backward compatibility.
METRIC_DEFS: list[MetricDef] = build_metric_defs(1)

# Comprehensive id → def lookup covering every zone variant, so label/unit
# lookups (display, violation strings) resolve any metric id regardless of how
# many ROIs the active project has.  Base ids keep their single-zone "Edge & ROI"
# def; later-zone variants carry their "ROI N" group.
METRIC_BY_ID: dict[str, MetricDef] = {m.id: m for m in build_metric_defs(MAX_ROIS)}
METRIC_BY_ID.update({m.id: m for m in METRIC_DEFS})
METRIC_IDS: list[str] = [m.id for m in METRIC_DEFS]


@dataclass
class ClipRef:
    """A window to score: identity plus its session and frame span."""

    window_id: str
    session_id: str
    start_frame: int
    end_frame: int


@dataclass
class Criterion:
    """A single mining constraint: ``low <= metric <= high`` (either bound optional)."""

    metric_id: str
    low: float | None = None
    high: float | None = None
    enabled: bool = True


@dataclass
class MiningResult:
    matched_ids: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # window_id -> match strength 0–1
    n_evaluated: int = 0


@dataclass
class EssenceCheckResult:
    """Audit of a known clip set against essence ranges (the inverse of mining).

    ``passed_ids`` sit inside the acceptable ranges; ``failed_ids`` fall outside
    at least one (AND mode) or all (OR mode).  ``violations`` maps each failed
    window to human-readable reasons (``"nose off edge 6.2 < 10 mm"``).
    ``no_data_ids`` had no finite value for any active criterion, so they could
    not be judged and are neither passed nor failed.
    """

    passed_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    no_data_ids: list[str] = field(default_factory=list)
    n_evaluated: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _format_violation(c: "Criterion", val: float, below: bool) -> str:
    """Readable reason a clip is outside an essence range, e.g. ``nose off edge 6.2 < 10 mm``."""
    m = METRIC_BY_ID.get(c.metric_id)
    label = m.label if m else c.metric_id
    # Qualify with the zone name when it's a per-ROI metric ("ROI 2 …"), so
    # violations stay unambiguous across zones; single-zone stays unprefixed.
    if m and m.group.startswith("ROI"):
        label = f"{m.group} {label}"
    # Append only genuine measurement units (mm, mm/s, s) — skip scale
    # descriptors like "0–1" or "×" that read as noise after a comparison.
    unit = f" {m.unit}" if m and m.unit and any(ch.isalpha() for ch in m.unit) else ""
    if below:
        return f"{label} {val:.1f} < {float(c.low):g}{unit}"
    return f"{label} {val:.1f} > {float(c.high):g}{unit}"


def _split_even(items: list[ClipRef], k: int) -> list[list[ClipRef]]:
    """Split *items* into up to *k* contiguous, near-equal chunks."""
    k = max(1, min(int(k), len(items)))
    size = -(-len(items) // k)  # ceil division
    return [items[i : i + size] for i in range(0, len(items), size)]


def _score_clips_task(
    project_root_str: str, refs: list[ClipRef]
) -> dict[str, dict[str, float]]:
    """Worker-process entry: score a batch of clips for one project.

    Rebuilds a fresh :class:`ClipMetricsService` inside the worker — its pose /
    ROI caches and loaded manifest can't be pickled across the process boundary —
    then returns only the per-window metric dicts, which are small enough to ship
    back cheaply.  ``refs`` is grouped by session by the caller, so each session's
    pose is loaded exactly once within this worker.
    """
    svc = ClipMetricsService()
    svc.set_project(Path(project_root_str))
    return {ref.window_id: svc._metrics_for_clip(ref) for ref in refs}


class ClipMetricsService:
    """Compute per-clip metrics and run mining / essence extraction."""

    # Persisted criteria live with the project so a saved feature set survives
    # a reload (metric ids + ranges + AND/OR mode).
    CRITERIA_FILE = Path("config") / "clip_mining_criteria.yaml"

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._pose_svc = PoseProcessingService()
        self._rois = ROIService()
        self._manifest = None
        self._fps: float = 30.0
        self._roi_count: int = 1
        self._metric_defs: list[MetricDef] = list(METRIC_DEFS)
        self._metric_ids: list[str] = list(METRIC_IDS)
        self._pose_cache: dict[str, object | None] = {}
        self._roi_cache: dict[str, list[dict | None]] = {}
        self._ppm_cache: dict[str, float] = {}
        self._subject_by_session: dict[str, str] = {}

    # -- setup ---------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._manifest = self._imports.load_manifest(project_root)
        self._pose_cache.clear()
        self._roi_cache.clear()
        self._ppm_cache.clear()
        self._subject_by_session.clear()
        cfg = read_yaml(project_root / "project.yaml", {})
        try:
            self._fps = float(cfg.get("default_fps", 30.0)) or 30.0
        except Exception:
            self._fps = 30.0
        try:
            self._roi_count = max(1, int(self._rois.get_roi_count(project_root)))
        except Exception:
            self._roi_count = 1
        self._metric_defs = build_metric_defs(self._roi_count)
        self._metric_ids = [m.id for m in self._metric_defs]
        if self._manifest is not None:
            for s in self._manifest.linked_sessions:
                self._subject_by_session[str(s.session_id)] = str(s.subject_id or "")

    def load_segment_pool(self, session_ids: set[str] | None = None) -> list[ClipRef]:
        """Return every feature-extraction segment as a mineable :class:`ClipRef`.

        This is the *full* candidate pool — all windows scored during feature
        extraction (``derived/representations/segment_features.parquet``), not
        just the clips already loaded into the review queue.  ``window_id`` is the
        ``segment_id`` (they are the same for segment-derived windows), so matches
        can be re-extracted and injected back into review.  Pass ``session_ids``
        to restrict the pool to particular sessions.
        """
        if self._project_root is None:
            return []
        path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(
                path, columns=["segment_id", "session_id", "start_frame", "end_frame"]
            )
        except Exception as exc:  # pragma: no cover - corrupt/locked cache
            logger.warning("Clip mining: could not read segment pool (%s)", exc)
            return []
        keep = {str(s) for s in session_ids} if session_ids else None
        out: list[ClipRef] = []
        for sid, seg, s, e in zip(
            df["session_id"].astype(str),
            df["segment_id"].astype(str),
            df["start_frame"],
            df["end_frame"],
        ):
            if keep is not None and sid not in keep:
                continue
            out.append(ClipRef(seg, sid, int(s), int(e)))
        return out

    def available_metrics(self) -> list[MetricDef]:
        """Metric list for the active project (one ROI block per configured zone)."""
        return list(self._metric_defs)

    def metric_groups(self) -> list[str]:
        seen: list[str] = []
        for m in self._metric_defs:
            if m.group not in seen:
                seen.append(m.group)
        return seen

    # -- criteria persistence (saved with the project) -----------------------

    def save_criteria(self, criteria: list["Criterion"], match_all: bool) -> None:
        """Persist the mining criteria + match mode to the project.

        Only criteria whose metric exists in *this* project's registry are
        written, so a stale row (a metric later removed, or a zone the project no
        longer has) is dropped rather than resurrected on the next load.
        """
        if self._project_root is None:
            return
        known = set(self._metric_ids)
        payload = {
            "match_all": bool(match_all),
            "criteria": [
                {
                    "metric_id": c.metric_id,
                    "low": None if c.low is None else float(c.low),
                    "high": None if c.high is None else float(c.high),
                    "enabled": bool(c.enabled),
                }
                for c in criteria
                if c.metric_id in known
            ],
        }
        try:
            write_yaml(self._project_root / self.CRITERIA_FILE, payload)
        except Exception as exc:  # pragma: no cover - disk/permission issues
            logger.warning("Clip mining: could not save criteria (%s)", exc)

    def load_criteria(self) -> tuple[list["Criterion"], bool | None]:
        """Return the project's saved ``(criteria, match_all)``.

        Criteria referencing a metric this project no longer exposes (removed
        feature, or a zone beyond the current ROI count) are silently skipped so
        the dialog never offers a control that can't be evaluated.  ``match_all``
        is ``None`` when nothing has been saved yet.
        """
        if self._project_root is None:
            return [], None
        raw = read_yaml(self._project_root / self.CRITERIA_FILE, {})
        if not isinstance(raw, dict):
            return [], None
        known = set(self._metric_ids)

        def _num(v: object) -> float | None:
            try:
                return None if v is None else float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        out: list[Criterion] = []
        for item in raw.get("criteria", []) or []:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("metric_id", ""))
            if mid not in known:
                continue
            out.append(
                Criterion(
                    metric_id=mid,
                    low=_num(item.get("low")),
                    high=_num(item.get("high")),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        ma = raw.get("match_all", None)
        match_all = None if ma is None else bool(ma)
        return out, match_all

    def ready(self) -> bool:
        """True when a project with a usable target ROI is loaded."""
        return self._project_root is not None and self._manifest is not None

    # -- per-session data ----------------------------------------------------

    def _pose_for(self, session_id: str):
        if session_id in self._pose_cache:
            return self._pose_cache[session_id]
        pose = None
        try:
            path = self._imports.pose_path_for_session(self._manifest, session_id)
            if path and Path(path).exists():
                pose = self._pose_svc.load(path)
                pose = self._pose_svc.clean_pose(
                    pose, likelihood_threshold=0.2, interpolate=True, smoothing_window=5
                )
        except Exception as exc:
            logger.warning("Clip metrics: could not load pose for %s (%s)", session_id, exc)
            pose = None
        self._pose_cache[session_id] = pose
        return pose

    def _rois_for(self, session_id: str) -> list[dict | None]:
        """Return every configured target zone for this session (``None`` = no area).

        The list is positional: index ``i`` is ROI ``i + 1``, matching the
        ``__roi{N}`` metric-id scheme, so a zone the subject never drew still
        occupies its slot (as ``None``) rather than shifting the others.
        """
        if session_id in self._roi_cache:
            return self._roi_cache[session_id]
        rois: list[dict | None] = []
        if self._project_root is not None:
            subject = self._subject_by_session.get(session_id, "")
            key = f"{subject}::{session_id}" if subject else session_id
            for candidate in self._rois.resolve_target_rois(self._project_root, key):
                rois.append(candidate if (candidate and roi_has_area(candidate)) else None)
        self._roi_cache[session_id] = rois
        return rois

    def _ppm_for(self, session_id: str) -> float:
        if session_id in self._ppm_cache:
            return self._ppm_cache[session_id]
        ppm = 1.0
        try:
            val = self._imports.pixels_per_mm_for_session(self._manifest, session_id)
            if val and float(val) > 0:
                ppm = float(val)
        except Exception:
            ppm = 1.0
        self._ppm_cache[session_id] = ppm
        return ppm

    # -- metric computation --------------------------------------------------

    @staticmethod
    def _part(pose, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        if name not in getattr(pose, "body_parts", []):
            return None
        return (np.asarray(pose.x[name], dtype=float), np.asarray(pose.y[name], dtype=float))

    def _metrics_for_clip(self, ref: ClipRef) -> dict[str, float]:
        out: dict[str, float] = {m: float("nan") for m in self._metric_ids}
        pose = self._pose_for(ref.session_id)
        if pose is None:
            return out
        n = len(pose.x)
        s = max(0, int(ref.start_frame))
        e = min(n - 1, int(ref.end_frame))
        if e < s:
            return out
        ppm = self._ppm_for(ref.session_id)
        fps = self._fps
        sl = slice(s, e + 1)
        nfr = e - s + 1

        out["duration_sec"] = nfr / fps if fps else float(nfr)

        nose = self._part(pose, "nose")
        tail = self._part(pose, "tail_base")
        rois = self._rois_for(ref.session_id)

        # Centroid over whatever parts exist (mean of tracked keypoints); kept
        # around so the per-zone loop below can reuse it for occupancy/proximity.
        cx = cy = None
        parts = [self._part(pose, p) for p in pose.body_parts]
        parts = [p for p in parts if p is not None]
        if parts:
            cx = np.nanmean(np.stack([p[0][sl] for p in parts], axis=0), axis=0)
            cy = np.nanmean(np.stack([p[1][sl] for p in parts], axis=0), axis=0)
            csp = self._speed_mm_s(cx, cy, ppm, fps)
            if csp.size:
                out["centroid_speed_mean"] = float(np.nanmean(csp))
                out["centroid_speed_max"] = float(np.nanmax(csp))
                out["centroid_speed_std"] = float(np.nanstd(csp))
                out["immobile_frac"] = float(np.nanmean((csp < IMMOBILE_MM_S).astype(float)))
                if csp.size >= 2:
                    acc = np.abs(np.diff(csp)) * fps  # |Δspeed| per second
                    if np.isfinite(acc).any():
                        out["centroid_accel_mean"] = float(np.nanmean(acc))
            path_mm = float("nan")
            if cx.size >= 2:
                step = np.hypot(np.diff(cx), np.diff(cy))
                if np.isfinite(step).any():
                    path_mm = float(np.nansum(step) / ppm)
                    out["centroid_path_mm"] = path_mm
            # Net displacement (first→last tracked point) + tortuosity.
            fin = np.isfinite(cx) & np.isfinite(cy)
            fidx = np.nonzero(fin)[0]
            if fidx.size >= 2:
                disp = float(
                    np.hypot(cx[fidx[-1]] - cx[fidx[0]], cy[fidx[-1]] - cy[fidx[0]]) / ppm
                )
                out["centroid_displacement_mm"] = disp
                if np.isfinite(path_mm) and disp > 1.0:  # undefined for a near-stationary clip
                    out["path_tortuosity"] = path_mm / disp
            # Spatial extent explored.
            if np.isfinite(cx).any():
                rx = (np.nanmax(cx) - np.nanmin(cx)) / ppm
                ry = (np.nanmax(cy) - np.nanmin(cy)) / ppm
                out["centroid_travel_mm"] = float(np.hypot(rx, ry))
                out["explore_area_mm2"] = float(rx * ry)
            # Heading changes (turning): direction of centroid velocity, ignoring
            # jitter frames where it barely moved so noise doesn't inflate turning.
            if cx.size >= 3:
                vx, vy = np.diff(cx), np.diff(cy)
                step_mm = np.hypot(vx, vy) / ppm
                head = np.arctan2(vy, vx)
                head[step_mm < 0.2] = np.nan
                dh = np.diff(head)
                dh = (dh + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
                adh = np.abs(np.degrees(dh))
                if np.isfinite(adh).any():
                    out["turning_total_deg"] = float(np.nansum(adh))
                    out["turning_rate_mean"] = float(np.nanmean(adh) * fps)

        if nose is not None:
            nx, ny = nose[0][sl], nose[1][sl]
            nsp = self._speed_mm_s(nx, ny, ppm, fps)
            if nsp.size:
                out["nose_speed_mean"] = float(np.nanmean(nsp))
                out["nose_speed_max"] = float(np.nanmax(nsp))
                out["nose_speed_std"] = float(np.nanstd(nsp))
            if np.isfinite(nx).any():
                dx = np.nanmax(nx) - np.nanmin(nx)
                dy = np.nanmax(ny) - np.nanmin(ny)
                out["nose_travel_mm"] = float(np.hypot(dx, dy) / ppm)

        if tail is not None:
            tx, ty = tail[0][sl], tail[1][sl]
            tsp = self._speed_mm_s(tx, ty, ppm, fps)
            if tsp.size:
                out["tail_speed_mean"] = float(np.nanmean(tsp))

        # Per-ROI geometry: one metric set per configured target zone.  ROI 1
        # uses the bare metric ids; later zones use the ``__roi{N}`` variants.
        for i, roi in enumerate(rois):
            if roi is None:
                continue
            if cx is not None:
                cin = roi_contains(roi, cx, cy).astype(bool)
                if cin.size:
                    out[roi_metric_id("centroid_in_roi_frac", i)] = float(
                        np.nanmean(cin.astype(float))
                    )
                    # Entries = outside→inside transitions (number of visits).
                    if cin.size >= 2:
                        entries = int(np.count_nonzero((~cin[:-1]) & cin[1:]))
                        out[roi_metric_id("roi_entry_count", i)] = float(entries)
                csd = roi_signed_distance(roi, cx, cy)  # + inside, - outside
                if np.isfinite(csd).any():
                    # Report distance TO the zone: negative inside, positive outside.
                    dcen = -csd / ppm
                    out[roi_metric_id("centroid_dist_to_roi_mm", i)] = float(np.nanmean(dcen))
                    out[roi_metric_id("centroid_dist_to_roi_min_mm", i)] = float(
                        np.nanmin(dcen)
                    )
            if nose is not None:
                nx, ny = nose[0][sl], nose[1][sl]
                sd = roi_signed_distance(roi, nx, ny)  # + inside, - outside
                if np.isfinite(sd).any():
                    out[roi_metric_id("nose_off_edge_mm", i)] = float(-np.nanmin(sd) / ppm)
                    out[roi_metric_id("nose_dist_to_roi_mm", i)] = float(-np.nanmean(sd) / ppm)
                    out[roi_metric_id("nose_in_roi_frac", i)] = float(
                        np.nanmean(roi_contains(roi, nx, ny).astype(float))
                    )
            if tail is not None:
                tx, ty = tail[0][sl], tail[1][sl]
                out[roi_metric_id("tail_in_roi_frac", i)] = float(
                    np.nanmean(roi_contains(roi, tx, ty).astype(float))
                )

        if nose is not None and tail is not None:
            nx, ny = nose[0][sl], nose[1][sl]
            tx, ty = tail[0][sl], tail[1][sl]
            blen = np.hypot(nx - tx, ny - ty) / ppm
            if np.isfinite(blen).any():
                med = float(np.nanmedian(blen))
                out["body_length_mean"] = float(np.nanmean(blen))
                out["body_length_max"] = float(np.nanmax(blen))
                out["body_length_range_mm"] = float(np.nanmax(blen) - np.nanmin(blen))
                out["body_stretch_ratio"] = float(np.nanmax(blen) / med) if med > 1e-6 else float("nan")

        return out

    @staticmethod
    def _speed_mm_s(x: np.ndarray, y: np.ndarray, ppm: float, fps: float) -> np.ndarray:
        if x.size < 2:
            return np.asarray([], dtype=float)
        d = np.hypot(np.diff(x), np.diff(y)) / max(ppm, 1e-9) * max(fps, 1e-9)
        return d

    # Below this many clips the process-spawn cost (each worker re-imports the
    # scoring module) outweighs any speedup, so scoring stays single-threaded.
    _PARALLEL_MIN_CLIPS = 600
    # When there are fewer sessions than workers a big session is split into
    # sub-chunks to fill the remaining cores, but each split reloads that
    # session's pose — only worth it once a chunk carries at least this many
    # clips, so small sessions are never over-split into a net loss.
    _MIN_SPLIT_CHUNK = 1500

    def compute(
        self,
        clips: list[ClipRef],
        progress_callback: Callable[[int, int], None] | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by ``window_id`` with one column per metric.

        Clips are grouped by session so each session's pose is loaded once.  For
        a large pool the sessions are scored in parallel across worker processes
        (see :meth:`_compute_parallel`); small pools, single-core machines, and
        ``max_workers=1`` use the in-process serial path.  Both paths produce an
        identical table (rows in input order), so the parallel path is a pure
        speedup.  ``max_workers`` caps the pool (``None`` = one per CPU).
        """
        if not clips:
            return pd.DataFrame(columns=self._metric_ids)
        by_session: dict[str, list[ClipRef]] = {}
        for c in clips:
            by_session.setdefault(c.session_id, []).append(c)

        want_workers = os.cpu_count() or 1 if max_workers is None else int(max_workers)
        rows: dict[str, dict[str, float]] | None = None
        if (
            want_workers > 1
            and len(clips) >= self._PARALLEL_MIN_CLIPS
            and self._project_root is not None
        ):
            chunks = self._plan_chunks(by_session, want_workers)
            # Only pay the process-pool cost when it actually buys parallelism —
            # a single chunk (one small session) is faster scored in-process.
            if len(chunks) >= 2:
                try:
                    rows = self._compute_parallel(
                        clips, chunks, want_workers, progress_callback
                    )
                except Exception as exc:  # pragma: no cover - pool/pickle/env failure
                    logger.warning(
                        "Clip mining: parallel scoring failed (%s); using serial path",
                        exc,
                    )
                    rows = None
        if rows is None:
            rows = self._compute_serial(by_session, len(clips), progress_callback)

        df = pd.DataFrame.from_dict(rows, orient="index")
        # Reindex to the input order (and full column set) so serial and parallel
        # runs are byte-for-byte identical regardless of task-completion order.
        return df.reindex(index=[c.window_id for c in clips], columns=self._metric_ids)

    def _compute_serial(
        self,
        by_session: dict[str, list[ClipRef]],
        total: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[str, dict[str, float]]:
        """Score every clip in-process, one session at a time (pose loaded once)."""
        rows: dict[str, dict[str, float]] = {}
        done = 0
        for sid, refs in by_session.items():
            for ref in refs:
                rows[ref.window_id] = self._metrics_for_clip(ref)
                done += 1
            if progress_callback:
                progress_callback(min(done, total), total)
            # Free this session's pose once its clips are scored (bounded memory).
            self._pose_cache.pop(sid, None)
        return rows

    def _compute_parallel(
        self,
        clips: list[ClipRef],
        chunks: list[list[ClipRef]],
        want_workers: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[str, dict[str, float]]:
        """Score the pool across worker processes, one task per session-chunk.

        Each worker rebuilds a fresh service from the project root (its caches and
        manifest can't cross the process boundary), loads only its own sessions'
        pose, and returns just the small per-window metric dicts.  Progress is
        reported per completed chunk.
        """
        n_workers = min(want_workers, len(chunks))
        root = str(self._project_root)
        total = len(clips)
        rows: dict[str, dict[str, float]] = {}
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_score_clips_task, root, chunk): len(chunk)
                for chunk in chunks
            }
            for fut in as_completed(futures):
                rows.update(fut.result())
                done += futures[fut]
                if progress_callback:
                    progress_callback(min(done, total), total)
        return rows

    @staticmethod
    def _plan_chunks(
        by_session: dict[str, list[ClipRef]], want_workers: int
    ) -> list[list[ClipRef]]:
        """Split the pool into worker tasks, keeping each session's pose load once.

        With at least ``want_workers`` sessions, each session is its own task —
        the ideal case, since pose is loaded exactly once per task and completion
        order balances the load.  With fewer (but larger) sessions, big sessions
        are split into contiguous sub-chunks — proportional to their size — so all
        cores stay busy, at the cost of reloading that session's pose per chunk.
        A session is only split while each resulting chunk still carries at least
        ``_MIN_SPLIT_CHUNK`` clips, so a small session stays whole (splitting it
        would reload its pose for too little recomputed work to pay for it).
        """
        sessions = list(by_session.values())
        if len(sessions) >= want_workers:
            return sessions
        total = sum(len(r) for r in sessions) or 1
        chunks: list[list[ClipRef]] = []
        for refs in sessions:
            proportional = round(want_workers * len(refs) / total)
            split_cap = max(1, len(refs) // ClipMetricsService._MIN_SPLIT_CHUNK)
            k = max(1, min(len(refs), proportional, split_cap))
            chunks.extend(_split_even(refs, k))
        return chunks

    # -- mining --------------------------------------------------------------

    @staticmethod
    def mine(df: pd.DataFrame, criteria: list[Criterion], match_all: bool = True) -> MiningResult:
        """Return clips whose metrics satisfy the active criteria.

        ``match_all`` True → a clip must pass every enabled criterion (AND);
        False → it needs at least one (OR).  ``scores`` is the fraction of
        enabled criteria each matched clip passes (always 1.0 in AND mode), used
        by the UI to rank near-misses.
        """
        active = [c for c in criteria if c.enabled and c.metric_id in df.columns and (c.low is not None or c.high is not None)]
        result = MiningResult(n_evaluated=int(len(df)))
        if df.empty or not active:
            return result

        n = len(df)
        pass_counts = np.zeros(n, dtype=float)
        eligible = np.zeros(n, dtype=bool)
        for c in active:
            col = pd.to_numeric(df[c.metric_id], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(col)
            passed = ok.copy()
            if c.low is not None:
                passed &= col >= float(c.low)
            if c.high is not None:
                passed &= col <= float(c.high)
            pass_counts += passed.astype(float)
            eligible |= ok

        n_active = len(active)
        if match_all:
            keep = pass_counts >= n_active
        else:
            keep = pass_counts >= 1.0

        ids = df.index.to_numpy()
        for i in np.nonzero(keep)[0]:
            wid = str(ids[i])
            result.matched_ids.append(wid)
            result.scores[wid] = float(pass_counts[i] / max(1, n_active))
        # Rank strongest matches first (mainly meaningful in OR mode).
        result.matched_ids.sort(key=lambda w: result.scores.get(w, 0.0), reverse=True)
        return result

    @staticmethod
    def check_essence(
        df: pd.DataFrame, criteria: list[Criterion], match_all: bool = True
    ) -> EssenceCheckResult:
        """Split an *already-chosen* clip set into inside/outside the essence ranges.

        The inverse of :meth:`mine`.  Where ``mine`` hunts the full pool for clips
        that satisfy the criteria, this audits the clips already in ``df`` and
        reports which ones fall *outside* the acceptable ranges (fail the essence
        test) and by how much.  It is used to re-flag reviewed clips after the
        operational definition of a behavior is tightened.

        ``match_all`` True → a clip fails if it violates *any* criterion it has
        data for; False → it fails only when it is outside *every* criterion.
        Each violated criterion contributes a readable reason string.  Clips with
        no finite value for any active criterion cannot be judged and land in
        ``no_data_ids`` (never flagged).
        """
        active = [
            c
            for c in criteria
            if c.enabled and c.metric_id in df.columns and (c.low is not None or c.high is not None)
        ]
        result = EssenceCheckResult(n_evaluated=int(len(df)))
        if df.empty or not active:
            # Nothing to test against — treat every clip as passing.
            result.passed_ids = [str(w) for w in df.index]
            return result

        for wid in df.index:
            key = str(wid)
            row = df.loc[wid]
            n_have = 0
            n_pass = 0
            viols: list[str] = []
            for c in active:
                try:
                    val = float(row[c.metric_id])
                except (TypeError, ValueError):
                    continue  # non-numeric / missing
                if not np.isfinite(val):
                    continue  # no data for this metric on this clip
                n_have += 1
                below = c.low is not None and val < float(c.low)
                above = c.high is not None and val > float(c.high)
                if below or above:
                    viols.append(_format_violation(c, float(val), below))
                else:
                    n_pass += 1
            if n_have == 0:
                result.no_data_ids.append(key)
                continue
            # Judge only on the metrics the clip actually has data for, so a clip
            # missing one metric isn't flagged solely for the gap.
            if match_all:
                passed = n_pass == n_have
            else:
                passed = n_pass >= 1
            if passed:
                result.passed_ids.append(key)
            else:
                result.failed_ids.append(key)
                result.violations[key] = viols
        return result

    # -- essence extraction --------------------------------------------------

    @staticmethod
    def extract_essence(
        df: pd.DataFrame,
        exemplar_ids: list[str],
        metric_ids: list[str],
        method: str = "minmax",
        pad_frac: float = 0.10,
    ) -> list[Criterion]:
        """Infer criteria from exemplar clips: the shared value range per metric.

        ``minmax`` uses the exemplars' full span; ``p10p90`` uses a robust
        10–90th-percentile span (better with an outlier exemplar).  Each range is
        padded by ``pad_frac`` of its width so the exemplars sit comfortably
        inside, not exactly on the boundary.  Metrics that are all-NaN across the
        exemplars are skipped.
        """
        present = [w for w in exemplar_ids if w in df.index]
        out: list[Criterion] = []
        if not present:
            return out
        sub = df.loc[present]
        for mid in metric_ids:
            if mid not in sub.columns:
                continue
            vals = pd.to_numeric(sub[mid], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if method == "p10p90" and vals.size >= 3:
                lo = float(np.percentile(vals, 10))
                hi = float(np.percentile(vals, 90))
            else:
                lo = float(np.min(vals))
                hi = float(np.max(vals))
            width = hi - lo
            pad = pad_frac * width if width > 0 else pad_frac * (abs(hi) + 1e-6)
            out.append(Criterion(metric_id=mid, low=round(lo - pad, 3), high=round(hi + pad, 3), enabled=True))
        return out

    @staticmethod
    def rank_shared_features(
        exemplar_df: pd.DataFrame,
        population_df: pd.DataFrame | None,
        min_finite_frac: float = 0.75,
    ) -> list[tuple[str, float]]:
        """Rank metrics by how *alike* the exemplars are on them (tightest first).

        The signature of a hand-picked clip set is the features on which those
        clips agree, made comparable across units by normalising each feature's
        exemplar spread (robust 10–90th-pct span, or min–max for very small sets):

        * **With a population baseline** (``population_df`` — e.g. the pool once
          "Find matches" has scored it) each spread is divided by that feature's
          spread across the pool, so a low ratio means the exemplars are consistent
          *and* the feature is discriminative.  This needs no full scan by the
          caller — it just reuses a baseline if one is already on hand.
        * **Without one** (``population_df is None``) it works straight off the
          highlighted clips: each spread is divided by the exemplars' own median
          magnitude (a scale-free relative spread / coefficient-of-variation), so
          essence extraction stays instant and never forces a population scan.

        Returns ``[(metric_id, normalised_spread), …]`` ascending (best first).
        Metrics without enough finite exemplar data — or, in population mode, whose
        pool spread is degenerate — are skipped.
        """
        out: list[tuple[str, float]] = []
        if exemplar_df is None or exemplar_df.empty:
            return out
        n_ex = int(len(exemplar_df))
        need = max(2, int(np.ceil(min_finite_frac * n_ex)))
        use_population = population_df is not None
        for mid in exemplar_df.columns:
            ex = pd.to_numeric(exemplar_df[mid], errors="coerce").to_numpy(dtype=float)
            ex = ex[np.isfinite(ex)]
            if ex.size < need:
                continue
            ex_spread = (
                float(np.percentile(ex, 90) - np.percentile(ex, 10))
                if ex.size >= 5
                else float(ex.max() - ex.min())
            )
            if use_population:
                pop_scale: float | None = None
                if mid in population_df.columns:
                    pop = pd.to_numeric(population_df[mid], errors="coerce").to_numpy(dtype=float)
                    pop = pop[np.isfinite(pop)]
                    if pop.size >= 4:
                        pop_scale = float(np.percentile(pop, 90) - np.percentile(pop, 10))
                if pop_scale is None or pop_scale <= 1e-9:
                    # No usable population scale → can't tell a signature feature
                    # from a naturally-constant one, so skip it rather than reward.
                    continue
                out.append((mid, ex_spread / pop_scale))
            else:
                # Scale-free relative spread — units cancel, so a fraction and a
                # mm/s metric are comparable without any population reference.
                scale = max(abs(float(np.median(ex))), 1e-6)
                out.append((mid, ex_spread / scale))
        out.sort(key=lambda t: t[1])
        return out

    @classmethod
    def extract_similar_essence(
        cls,
        exemplar_df: pd.DataFrame,
        population_df: pd.DataFrame | None,
        k: int = 3,
        method: str = "minmax",
        pad_frac: float = 0.10,
    ) -> list[Criterion]:
        """Discover the *k* features the exemplars most agree on and range them.

        Combines :meth:`rank_shared_features` (which features define this clip set)
        with :meth:`extract_essence` (their padded value ranges), so the caller
        gets ready-to-mine criteria for the top-*k* shared features without the
        user having to pick which features to look at.
        """
        ranked = cls.rank_shared_features(exemplar_df, population_df)
        top = [mid for mid, _score in ranked[: max(1, int(k))]]
        if not top:
            return []
        return cls.extract_essence(
            exemplar_df, list(exemplar_df.index), top, method=method, pad_frac=pad_frac
        )
