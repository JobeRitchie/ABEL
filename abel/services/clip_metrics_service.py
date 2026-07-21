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

**Two metric spaces.**  The interpretable metrics above are what a human sets by
hand ("nose past edge > 10 mm") and are the only ones offered in the criteria
search box.  Essence extraction — which picks its own features from exemplar
clips — additionally draws on the *shipped* per-window features the classifier
itself is trained on (``derived/representations/segment_features.parquet``:
~1200 columns of oscillation / periodicity / angular-velocity / jerk / context
features).  Those are machine-named, so they are poor manual controls, but they
are exactly where the signal for behaviours like wet-dog-shake lives, and they
are precomputed — so essence over them needs no pose recompute and no mounted
pose drive.  Rich ids are namespaced ``feat:<column>`` so they never collide
with an interpretable metric id and can be told apart anywhere.

Pipeline position:
    Pose Features + ROI geometry → **Clip Metrics / Mining** ← here
    Feature Extraction (segment_features) ↗
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

# Metrics that carry no essence signal and must never be chosen as a criterion:
# ``duration_sec`` is constant across the fixed-window segment pool (every segment
# is the same length), so it "agrees" perfectly across any exemplar set yet
# discriminates nothing — exactly the kind of degenerate feature the old
# tightest-spread ranking used to surface.
_ESSENCE_SKIP_METRICS: frozenset[str] = frozenset({"duration_sec"})

# How many candidate features the greedy essence search may consider, taken
# best-separated-first.  The search is O(features) per round, so letting it range
# over the full ~1100-column shipped feature space would take tens of seconds;
# the box only ever commits ~5–8 features anyway, and piloting on real data put
# the capped result within noise of the uncapped one.  Ignored (a no-op) for the
# ~30 interpretable clip metrics.
ESSENCE_MAX_FEATURES = 40


# ---------------------------------------------------------------------------
# Rich (shipped) feature space
# ---------------------------------------------------------------------------
# The per-window features produced by Feature Extraction — the very vectors the
# classifier, active learning and UMAP consume.  Essence extraction can range
# over them as well as over the interpretable clip metrics above; they are
# namespaced so a rich id is always recognisable and can never collide with a
# clip-metric id (``nose_speed_mean`` exists in both spaces, in different units).

RICH_PREFIX = "feat:"
RICH_FEATURES_FILE = Path("derived") / "representations" / "segment_features.parquet"

# Identity / bookkeeping columns, plus the model's own outputs.  The latter are
# excluded deliberately: an essence built on ``prediction_prob`` would describe
# what the current model already believes, not what the animal is doing.
_RICH_META_COLS: frozenset[str] = frozenset({
    "segment_id", "start_frame", "end_frame", "animal_id", "session_id",
    "label", "label_source", "reviewer_confidence", "overlap_allowed",
    "uncertainty_entropy", "uncertainty_margin", "density_outlier_score",
    "uncertainty_score", "prediction_prob", "prediction_prob_fused",
    "prediction_variance",
})

# Trailing summary-statistic suffixes, rendered as a parenthetical so
# ``ear_right_acceleration_median`` reads as "Ear right acceleration (median)".
_RICH_STAT_SUFFIXES: dict[str, str] = {
    "mean": "mean", "std": "variability", "median": "median", "max": "max",
    "min": "min", "p10": "10th pct", "p90": "90th pct", "energy": "energy",
    "periodicity": "periodicity", "range": "range", "sum": "total",
    "delta": "change", "rate": "rate", "frac": "fraction", "count": "count",
    "var": "variance", "skew": "skew", "kurtosis": "kurtosis",
    "entropy": "entropy", "slope": "slope", "iqr": "spread", "cv": "rel. spread",
}

# Coarse family → group label, so the criteria list says *what kind* of feature
# the essence locked onto rather than a flat "Feature" for all 1200 of them.
# First match wins, so the order is meaningful (rhythm before kinematics).
_RICH_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("oscillation", "periodicity", "frequency", "autocorr", "rhythm", "energy"),
     "Feature · Rhythm"),
    (("angular", "orientation", "heading", "rotation", "turn", "angle"),
     "Feature · Rotation"),
    (("accel", "jerk", "velocity", "speed", "movement", "displacement"),
     "Feature · Kinematics"),
    (("dist_", "length", "height", "area", "elong", "curvature", "posture"),
     "Feature · Posture"),
    (("roi", "zone", "wall", "center", "centre", "corner", "arena", "context",
      "social", "partner"), "Feature · Context"),
    (("video", "pixel", "intensity", "optical", "texture"), "Feature · Video"),
)
_RICH_GROUP_DEFAULT = "Feature · Other"


def is_rich_metric(metric_id: str) -> bool:
    """True for a namespaced shipped-feature id (``feat:<column>``)."""
    return str(metric_id).startswith(RICH_PREFIX)


def rich_metric_id(column: str) -> str:
    """Namespaced metric id for a ``segment_features`` column."""
    return f"{RICH_PREFIX}{column}"


def rich_column(metric_id: str) -> str:
    """The underlying parquet column behind a namespaced rich metric id."""
    mid = str(metric_id)
    return mid[len(RICH_PREFIX):] if mid.startswith(RICH_PREFIX) else mid


def rich_metric_def(metric_id: str) -> MetricDef:
    """Build a display definition for a rich feature id on the fly.

    Rich columns are machine-named, so the label is humanised (underscores to
    spaces, sentence case, trailing summary statistic moved into parentheses)
    and the group names the feature *family*.  The raw column name is kept in the
    description so a user can still trace a criterion back to the feature table.
    """
    col = rich_column(metric_id)
    stem, _, tail = col.rpartition("_")
    if stem and tail in _RICH_STAT_SUFFIXES:
        base, stat = stem, _RICH_STAT_SUFFIXES[tail]
    else:
        base, stat = col, ""
    label = base.replace("_", " ").strip()
    label = (label[:1].upper() + label[1:]) if label else col
    if stat:
        label = f"{label} ({stat})"
    low = col.lower()
    group = next(
        (g for keys, g in _RICH_GROUPS if any(k in low for k in keys)),
        _RICH_GROUP_DEFAULT,
    )
    return MetricDef(
        id=rich_metric_id(col),
        label=label,
        group=group,
        unit="",
        description=(
            f"Extracted feature '{col}' from ABEL's feature extraction — one of "
            "the same per-window features the classifier is trained on. Units are "
            "the extractor's internal ones, so compare it against the scope range "
            "rather than reading it as mm or mm/s."
        ),
    )


def metric_def_for(metric_id: str) -> MetricDef | None:
    """Definition for any criterion id — interpretable metric or rich feature."""
    m = METRIC_BY_ID.get(metric_id)
    if m is not None:
        return m
    return rich_metric_def(metric_id) if is_rich_metric(metric_id) else None


def metric_label(metric_id: str) -> str:
    """Human-readable name for any criterion id (falls back to the id itself)."""
    m = metric_def_for(metric_id)
    return m.label if m else str(metric_id)


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
class EssenceFrames:
    """The exemplar/background tables essence ranges over, plus how they were built.

    ``sources`` names the metric spaces that contributed ("pose metrics",
    "extracted features") and ``notes`` carries any user-facing reason a space was
    left out, so the UI can say *why* an essence is narrower than it could be
    instead of silently returning a weaker definition.
    """

    exemplars: pd.DataFrame
    background: pd.DataFrame
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def usable(self) -> bool:
        return not self.exemplars.empty and not self.background.empty


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
    m = metric_def_for(c.metric_id)
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


def _has_data(df: "pd.DataFrame | None") -> bool:
    """True when *df* exists and holds at least one finite value."""
    return df is not None and not df.empty and bool(df.notna().to_numpy().any())


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
        # Namespaced ids of the project's shipped features (schema-only read),
        # cached per project; None = not looked up yet.
        self._rich_ids: list[str] | None = None

    # -- setup ---------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._manifest = self._imports.load_manifest(project_root)
        self._pose_cache.clear()
        self._roi_cache.clear()
        self._ppm_cache.clear()
        self._subject_by_session.clear()
        self._rich_ids = None
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

    def unresolved_pose_clips(self, clips: list[ClipRef]) -> dict[str, str | None]:
        """Return sessions whose raw pose file can't be read, keyed by session id.

        Every metric is recomputed from each session's *raw* pose at the path
        recorded in the project manifest.  If that file has moved or its drive is
        unmounted, the clip scores all-NaN — which downstream looks exactly like
        "no shared features".  Callers use this to warn the user explicitly
        rather than fail silently.  The value is the unreachable path (``None`` if
        the manifest has no pose entry for the session at all).  Sessions are
        de-duplicated, so passing the whole pool is only one check per session.
        """
        out: dict[str, str | None] = {}
        if self._manifest is None:
            return out
        for sid in {c.session_id for c in clips}:
            try:
                path = self._imports.pose_path_for_session(self._manifest, sid)
            except Exception:
                path = None
            if not path or not Path(path).exists():
                out[sid] = str(path) if path else None
        return out

    # -- rich (shipped) features ---------------------------------------------

    # Columns are read in blocks so a wide feature table never materialises in
    # full: 1200 columns × 40k windows is ~440 MB, one block is a fortieth of it.
    _RICH_COL_BLOCK = 250

    def rich_features_path(self) -> Path | None:
        """Path to the project's shipped per-window feature table (may not exist)."""
        if self._project_root is None:
            return None
        return self._project_root / RICH_FEATURES_FILE

    def rich_feature_ids(self) -> list[str]:
        """Namespaced ids of every mineable shipped feature (``[]`` when absent).

        Read from the parquet *schema* only, so this costs no data read and can be
        called freely (e.g. to validate saved criteria on load).
        """
        if self._rich_ids is not None:
            return self._rich_ids
        path = self.rich_features_path()
        self._rich_ids = []
        if path is not None and path.exists():
            try:
                import pyarrow.parquet as pq

                names = pq.read_schema(path).names
                self._rich_ids = [
                    rich_metric_id(c) for c in names if c not in _RICH_META_COLS
                ]
            except Exception as exc:  # pragma: no cover - corrupt/locked cache
                logger.warning("Clip mining: could not read feature schema (%s)", exc)
        return self._rich_ids

    def has_rich_features(self) -> bool:
        """True when this project has a usable shipped feature table to mine."""
        return bool(self.rich_feature_ids())

    def load_rich_features(
        self,
        metric_ids: list[str] | None = None,
        segment_ids: "set[str] | list[str] | None" = None,
    ) -> pd.DataFrame:
        """Shipped per-window features, indexed by segment id, ``feat:``-namespaced.

        ``metric_ids`` restricts the columns (the usual case once essence has
        chosen a handful); ``segment_ids`` restricts the rows.  Values are read in
        column blocks and downcast to float32, so pulling every feature for a
        background sample costs a fraction of the full table's memory.  Returns an
        empty frame when the project has no feature table.
        """
        path = self.rich_features_path()
        if path is None or not path.exists():
            return pd.DataFrame()
        known = {rich_column(m) for m in self.rich_feature_ids()}
        cols = (
            [rich_column(m) for m in metric_ids if rich_column(m) in known]
            if metric_ids is not None
            else sorted(known, key=lambda c: c)
        )
        if not cols:
            return pd.DataFrame()
        try:
            index = pd.read_parquet(path, columns=["segment_id"])["segment_id"].astype(str)
            keep = (
                index.isin({str(s) for s in segment_ids}).to_numpy()
                if segment_ids is not None
                else np.ones(len(index), dtype=bool)
            )
            blocks: list[pd.DataFrame] = []
            for i in range(0, len(cols), self._RICH_COL_BLOCK):
                part = pd.read_parquet(path, columns=cols[i : i + self._RICH_COL_BLOCK])
                blocks.append(
                    part.loc[keep].apply(pd.to_numeric, errors="coerce").astype("float32")
                )
                del part
            df = pd.concat(blocks, axis=1) if blocks else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - corrupt/locked cache
            logger.warning("Clip mining: could not read shipped features (%s)", exc)
            return pd.DataFrame()
        df.index = index[keep].to_numpy()
        df.columns = [rich_metric_id(c) for c in df.columns]
        return df

    def attach_rich_columns(
        self, df: pd.DataFrame, metric_ids: list[str]
    ) -> pd.DataFrame:
        """Return *df* with the named shipped-feature columns joined on.

        Essence may commit criteria on features the scored clip-metric table has no
        column for; this pulls just those few columns (for the rows already in
        ``df``) so mining can evaluate them.  Columns already present, non-rich ids
        and features the project doesn't have are ignored, and the frame is
        returned unchanged when there is nothing to add.
        """
        want = [
            m for m in dict.fromkeys(metric_ids)
            if is_rich_metric(m) and m not in df.columns
        ]
        if df.empty or not want:
            return df
        extra = self.load_rich_features(
            metric_ids=want, segment_ids={str(w) for w in df.index}
        )
        if extra.empty:
            return df
        extra = extra.reindex([str(w) for w in df.index])
        extra.index = df.index  # align to the caller's index dtype/order exactly
        return df.join(extra)

    def rich_essence_frame(
        self,
        segment_ids: "list[str] | set[str]",
        min_finite_frac: float = 0.5,
    ) -> pd.DataFrame:
        """Essence-ready shipped features for the given windows.

        Drops columns too sparse to anchor a criterion bound (finite in fewer than
        ``min_finite_frac`` of the rows) and columns that are constant across them
        — both would otherwise masquerade as a perfect "signature".  The row set is
        the exemplars *plus* their background, so a column is judged on the same
        rows the contrast will use.
        """
        df = self.load_rich_features(segment_ids=segment_ids)
        if df.empty:
            return df
        finite = df.notna().mean()
        spread = df.max(skipna=True) - df.min(skipna=True)
        keep = [
            c for c in df.columns
            if float(finite.get(c, 0.0)) >= min_finite_frac
            and float(spread.get(c, 0.0) or 0.0) > 0.0
        ]
        return df[keep]

    def essence_frames(
        self,
        exemplar_ids: list[str],
        background_ids: list[str],
        ex_metrics: "pd.DataFrame | None" = None,
        bg_metrics: "pd.DataFrame | None" = None,
        min_exemplars: int = 2,
    ) -> "EssenceFrames":
        """Assemble the exemplar/background tables essence extraction ranges over.

        The two metric spaces are *unioned* column-wise rather than one replacing
        the other: the shipped features bring the oscillation / angular / jerk
        signal the interpretable metrics simply do not have, while the clip
        metrics keep zone geometry ("nose past edge") — which the feature table
        has no equivalent of — in the running.  Whichever separates the exemplars
        best then wins on merit inside the greedy search.

        Each space contributes only if it can supply *both* frames (a contrast
        needs a background), so the result degrades cleanly rather than failing:
        no feature table → the classic clip metrics alone; unreadable pose → the
        shipped features alone (and no pose drive needed); neither → an empty
        frame the caller reports.  ``ex_metrics``/``bg_metrics`` are pre-computed
        clip-metric tables (the caller already has them, and they are expensive).
        """
        ex_ids = [str(w) for w in exemplar_ids]
        bg_ids = [str(w) for w in background_ids]
        parts_ex: list[pd.DataFrame] = []
        parts_bg: list[pd.DataFrame] = []
        used: list[str] = []
        notes: list[str] = []

        # An all-NaN table means the pose behind those clips could not be read;
        # it is not a usable space, and claiming it as a source would be a lie.
        if _has_data(ex_metrics) and _has_data(bg_metrics):
            parts_ex.append(ex_metrics.reindex(ex_ids))
            parts_bg.append(bg_metrics)
            used.append("pose metrics")

        if self.has_rich_features():
            rich = self.rich_essence_frame(set(ex_ids) | set(bg_ids))
            rich_ex = rich.reindex(ex_ids).dropna(how="all") if not rich.empty else rich
            rich_bg = rich.reindex(bg_ids).dropna(how="all") if not rich.empty else rich
            if len(rich_ex) >= int(min_exemplars) and not rich_bg.empty:
                # Reindex back onto the full id lists so both spaces share rows;
                # windows absent from the feature table simply carry NaN, which the
                # per-metric finite-data floor already handles.
                parts_ex.append(rich.reindex(ex_ids))
                parts_bg.append(rich.reindex(bg_ids).dropna(how="all"))
                used.append("extracted features")
            elif not rich.empty:
                notes.append(
                    "The selected clips aren't in this project's extracted feature "
                    "table (re-run Feature Extraction to include them) — essence "
                    "used the pose metrics only."
                )
        else:
            notes.append(
                "No extracted feature table for this project "
                "(derived/representations/segment_features.parquet) — essence used "
                "the pose metrics only. Run Feature Extraction for a sharper essence."
            )

        if not parts_ex or not parts_bg:
            return EssenceFrames(pd.DataFrame(), pd.DataFrame(), used, notes)
        # Background rows differ per space (rich drops windows it lacks); align on
        # the union so a partially-covered pool still contrasts on every column.
        bg_index = parts_bg[0].index
        for p in parts_bg[1:]:
            bg_index = bg_index.union(p.index)
        E = pd.concat([p for p in parts_ex], axis=1)
        B = pd.concat([p.reindex(bg_index) for p in parts_bg], axis=1)
        # Ids are namespaced per space, so a duplicate can only come from a caller
        # passing feature columns in as pose metrics; keep the first either way.
        E = E.loc[:, ~E.columns.duplicated()]
        B = B.loc[:, ~B.columns.duplicated()]
        return EssenceFrames(E, B, used, notes)

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
        Essence-chosen shipped features count as known too, so a rich definition
        survives a reload exactly like a hand-set one.
        """
        if self._project_root is None:
            return
        known = set(self._metric_ids) | set(self.rich_feature_ids())
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
        known = set(self._metric_ids) | set(self.rich_feature_ids())

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
    def mine(
        df: pd.DataFrame,
        criteria: list[Criterion],
        match_all: bool = True,
        rank_scores: "pd.Series | None" = None,
    ) -> MiningResult:
        """Return clips whose metrics satisfy the active criteria.

        ``match_all`` True → a clip must pass every enabled criterion (AND);
        False → it needs at least one (OR).  ``scores`` is the fraction of
        enabled criteria each matched clip passes (always 1.0 in AND mode), used
        by the UI to rank near-misses.

        ``rank_scores`` — an optional per-window graded "essence-likeness" score
        (see :meth:`build_essence_scorer`).  When given, the matched clips are
        ranked by it and their ``scores`` carry it, so a behaviour that genuinely
        overlaps others (and therefore still yields a broad AND match) still gets
        loaded best-first up to the UI cap, instead of the flat 1.0 AND score that
        can't tell one match from another.
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

        graded = None
        if rank_scores is not None:
            graded = pd.to_numeric(rank_scores, errors="coerce")

        ids = df.index.to_numpy()
        for i in np.nonzero(keep)[0]:
            wid = str(ids[i])
            result.matched_ids.append(wid)
            if graded is not None and wid in graded.index:
                gv = float(graded.loc[wid])
                result.scores[wid] = gv if np.isfinite(gv) else 0.0
            else:
                result.scores[wid] = float(pass_counts[i] / max(1, n_active))
        # Rank strongest matches first — by essence-likeness when a ranker was
        # supplied, else by the fraction of criteria passed (meaningful in OR mode).
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
            if mid in _ESSENCE_SKIP_METRICS:
                continue
            ex = pd.to_numeric(exemplar_df[mid], errors="coerce").to_numpy(dtype=float)
            ex = ex[np.isfinite(ex)]
            if ex.size < need:
                continue
            # A feature the exemplars are perfectly constant on (e.g. the fixed
            # segment ``duration_sec``, or a saturated fraction) has zero spread,
            # so it would rank as the "tightest" signature yet range to a
            # degenerate point and discriminate nothing — skip it.
            if float(ex.max() - ex.min()) <= 1e-9:
                continue
            # Dead / all-zero features (e.g. an inactive keypoint's distance for a
            # paw-less model) have a perfect zero spread, so they'd masquerade as
            # the tightest "signature" and flood the top-k. A feature the exemplars
            # agree on only because it's constantly zero carries no essence — skip it.
            if np.all(np.abs(ex) <= 1e-9):
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
        recall_target: float = 0.8,
    ) -> list[Criterion]:
        """Infer ready-to-mine criteria that capture what the exemplars share.

        With a **background** (``population_df`` — a random pool sample or the
        already-scored pool) this delegates to :meth:`extract_contrastive_essence`,
        which chooses the features and ranges by how the exemplars *differ from the
        pool*, not merely by where they happen to agree.  That is the difference
        between "these clips are all fast" (half the project is fast) and "these
        clips are faster / jerkier / more stretched than the pool" — only the
        latter is a usable definition.

        Without any background it falls back to the legacy shared-feature ranging
        (:meth:`rank_shared_features` + :meth:`extract_essence`), which is broad but
        is all that can be inferred from the exemplars alone.
        """
        if population_df is not None and not population_df.empty:
            crits = cls.extract_contrastive_essence(
                exemplar_df, population_df, k=k, recall_target=recall_target
            )
            if crits:
                return crits
            # Fall through to the legacy path only if the contrastive search found
            # nothing separable (degenerate exemplars / background).
        ranked = cls.rank_shared_features(exemplar_df, population_df)
        top = [mid for mid, _score in ranked[: max(1, int(k))]]
        if not top:
            return []
        return cls.extract_essence(
            exemplar_df, list(exemplar_df.index), top, method=method, pad_frac=pad_frac
        )

    # -- contrastive essence (exemplars vs. a background sample) --------------

    @staticmethod
    def _separation(e: np.ndarray, b: np.ndarray) -> float:
        """|AUC − 0.5| of exemplar values *e* against background values *b*.

        A units-free, distribution-free measure of how far the exemplars sit from
        the pool on one metric (0 = indistinguishable, 0.5 = perfectly separated).
        Stable at tiny exemplar counts, where a fitted effect size is not.
        """
        e = e[np.isfinite(e)]
        b = b[np.isfinite(b)]
        if e.size == 0 or b.size == 0:
            return 0.0
        allv = np.concatenate([e, b])
        order = allv.argsort(kind="mergesort")
        ranks = np.empty(order.size, dtype=float)
        ranks[order] = np.arange(order.size, dtype=float)
        auc = (ranks[: e.size].sum() - e.size * (e.size - 1) / 2.0) / (e.size * b.size)
        return abs(auc - 0.5)

    @classmethod
    def _usable_essence_metrics(
        cls,
        E: pd.DataFrame,
        B: pd.DataFrame,
        min_finite_frac: float = 0.5,
        sep_floor: float = 0.10,
        max_features: int | None = ESSENCE_MAX_FEATURES,
    ) -> list[str]:
        """Metrics worth considering for essence: enough exemplar data, non-constant,
        and at least mildly separated from the background.

        The separation floor is what stops a metric the exemplars are tight on *by
        coincidence* (noise) from being committed when only a handful of clips are
        selected — the small-exemplar overfitting failure.  If nothing clears the
        floor (very few or very homogeneous clips), the best-separated few are
        returned so the search still has something to work with.

        Returned best-separated first and truncated to ``max_features``, which is
        what keeps the greedy criteria search (O(features) per round) interactive
        over the ~1100-column shipped feature space; it is a no-op for the far
        smaller interpretable metric set.
        """
        n_ex = int(len(E))
        need = max(2, int(np.ceil(min_finite_frac * n_ex)))
        scored: list[tuple[str, float]] = []
        for mid in E.columns:
            if mid in _ESSENCE_SKIP_METRICS or mid not in B.columns:
                continue
            e = pd.to_numeric(E[mid], errors="coerce").to_numpy(float)
            ef = e[np.isfinite(e)]
            if ef.size < need or float(ef.max() - ef.min()) <= 1e-9:
                continue
            if float(np.nanmax(np.abs(ef))) <= 1e-9:  # constant-zero (dead feature)
                continue
            b = pd.to_numeric(B[mid], errors="coerce").to_numpy(float)
            scored.append((mid, cls._separation(e, b)))
        scored.sort(key=lambda t: t[1], reverse=True)
        keep = [mid for mid, s in scored if s >= sep_floor] or [
            mid for mid, _s in scored[:3]
        ]
        if max_features is not None and max_features > 0:
            keep = keep[: int(max_features)]
        return keep

    @classmethod
    def extract_contrastive_essence(
        cls,
        exemplar_df: pd.DataFrame,
        background_df: pd.DataFrame,
        k: int = 5,
        recall_target: float = 0.8,
        max_leak: float = 0.02,
        min_gain_frac: float = 0.005,
    ) -> list[Criterion]:
        """Greedily build a small AND-box that keeps the exemplars but not the pool.

        Starting from the empty conjunction, at each step it picks the single
        metric bound (a lower bound, an upper bound, or a two-sided band, with the
        threshold drawn from the exemplars' own quantiles) that removes the most
        still-surviving *background* clips while retaining at least
        ``recall_target`` of the exemplars.  It stops after ``k`` features, once the
        surviving background falls below ``max_leak`` of the pool, or once no
        remaining metric removes a meaningful share of background.

        This is what makes the result both *tight* (it is scored against the pool,
        so it can't settle on a range that half the project also satisfies) and
        *robust across exemplar count*: with a handful of clips the recall floor
        forces the bounds out to the exemplars' span (so none are dropped) and the
        one-sided direction is set by which side of the pool they sit on; with
        hundreds the quantile thresholds trim outliers and the box tightens instead
        of ballooning.  Ranges are returned one-sided wherever the pool only sits
        on one side, which reads far more naturally than a spurious two-sided box.
        """
        if exemplar_df is None or exemplar_df.empty or background_df is None or background_df.empty:
            return []
        feats = cls._usable_essence_metrics(exemplar_df, background_df)
        if not feats:
            return []
        n_ex = int(len(exemplar_df))
        n_bg = int(len(background_df))
        keep_min = max(1, int(np.ceil(recall_target * n_ex)))
        gain_min = max(1, int(np.ceil(min_gain_frac * n_bg)))

        ex_cols = {m: pd.to_numeric(exemplar_df[m], errors="coerce").to_numpy(float) for m in feats}
        bg_cols = {m: pd.to_numeric(background_df[m], errors="coerce").to_numpy(float) for m in feats}
        # Per-metric pad (2% of the exemplar 10–90 span) so exemplars sit just
        # inside the bound rather than exactly on it — helps held-out clips clear it.
        pads = {}
        for m in feats:
            ef = ex_cols[m][np.isfinite(ex_cols[m])]
            spread = float(np.percentile(ef, 90) - np.percentile(ef, 10)) if ef.size >= 5 else float(ef.max() - ef.min())
            pads[m] = 0.02 * spread if spread > 0 else 0.02 * (abs(float(np.median(ef))) + 1e-6)

        surv_ex = np.ones(n_ex, dtype=bool)
        surv_bg = np.ones(n_bg, dtype=bool)
        chosen: list[tuple[str, float | None, float | None]] = []
        used: set[str] = set()

        low_qs = (0, 1, 2, 3, 5, 8, 12)
        high_qs = (88, 92, 95, 97, 98, 99, 100)

        for _step in range(max(1, int(k))):
            best = None  # (removed, recall, metric, low, high, new_ex, new_bg)
            for m in feats:
                if m in used:
                    continue
                e = ex_cols[m]
                b = bg_cols[m]
                fin_e = np.isfinite(e)
                fin_b = np.isfinite(b)
                surv_e_vals = e[surv_ex & fin_e]
                if surv_e_vals.size < 2:
                    continue
                lows = [float(np.percentile(surv_e_vals, q)) - pads[m] for q in low_qs]
                highs = [float(np.percentile(surv_e_vals, q)) + pads[m] for q in high_qs]

                def _eval(low, high):
                    ok_e = np.ones(n_ex, dtype=bool)
                    if low is not None:
                        ok_e &= (e >= low) | ~fin_e
                    if high is not None:
                        ok_e &= (e <= high) | ~fin_e
                    new_ex = surv_ex & ok_e
                    if int(new_ex.sum()) < keep_min:
                        return None
                    ok_b = np.ones(n_bg, dtype=bool)
                    if low is not None:
                        ok_b &= (b >= low) | ~fin_b
                    if high is not None:
                        ok_b &= (b <= high) | ~fin_b
                    new_bg = surv_bg & ok_b
                    removed = int(surv_bg.sum() - new_bg.sum())
                    return removed, int(new_ex.sum()), new_ex, new_bg

                cands: list[tuple[float | None, float | None]] = []
                cands += [(lo, None) for lo in lows]
                cands += [(None, hi) for hi in highs]
                cands += [(lo, hi) for lo in lows[::2] for hi in highs[::2] if hi > lo]
                for low, high in cands:
                    r = _eval(low, high)
                    if r is None:
                        continue
                    removed, rec, new_ex, new_bg = r
                    if best is None or removed > best[0] or (removed == best[0] and rec > best[1]):
                        best = (removed, rec, m, low, high, new_ex, new_bg)
            if best is None:
                break
            removed, _rec, m, low, high, new_ex, new_bg = best
            if removed < gain_min and chosen:
                break
            surv_ex, surv_bg = new_ex, new_bg
            used.add(m)
            chosen.append(
                (m, None if low is None else round(low, 3), None if high is None else round(high, 3))
            )
            if surv_bg.sum() / max(1, n_bg) <= max_leak:
                break

        return [Criterion(metric_id=m, low=lo, high=hi, enabled=True) for (m, lo, hi) in chosen]

    @classmethod
    def build_essence_scorer(
        cls,
        exemplar_df: pd.DataFrame,
        background_df: pd.DataFrame,
        feature_ids: list[str] | None = None,
        n_features: int = 8,
    ) -> "EssenceScorer | None":
        """Build a graded exemplar-likeness ranker (higher = more exemplar-like).

        Used to order matched clips so "load top N" gets the *most* exemplar-like
        first — decisive for behaviours that overlap others, where even a good box
        still matches broadly.  Prefers a regularised logistic fit of exemplars
        vs. background over the most discriminative features; falls back to a
        stateless, discrimination-weighted contrastive z-score when there are too
        few exemplars to fit or scikit-learn is unavailable.  Returns ``None`` only
        when there is no separable signal at all.
        """
        if exemplar_df is None or exemplar_df.empty or background_df is None or background_df.empty:
            return None
        feats = feature_ids or cls._usable_essence_metrics(exemplar_df, background_df)
        if not feats:
            return None
        # Keep the most discriminative handful for a stable, low-variance ranker.
        ranked = sorted(
            feats,
            key=lambda m: cls._separation(
                pd.to_numeric(exemplar_df[m], errors="coerce").to_numpy(float),
                pd.to_numeric(background_df[m], errors="coerce").to_numpy(float),
            ),
            reverse=True,
        )
        feats = ranked[: max(1, int(n_features))]
        return EssenceScorer.build(exemplar_df, background_df, feats)


@dataclass
class EssenceScorer:
    """A fitted, reusable exemplar-likeness scorer over a fixed feature set.

    ``score(df)`` returns a per-row Series (higher = more exemplar-like).  Two
    backends: a logistic fit (when enough exemplars) and a discrimination-weighted
    contrastive z-score fallback; both consume the same metric DataFrame the rest
    of mining uses, so the ranker needs no extra feature pass.
    """

    feature_ids: list[str]
    bg_median: dict[str, float]
    bg_scale: dict[str, float]
    direction: dict[str, float]
    weight: dict[str, float]
    _model: object | None = None  # fitted sklearn pipeline, when available

    # Below this many exemplars a logistic fit is too unstable; use the
    # stateless weighted z-score instead.
    _MIN_LOGISTIC_EXEMPLARS = 6

    @classmethod
    def build(
        cls,
        exemplar_df: pd.DataFrame,
        background_df: pd.DataFrame,
        feature_ids: list[str],
        n_bg_max: int = 1500,
    ) -> "EssenceScorer | None":
        feats = [m for m in feature_ids if m in exemplar_df.columns and m in background_df.columns]
        if not feats:
            return None
        B = background_df[feats].apply(pd.to_numeric, errors="coerce")
        E = exemplar_df[feats].apply(pd.to_numeric, errors="coerce")
        bmed = B.median()
        bscale = (B - bmed).abs().median() * 1.4826  # robust std (MAD)
        bscale = bscale.replace(0, np.nan).fillna(B.std()).replace(0, 1.0)
        emed = E.median()
        direction = np.sign(emed - bmed).replace(0, 1.0)
        weight = {}
        for m in feats:
            weight[m] = float(
                ClipMetricsService._separation(
                    E[m].to_numpy(float), B[m].to_numpy(float)
                )
            )
        wsum = sum(weight.values()) or 1.0
        weight = {m: (w / wsum) for m, w in weight.items()}

        model = None
        if len(exemplar_df) >= cls._MIN_LOGISTIC_EXEMPLARS:
            model = cls._try_fit_logistic(E, B, feats, n_bg_max)

        return cls(
            feature_ids=feats,
            bg_median={m: float(bmed[m]) for m in feats},
            bg_scale={m: float(bscale[m]) for m in feats},
            direction={m: float(direction[m]) for m in feats},
            weight=weight,
            _model=model,
        )

    @staticmethod
    def _try_fit_logistic(E, B, feats, n_bg_max):
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import RobustScaler
        except Exception:
            return None
        try:
            rng = np.random.default_rng(0)
            bg = B
            if len(B) > n_bg_max:
                idx = rng.choice(len(B), size=n_bg_max, replace=False)
                bg = B.iloc[idx]
            X = np.vstack([E.to_numpy(float), bg.to_numpy(float)])
            y = np.r_[np.ones(len(E)), np.zeros(len(bg))]
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3),
            )
            pipe.fit(X, y)
            return pipe
        except Exception:
            return None

    def score(self, df: pd.DataFrame) -> pd.Series:
        cols = [m for m in self.feature_ids if m in df.columns]
        if not cols:
            return pd.Series(0.0, index=df.index)
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        if self._model is not None:
            try:
                proba = self._model.predict_proba(X[self.feature_ids].to_numpy(float))[:, 1]
                return pd.Series(proba, index=df.index)
            except Exception:
                pass  # fall back to the stateless score
        # Discrimination-weighted, direction-aware contrastive z (clipped so a
        # single wild feature can't dominate the ranking).
        total = pd.Series(0.0, index=df.index)
        for m in cols:
            z = (X[m] - self.bg_median[m]) / (self.bg_scale[m] or 1.0)
            z = (z * self.direction[m]).clip(-4, 4)
            total = total.add(z * self.weight.get(m, 0.0), fill_value=0.0)
        return total
