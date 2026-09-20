"""Project and subject ROI configuration management."""

from __future__ import annotations

import copy
import re
import threading
from pathlib import Path
from typing import Any

from abel.storage.file_store import read_yaml, write_yaml
from abel.utils.roi_geometry import (
    normalize_roi as _normalize_roi_geom,
    roi_shape,
    simplify_freehand,
)

# Per-ROI overlay colors (index 0 = ROI 1, index 1 = ROI 2, …).
# Eight slots covers reasonable multi-zone experiments.
ROI_COLORS = [
    "#FFC107",  # ROI 1: amber/yellow  (legacy "Target Zone" color)
    "#4FC3F7",  # ROI 2: light blue
    "#FF7043",  # ROI 3: orange
    "#CE93D8",  # ROI 4: lavender
    "#A5D6A7",  # ROI 5: light green
    "#F48FB1",  # ROI 6: pink
    "#80CBC4",  # ROI 7: teal
    "#FFCC80",  # ROI 8: pale amber
]
MAX_ROIS = len(ROI_COLORS)

# MOG2 variance threshold for the local background-subtraction windows (squared
# Mahalanobis distance a pixel must exceed to count as foreground): lower = more
# sensitive.  Stored as motion.bg_var_threshold only when it differs from the
# default, so projects that never touch it keep a byte-identical ROI file and
# their context-feature cache stays valid.
DEFAULT_BG_VAR_THRESHOLD = 16
BG_VAR_THRESHOLD_MIN = 4
BG_VAR_THRESHOLD_MAX = 100

# Every feature emitted against a target zone, at frame or segment level:
#   in_roi_1_nose, nose_to_roi_1_edge_dist, nose_roi_1_axial_abs_p90,
#   roi_1_present, nose_to_target_dist_max, head_angle_to_target_mean.
_ROI_COLUMN_RE = re.compile(r"(^|_)roi_\d+(_|$)|_to_target(_|$)|_angle_to_target(_|$)")


def is_roi_column(name: str) -> bool:
    """True when *name* is a feature computed against a target-zone ROI.

    Such columns go all-NaN when the session's ROI resolves to no area, so
    callers use this to tell whether a model actually depends on ROIs before
    deciding that a missing zone matters.
    """
    return bool(_ROI_COLUMN_RE.search(str(name)))


# Parsed-config cache, keyed by (resolved path, mtime_ns, size).
#
# Every tab and service builds its own ROIService, and the ROI Definition tab
# alone calls load() four times per subject switch.  Re-reading and re-parsing
# the whole file each time is invisible for a rectangles-only project (~10 KB)
# and crippling for hand-drawn polygon ROIs, where the file runs to megabytes.
# The cache is module-level so those separate instances share one parse, and is
# keyed on the file's mtime+size so an edit from anywhere (another process, a
# hand-edited YAML) invalidates it without any explicit notification.
_CONFIG_CACHE: dict[Path, tuple[int, int, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def clear_roi_config_cache() -> None:
    """Drop every cached ROI config (tests, and project teardown)."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()


class ROIService:
    """Persist and resolve ROI settings with subject-level overrides.

    Supports an arbitrary number of per-subject *target zones* (up to
    MAX_ROIS).  The number of zones is stored as ``roi_count`` in the
    project YAML.  Legacy single-zone projects are automatically migrated
    to the new ``target_zones`` list format on first load.
    """

    ROI_FILE = Path("config") / "environment_rois.yaml"

    @staticmethod
    def _default_roi() -> dict[str, int]:
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "schema_version": "0.3.0",
            "roi_count": 1,
            "project_rois": {
                "target_zones": [cls._default_roi()],
                "subject_crop": cls._default_roi(),
            },
            "subject_rois": {},
            "motion": {
                "local_radius_px": 36,
            },
            "roi_excluded_day_labels": [],
        }

    @staticmethod
    def _normalize_roi(raw: Any) -> dict[str, Any]:
        """Canonicalize an ROI dict, preserving shape (rect/circle/polygon).

        Delegates to :func:`abel.utils.roi_geometry.normalize_roi`, which keeps
        shape parameters and always (re)derives the ``x/y/w/h`` bounding box so
        rectangle-only consumers keep working unchanged.
        """
        return _normalize_roi_geom(raw)

    @classmethod
    def _extract_target_zones(cls, roi_block: dict, legacy_fallback: Any = None) -> list[dict[str, int]]:
        """Extract a normalized list of target-zone dicts from a config block.

        Handles three storage layouts:
        - New: ``{"target_zones": [{…}, …]}``
        - Old single-zone: ``{"target_zone": {…}}``
        - Legacy flat key (top-level ``TMT_zone``): passed via *legacy_fallback*.
        """
        if "target_zones" in roi_block and isinstance(roi_block["target_zones"], list):
            zones = [cls._normalize_roi(z) for z in roi_block["target_zones"]]
            return zones if zones else [cls._default_roi()]
        if "target_zone" in roi_block:
            return [cls._normalize_roi(roi_block["target_zone"])]
        if legacy_fallback:
            return [cls._normalize_roi(legacy_fallback)]
        return [cls._default_roi()]

    @classmethod
    def _normalize(cls, raw: Any) -> dict[str, Any]:
        data = raw if isinstance(raw, dict) else {}
        cfg = cls.default_config()

        # Backward compatibility with legacy flat keys.
        legacy_tgt = data.get("TMT_zone", {})
        legacy_crop = data.get("subject_crop", {})

        project_rois = data.get("project_rois", {}) if isinstance(data.get("project_rois", {}), dict) else {}

        target_zones = cls._extract_target_zones(project_rois, legacy_tgt)
        cfg["project_rois"]["subject_crop"] = cls._normalize_roi(
            project_rois.get("subject_crop", legacy_crop)
        )

        # roi_count: prefer the explicit key, fall back to the number of zones saved.
        raw_count = data.get("roi_count", len(target_zones))
        roi_count = max(1, min(int(raw_count or 1), MAX_ROIS))
        cfg["roi_count"] = roi_count

        # Ensure the zones list is exactly roi_count long.
        while len(target_zones) < roi_count:
            target_zones.append(cls._default_roi())
        cfg["project_rois"]["target_zones"] = target_zones[:roi_count]

        # Per-subject ROIs.
        subject_rois: dict[str, Any] = {}
        raw_subject = data.get("subject_rois", {})
        if isinstance(raw_subject, dict):
            for subject_id, subject_cfg in raw_subject.items():
                if not isinstance(subject_cfg, dict):
                    continue
                s_zones = cls._extract_target_zones(subject_cfg)
                subject_rois[str(subject_id)] = {
                    "target_zones": s_zones,
                    "subject_crop": cls._normalize_roi(subject_cfg.get("subject_crop", {})),
                }
        cfg["subject_rois"] = subject_rois

        motion = data.get("motion", {}) if isinstance(data.get("motion", {}), dict) else {}
        cfg["motion"]["local_radius_px"] = max(8, int(motion.get("local_radius_px", 36) or 36))
        if "bg_var_threshold" in motion:
            bg = cls._clamp_bg_threshold(motion.get("bg_var_threshold"))
            if bg != DEFAULT_BG_VAR_THRESHOLD:
                cfg["motion"]["bg_var_threshold"] = bg
        raw_excl = data.get("roi_excluded_day_labels", [])
        cfg["roi_excluded_day_labels"] = [
            str(d) for d in (raw_excl if isinstance(raw_excl, list) else []) if d
        ]
        return cfg

    def load(self, project_root: Path, *, mutable: bool = True) -> dict[str, Any]:
        """Return the project's ROI config.

        ``mutable=False`` hands back the cached object itself instead of a copy.
        On a megabyte-scale file the defensive deep copy costs more than
        everything else put together, so read-only callers, which build fresh
        dicts out of what they read, opt out.  Anyone who edits the result and
        saves it must take the default.
        """
        path = (project_root / self.ROI_FILE).resolve()
        stamp = self._file_stamp(path)
        if stamp is not None:
            with _CACHE_LOCK:
                hit = _CONFIG_CACHE.get(path)
            if hit is not None and (hit[0], hit[1]) == stamp:
                # Callers mutate what they get back (edit a subject, then save),
                # so hand out a copy and keep the cached parse pristine.
                return copy.deepcopy(hit[2]) if mutable else hit[2]

        cfg = self._normalize(read_yaml(path, {}))
        # Re-stat after reading: if the file changed underneath us, the stamp we
        # would cache no longer describes the bytes we parsed.
        if stamp is not None and self._file_stamp(path) == stamp:
            with _CACHE_LOCK:
                _CONFIG_CACHE[path] = (stamp[0], stamp[1], copy.deepcopy(cfg))
        return cfg

    def save(self, project_root: Path, config: dict[str, Any]) -> None:
        path = (project_root / self.ROI_FILE).resolve()
        clean = self._normalize(config)
        write_yaml(path, clean)
        stamp = self._file_stamp(path)
        with _CACHE_LOCK:
            if stamp is None:
                _CONFIG_CACHE.pop(path, None)
            else:
                # We just produced the canonical form: cache it rather than
                # making the next load() re-parse what we only now wrote out.
                _CONFIG_CACHE[path] = (stamp[0], stamp[1], copy.deepcopy(clean))

    @staticmethod
    def _file_stamp(path: Path) -> tuple[int, int] | None:
        """``(mtime_ns, size)`` for *path*, or None when it cannot be stat'd."""
        try:
            st = path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def compact_polygons(self, project_root: Path) -> dict[str, int]:
        """Decimate every stored freehand polygon in place; report what it saved.

        Projects drawn before capture-time decimation shipped carry the full raw
        traces, so the file stays slow until it is rewritten once.  Returns
        ``{"polygons", "points_before", "points_after", "bytes_before",
        "bytes_after"}``; ``polygons == 0`` means there was nothing to do and
        the file is left untouched.
        """
        path = (project_root / self.ROI_FILE).resolve()
        stats = {"polygons": 0, "points_before": 0, "points_after": 0,
                 "bytes_before": 0, "bytes_after": 0}
        try:
            stats["bytes_before"] = path.stat().st_size
        except OSError:
            return stats
        cfg = self.load(project_root)

        def _compact(block: dict) -> None:
            zones = block.get("target_zones")
            if not isinstance(zones, list):
                return
            for i, zone in enumerate(zones):
                if roi_shape(zone) != "polygon":
                    continue
                pts = zone.get("points") or []
                simplified = simplify_freehand(pts)
                if len(simplified) >= len(pts):
                    continue
                stats["polygons"] += 1
                stats["points_before"] += len(pts)
                stats["points_after"] += len(simplified)
                zones[i] = _normalize_roi_geom(
                    {"shape": "polygon", "points": simplified}
                )

        proj = cfg.get("project_rois")
        if isinstance(proj, dict):
            _compact(proj)
        for block in (cfg.get("subject_rois") or {}).values():
            if isinstance(block, dict):
                _compact(block)

        if stats["polygons"]:
            self.save(project_root, cfg)
        try:
            stats["bytes_after"] = path.stat().st_size
        except OSError:
            stats["bytes_after"] = stats["bytes_before"]
        return stats

    def get_roi_count(self, project_root: Path) -> int:
        """Return the number of target zones configured for this project."""
        cfg = self.load(project_root, mutable=False)
        return max(1, int(cfg.get("roi_count", 1)))

    def resolve_target_rois(
        self, project_root: Path, subject_id: str | None = None
    ) -> list[dict[str, int]]:
        """Return the full list of target-zone ROIs for *subject_id*.

        Accepts a plain subject_id or a composite ``"subject::session"`` key.
        Lookup order: composite key → base subject_id → project defaults.
        For each slot, the subject/session override is used when it has non-zero
        dimensions; otherwise the project default fills the slot.
        """
        cfg = self.load(project_root, mutable=False)
        roi_count = max(1, int(cfg.get("roi_count", 1)))
        proj_zones: list[dict] = cfg.get("project_rois", {}).get("target_zones", [])
        subject_rois = cfg.get("subject_rois", {})

        s_block: dict = {}
        if subject_id:
            key = str(subject_id)
            s_block = subject_rois.get(key, {})
            # Composite key fallback: try base subject_id if composite not found
            if not s_block and "::" in key:
                base_sid = key.split("::", 1)[0]
                s_block = subject_rois.get(base_sid, {})

        if s_block:
            s_zones: list[dict] = s_block.get("target_zones", [])
            result = []
            for i in range(roi_count):
                s_roi = self._normalize_roi(s_zones[i]) if i < len(s_zones) else self._default_roi()
                if s_roi["w"] > 0 and s_roi["h"] > 0:
                    result.append(s_roi)
                else:
                    p_roi = self._normalize_roi(proj_zones[i]) if i < len(proj_zones) else self._default_roi()
                    result.append(p_roi)
            return result

        result = []
        for i in range(roi_count):
            z = proj_zones[i] if i < len(proj_zones) else self._default_roi()
            result.append(self._normalize_roi(z))
        return result

    def get_roi_excluded_days(self, project_root: Path) -> list[str]:
        """Return day labels for which ROI features should be suppressed."""
        cfg = self.load(project_root, mutable=False)
        return list(cfg.get("roi_excluded_day_labels", []))

    def resolve_target_roi(
        self, project_root: Path, subject_id: str | None = None
    ) -> dict[str, int]:
        """Return the primary (first) target-zone ROI.  Backward-compatible."""
        rois = self.resolve_target_rois(project_root, subject_id)
        return rois[0] if rois else self._default_roi()

    def subjects_without_target_area(
        self, project_root: Path, subject_keys: list[str]
    ) -> list[str]:
        """Return the *subject_keys* whose target zones all resolve to no area.

        A subject with no ``subject_rois`` entry falls back to the project
        defaults, which are a zero-size box unless the project draws one.  That
        resolves silently, and every downstream ROI/target feature then comes out
        all-NaN (see ``context_feature_service._roi_point_features``), models
        that lean on those features score such a session off a constant input.
        Callers use this to refuse the run instead of emitting a flat trace.

        A subject counts as covered when *any* of its zones has area, matching
        ``resolve_target_rois``' per-slot project fallback.
        """
        from abel.utils import roi_geometry

        return [
            key for key in subject_keys
            if not any(
                roi_geometry.roi_has_area(roi)
                for roi in self.resolve_target_rois(project_root, key)
            )
        ]

    def resolve_subject_crop_roi(
        self, project_root: Path, subject_id: str | None = None
    ) -> dict[str, int]:
        cfg = self.load(project_root, mutable=False)
        if subject_id:
            subject_rois = cfg.get("subject_rois", {})
            key = str(subject_id)
            s_block = subject_rois.get(key, {})
            if not s_block and "::" in key:
                s_block = subject_rois.get(key.split("::", 1)[0], {})
            raw = s_block.get("subject_crop")
            roi = self._normalize_roi(raw)
            if roi["w"] > 0 and roi["h"] > 0:
                return roi
        return self._normalize_roi(cfg.get("project_rois", {}).get("subject_crop", {}))

    def local_motion_radius(self, project_root: Path) -> int:
        cfg = self.load(project_root, mutable=False)
        motion = cfg.get("motion", {})
        return max(8, int(motion.get("local_radius_px", 36) or 36))

    def bg_var_threshold(self, project_root: Path) -> int:
        """MOG2 variance threshold for the local background-subtraction windows."""
        cfg = self.load(project_root, mutable=False)
        return self._clamp_bg_threshold(cfg.get("motion", {}).get("bg_var_threshold"))

    def set_bg_var_threshold(self, project_root: Path, value: int) -> bool:
        """Persist the threshold; returns False (and writes nothing) when unchanged.

        Skipping no-op writes matters: the context-feature cache signature
        hashes this file, so even a cosmetic rewrite could force a rebuild.
        """
        value = self._clamp_bg_threshold(value)
        if value == self.bg_var_threshold(project_root):
            return False
        cfg = self.load(project_root)
        cfg.setdefault("motion", {})["bg_var_threshold"] = value
        self.save(project_root, cfg)
        return True

    @staticmethod
    def _clamp_bg_threshold(raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_BG_VAR_THRESHOLD
        return max(BG_VAR_THRESHOLD_MIN, min(BG_VAR_THRESHOLD_MAX, value))
