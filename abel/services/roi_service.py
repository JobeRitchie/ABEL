"""Project and subject ROI configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abel.storage.file_store import read_yaml, write_yaml

# Per-ROI overlay colours (index 0 = ROI 1, index 1 = ROI 2, …).
# Eight slots covers reasonable multi-zone experiments.
ROI_COLORS = [
    "#FFC107",  # ROI 1 — amber/yellow  (legacy "Target Zone" colour)
    "#4FC3F7",  # ROI 2 — light blue
    "#FF7043",  # ROI 3 — orange
    "#CE93D8",  # ROI 4 — lavender
    "#A5D6A7",  # ROI 5 — light green
    "#F48FB1",  # ROI 6 — pink
    "#80CBC4",  # ROI 7 — teal
    "#FFCC80",  # ROI 8 — pale amber
]
MAX_ROIS = len(ROI_COLORS)


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
    def _normalize_roi(raw: Any) -> dict[str, int]:
        src = raw if isinstance(raw, dict) else {}
        return {
            "x": int(src.get("x", 0) or 0),
            "y": int(src.get("y", 0) or 0),
            "w": max(0, int(src.get("w", 0) or 0)),
            "h": max(0, int(src.get("h", 0) or 0)),
        }

    @classmethod
    def _extract_target_zones(cls, roi_block: dict, legacy_fallback: Any = None) -> list[dict[str, int]]:
        """Extract a normalised list of target-zone dicts from a config block.

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
        raw_excl = data.get("roi_excluded_day_labels", [])
        cfg["roi_excluded_day_labels"] = [
            str(d) for d in (raw_excl if isinstance(raw_excl, list) else []) if d
        ]
        return cfg

    def load(self, project_root: Path) -> dict[str, Any]:
        raw = read_yaml(project_root / self.ROI_FILE, {})
        cfg = self._normalize(raw)
        return cfg

    def save(self, project_root: Path, config: dict[str, Any]) -> None:
        clean = self._normalize(config)
        write_yaml(project_root / self.ROI_FILE, clean)

    def get_roi_count(self, project_root: Path) -> int:
        """Return the number of target zones configured for this project."""
        cfg = self.load(project_root)
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
        cfg = self.load(project_root)
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
        cfg = self.load(project_root)
        return list(cfg.get("roi_excluded_day_labels", []))

    def resolve_target_roi(
        self, project_root: Path, subject_id: str | None = None
    ) -> dict[str, int]:
        """Return the primary (first) target-zone ROI.  Backward-compatible."""
        rois = self.resolve_target_rois(project_root, subject_id)
        return rois[0] if rois else self._default_roi()

    def resolve_subject_crop_roi(
        self, project_root: Path, subject_id: str | None = None
    ) -> dict[str, int]:
        cfg = self.load(project_root)
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
        cfg = self.load(project_root)
        motion = cfg.get("motion", {})
        return max(8, int(motion.get("local_radius_px", 36) or 36))
