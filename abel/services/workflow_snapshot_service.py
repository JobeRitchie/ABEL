"""Workflow snapshot — serialise the current trained-pipeline state.

A *workflow snapshot* records everything needed to apply a trained ABEL
model to a new batch of videos, without re-running the full active-learning
loop.  It is written automatically every time a model is successfully trained
(full pipeline or retrain) and can be applied to arbitrary new sessions via
the "Batch Run" panel.

Snapshot layout (derived/workflow_snapshot.json):
{
  "schema_version": "1.0",
  "created_at": "<ISO-8601>",
  "model_version": "<dir name under derived/models/>",  # target behaviour's model
  "target_behavior": "<behavior_id>",                    # first active behaviour
  "selected_behavior_models": {behavior_id: model_dir},  # ALL competing models
  "excluded_behavior_ids": [behavior_id, ...],
  "behavior_definitions": [ {...}, ... ],
  "segment_window_frames": int,
  "segment_stride_frames": int,
  "excluded_feature_cols": [str, ...],
  "fps": float | null,
  "context_feature_config": { farneback params ... },
  "pose_preset_id": str | null,
  "export_csv": bool,
  "export_xlsx": bool,
  "export_labeled_video": bool,
  "run_settings": { ... }   <- verbatim ui_settings from run_settings.json
}

``selected_behavior_models`` is the key field for Direct Use: it lists every
behaviour that participates in the competitive inference run.  ``build_from_
project`` auto-discovers all trained models from ``derived/models/`` (mirroring
the live inference path) so the snapshot replays the full multi-behaviour
competition rather than collapsing to a single model.  ``no_behavior`` is not
run as a competitor unless explicitly selected in the Temporal Refinement tab.

The snapshot is *always overwritten* to reflect the most recent trained state.
Previous snapshots are archived to derived/workflow_snapshot_archive/ so the
user can audit history but the live snapshot is never stale.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")

SNAPSHOT_SCHEMA_VERSION = "1.0"
SNAPSHOT_FILENAME = "workflow_snapshot.json"
ARCHIVE_DIR = "workflow_snapshot_archive"


@dataclass
class WorkflowSnapshot:
    """All parameters needed to replay inference on new sessions."""

    model_version: str
    target_behavior: str
    segment_window_frames: int = 60
    segment_stride_frames: int = 15
    excluded_feature_cols: list[str] = field(default_factory=list)
    fps: float | None = None
    context_feature_config: dict[str, Any] = field(default_factory=dict)
    # Whether the source model was trained with video-derived features
    # (optical flow + substrate motion + spatial context).  Direct Use MUST
    # match this: if the model used context features, they have to be
    # recomputed on the new data or inference receives all-zero columns.
    use_video_features: bool = False
    # Keypoint (bodypart) names the model was trained on.  Direct Use compares
    # these against the new pose files and lets the user map mismatched names so
    # derived feature columns line up with what the model expects.
    pose_keypoints: list[str] = field(default_factory=list)
    pose_preset_id: str | None = None
    export_csv: bool = True
    export_xlsx: bool = False
    export_labeled_video: bool = False
    run_settings: dict[str, Any] = field(default_factory=dict)
    # Temporal refinement settings captured at snapshot time (keys: "__all__" and per-behavior).
    # If non-empty and the model pkl exists, batch run can offer temporal precision inference.
    temporal_refinement_settings: dict[str, Any] = field(default_factory=dict)
    # Per-behavior review thresholds from the Temporal Review tab.
    # Structure: {"__all__": {...}, "by_behavior": {behavior_id: {onset_threshold, ...}}}
    temporal_review_settings: dict[str, Any] = field(default_factory=dict)
    # Mapping of behavior_id → model_version directory name (all behaviours
    # that should participate in the competitive inference run).
    selected_behavior_models: dict[str, str] = field(default_factory=dict)
    # Behavior IDs excluded from inference (e.g. "no_behavior").
    excluded_behavior_ids: list[str] = field(default_factory=list)
    # Behavior definitions from the source project
    behavior_definitions: list[dict[str, Any]] = field(default_factory=list)
    # Smoothing/preprocessing settings from the source project
    smoothing_settings: dict[str, Any] = field(default_factory=dict)
    # Feature-extraction settings (project.yaml "feature_extraction" block) from
    # the source project.  Replaying these into the target keeps the Features tab
    # in sync (e.g. the "Include video features" checkbox) and ensures Run Models
    # resolves use_video_features the same way the model was trained.
    feature_extraction_settings: dict[str, Any] = field(default_factory=dict)
    # Per-step timing data (seconds) from the source project for progress estimation.
    # Keys: "pose_clean", "pose_features", "context_features", "representation",
    #        "inference", "temporal_refinement", "bout_extraction"
    step_timings: dict[str, float] = field(default_factory=dict)
    # Source project path for reference
    source_project_path: str = ""
    created_at: str = ""
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at or datetime.now(tz=timezone.utc).isoformat(),
            "model_version": self.model_version,
            "target_behavior": self.target_behavior,
            "segment_window_frames": self.segment_window_frames,
            "segment_stride_frames": self.segment_stride_frames,
            "excluded_feature_cols": list(self.excluded_feature_cols),
            "fps": self.fps,
            "context_feature_config": dict(self.context_feature_config),
            "use_video_features": bool(self.use_video_features),
            "pose_keypoints": list(self.pose_keypoints),
            "pose_preset_id": self.pose_preset_id,
            "export_csv": self.export_csv,
            "export_xlsx": self.export_xlsx,
            "export_labeled_video": self.export_labeled_video,
            "run_settings": dict(self.run_settings),
            "temporal_refinement_settings": dict(self.temporal_refinement_settings),
            "temporal_review_settings": dict(self.temporal_review_settings),
            "selected_behavior_models": dict(self.selected_behavior_models),
            "excluded_behavior_ids": list(self.excluded_behavior_ids),
            "behavior_definitions": list(self.behavior_definitions),
            "smoothing_settings": dict(self.smoothing_settings),
            "feature_extraction_settings": dict(self.feature_extraction_settings),
            "step_timings": dict(self.step_timings),
            "source_project_path": self.source_project_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowSnapshot":
        return cls(
            model_version=str(d.get("model_version") or ""),
            target_behavior=str(d.get("target_behavior") or ""),
            segment_window_frames=int(d.get("segment_window_frames") or 60),
            segment_stride_frames=int(d.get("segment_stride_frames") or 15),
            excluded_feature_cols=list(d.get("excluded_feature_cols") or []),
            fps=float(d["fps"]) if d.get("fps") is not None else None,
            context_feature_config=dict(d.get("context_feature_config") or {}),
            use_video_features=bool(d.get("use_video_features", False)),
            pose_keypoints=list(d.get("pose_keypoints") or []),
            pose_preset_id=str(d["pose_preset_id"]) if d.get("pose_preset_id") else None,
            export_csv=bool(d.get("export_csv", True)),
            export_xlsx=bool(d.get("export_xlsx", False)),
            export_labeled_video=bool(d.get("export_labeled_video", False)),
            run_settings=dict(d.get("run_settings") or {}),
            temporal_refinement_settings=dict(d.get("temporal_refinement_settings") or {}),
            temporal_review_settings=dict(d.get("temporal_review_settings") or {}),
            selected_behavior_models=dict(d.get("selected_behavior_models") or {}),
            excluded_behavior_ids=list(d.get("excluded_behavior_ids") or []),
            behavior_definitions=list(d.get("behavior_definitions") or []),
            smoothing_settings=dict(d.get("smoothing_settings") or {}),
            feature_extraction_settings=dict(d.get("feature_extraction_settings") or {}),
            step_timings=dict(d.get("step_timings") or {}),
            source_project_path=str(d.get("source_project_path") or ""),
            created_at=str(d.get("created_at") or ""),
            schema_version=str(d.get("schema_version") or SNAPSHOT_SCHEMA_VERSION),
        )


class WorkflowSnapshotService:
    """Read/write the current workflow snapshot for a project."""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, project_root: Path) -> WorkflowSnapshot | None:
        """Load the current snapshot; returns None if none has been saved yet."""
        path = project_root / "derived" / SNAPSHOT_FILENAME
        if not path.exists():
            return None
        try:
            data = read_json(path, {})
            if not data:
                return None
            return WorkflowSnapshot.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load workflow snapshot: %s", exc)
            return None

    def is_valid(self, project_root: Path, snapshot: WorkflowSnapshot) -> tuple[bool, str]:
        """Check whether the snapshot's model dir still exists and is usable.

        Returns (valid, reason_if_invalid).
        """
        if not snapshot.model_version:
            return False, "No model version recorded."
        model_dir = project_root / "derived" / "models" / snapshot.model_version
        if not model_dir.exists():
            return False, f"Model directory '{snapshot.model_version}' no longer exists."
        if not (model_dir / "model_state.pkl").exists():
            return False, f"model_state.pkl missing from '{snapshot.model_version}'."
        return True, ""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, project_root: Path, snapshot: WorkflowSnapshot) -> None:
        """Overwrite the live snapshot; archive the previous one first."""
        derived = project_root / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        path = derived / SNAPSHOT_FILENAME

        # Archive previous snapshot before overwriting.
        if path.exists():
            archive_dir = derived / ARCHIVE_DIR
            archive_dir.mkdir(parents=True, exist_ok=True)
            try:
                prev = read_json(path, {})
                ts = str(prev.get("created_at") or "").replace(":", "-").replace("+", "")[:19]
                mv = str(prev.get("model_version") or "unknown")
                archive_name = f"{ts}_{mv}.json" if ts else f"snapshot_{mv}.json"
                shutil.copy2(path, archive_dir / archive_name)
            except Exception as exc:
                logger.debug("Could not archive previous snapshot: %s", exc)

        snapshot.created_at = datetime.now(tz=timezone.utc).isoformat()
        data = snapshot.to_dict()
        write_json(path, data)
        logger.info(
            "Workflow snapshot saved: model=%s behavior=%s",
            snapshot.model_version,
            snapshot.target_behavior,
        )

    # ------------------------------------------------------------------
    # Build from project state
    # ------------------------------------------------------------------

    def build_from_project(self, project_root: Path) -> WorkflowSnapshot:
        """Assemble a snapshot from the current project's persisted state.

        Reads the existing snapshot (if any) as a baseline, then overlays
        current config/model data so every setting the Direct Run pipeline
        needs is captured.  Raises if no model version can be determined.
        """
        import yaml  # noqa: PLC0415

        base = self.load(project_root)
        if base is None:
            base = WorkflowSnapshot(model_version="", target_behavior="")

        # ── Model version: pick the latest model directory ──────────
        models_dir = project_root / "derived" / "models"
        if models_dir.exists():
            versions = sorted(
                [d.name for d in models_dir.iterdir() if d.is_dir()],
                reverse=True,
            )
            if versions:
                base.model_version = versions[0]

        if not base.model_version:
            raise ValueError(
                "No trained model found. Run the full pipeline at least once "
                "before creating a snapshot."
            )

        # ── Behavior definitions ────────────────────────────────────
        behavior_path = project_root / "config" / "behavior_definitions.yaml"
        if behavior_path.exists():
            try:
                raw = yaml.safe_load(behavior_path.read_text(encoding="utf-8")) or {}
                base.behavior_definitions = raw.get("behaviors", [])
            except Exception:
                pass

        # ── Preprocessing / smoothing settings ──────────────────────
        preproc_path = project_root / "config" / "preprocessing.yaml"
        if preproc_path.exists():
            try:
                raw = yaml.safe_load(preproc_path.read_text(encoding="utf-8")) or {}
                base.smoothing_settings = raw
            except Exception:
                pass

        # ── Pipeline settings from project.yaml ─────────────────────
        proj_path = project_root / "project.yaml"
        if proj_path.exists():
            try:
                raw = yaml.safe_load(proj_path.read_text(encoding="utf-8")) or {}
                bm = raw.get("behavior_model") or {}
                if bm.get("segment_window_frames"):
                    base.segment_window_frames = int(bm["segment_window_frames"])
                if bm.get("segment_stride_frames"):
                    base.segment_stride_frames = int(bm["segment_stride_frames"])
                if raw.get("default_fps"):
                    base.fps = float(raw["default_fps"])
                if "use_video_features" in bm:
                    base.use_video_features = bool(bm["use_video_features"])
                # Capture the whole feature_extraction block so the target
                # project's Features tab opens with identical settings.
                fx = raw.get("feature_extraction")
                if isinstance(fx, dict) and fx:
                    base.feature_extraction_settings = dict(fx)
            except Exception:
                pass

        # ── use_video_features: prefer the value the model was actually
        #    trained with (model run_settings.json), which is authoritative
        #    over the live project.yaml.  Without this flag Direct Use would
        #    skip context-feature extraction and feed the model all-zero
        #    optical-flow columns, wrecking inference. ─────────────────
        trained_use_video = self._trained_use_video_features(
            project_root, base.model_version,
        )
        if trained_use_video is not None:
            base.use_video_features = trained_use_video

        # ── Pose keypoints the model was trained on ──────────────────
        keypoints = self._read_pose_keypoints(project_root)
        if keypoints:
            base.pose_keypoints = keypoints

        # ── Temporal refinement settings (inference-phase config) ────
        tr_path = project_root / "config" / "temporal_refinement_settings.json"
        if tr_path.exists():
            try:
                base.temporal_refinement_settings = read_json(tr_path, {})
            except Exception:
                pass

        # ── Temporal review settings (per-behavior thresholds) ───────
        review_path = project_root / "config" / "temporal_review_settings.json"
        if review_path.exists():
            try:
                base.temporal_review_settings = read_json(review_path, {})
            except Exception:
                pass

        # ── Resolve selected_behavior_models ─────────────────────────
        # Priority:
        #   1. An explicit map the user configured in the Temporal
        #      Refinement tab (temporal_refinement_settings → by_behavior
        #      → target_behavior → selected_behavior_models).
        #   2. Auto-discovery of *every* trained model on disk, mirroring
        #      the live inference path (TemporalRefinementService.
        #      _resolve_competition_model_versions).  This is what makes
        #      Direct Use replay the full multi-behaviour competition
        #      instead of collapsing to a single model.
        by_behavior = (base.temporal_refinement_settings.get("by_behavior") or {})
        tb_block = by_behavior.get("target_behavior") or {}
        explicit_sbm = {
            str(k): str(v)
            for k, v in (tb_block.get("selected_behavior_models") or {}).items()
            if str(v).strip()
        }
        explicit_excluded = [
            str(v).strip()
            for v in (tb_block.get("excluded_behavior_ids") or [])
            if str(v).strip()
        ]

        auto_sbm = self._auto_resolve_behavior_models(project_root)
        if explicit_sbm:
            # Start from auto-discovery so newly-trained behaviours that the
            # user never revisited in the TR tab are still captured, then
            # let the explicit choices override.
            resolved = dict(auto_sbm)
            resolved.update(explicit_sbm)
            base.selected_behavior_models = resolved
        elif auto_sbm:
            base.selected_behavior_models = auto_sbm
        # else: leave whatever the baseline snapshot had.

        # ── Exclusions ───────────────────────────────────────────────
        # Carry forward whatever the user excluded in the TR tab.  We do
        # NOT auto-include no_behavior as a competitor (see
        # _auto_resolve_behavior_models), so there is no need to special-
        # case it here — it simply never enters the map unless the user
        # picked it explicitly, in which case we honour that choice.
        excluded_set = set(explicit_excluded) | set(base.excluded_behavior_ids or [])
        # Drop any stale exclusions that no longer name a captured behaviour.
        base.excluded_behavior_ids = sorted(
            bid for bid in excluded_set if bid in base.selected_behavior_models
        )

        # ── Derive target_behavior (first active, non-excluded) ──────
        # Stored target_behavior may be stale from a prior snapshot, so we
        # always recompute it from the current behaviour map/definitions.
        active_ids = [
            bid for bid in base.selected_behavior_models
            if bid not in base.excluded_behavior_ids
        ]
        base.target_behavior = ""
        if active_ids:
            # Prefer the first behaviour (by definition order) that is active.
            def_order = [
                b.get("behavior_id", b.get("name", ""))
                for b in base.behavior_definitions
            ]
            ordered = [bid for bid in def_order if bid in active_ids]
            ordered += [bid for bid in active_ids if bid not in ordered]
            base.target_behavior = ordered[0] if ordered else active_ids[0]
        if not base.target_behavior:
            # Fallback: first active behaviour from definitions.  Note the
            # field is "is_active" (legacy snapshots may use "active").
            for b in base.behavior_definitions:
                bid = b.get("behavior_id", b.get("name", ""))
                is_active = b.get("is_active", b.get("active", True))
                if is_active and not self._is_no_behavior_label(bid):
                    base.target_behavior = bid
                    break

        # ── model_version: keep it consistent with target_behavior ───
        # It is used by is_valid() and as the single-model legacy fallback,
        # so point it at the chosen target behaviour's model when possible.
        if base.target_behavior and base.target_behavior in base.selected_behavior_models:
            base.model_version = base.selected_behavior_models[base.target_behavior]

        base.source_project_path = str(project_root)
        return base

    # ------------------------------------------------------------------
    # Behavior-model auto-discovery (mirrors TemporalRefinementService)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(value).strip()
        )
        return safe or "target_behavior"

    @staticmethod
    def _is_no_behavior_label(label: str) -> bool:
        import re  # noqa: PLC0415
        token = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
        return token in {
            "no_behavior", "no_behaviour", "nobehavior", "nobehaviour",
        }

    @staticmethod
    def _trained_use_video_features(
        project_root: Path, model_version: str,
    ) -> bool | None:
        """Read ``use_video_features`` from the model's ``run_settings.json``.

        This is the ground truth for what the model was trained on.  Returns
        None when no run_settings could be read (caller keeps its current
        value).
        """
        if not model_version:
            return None
        rs_path = (
            project_root / "derived" / "models" / model_version / "run_settings.json"
        )
        if not rs_path.exists():
            return None
        rs = read_json(rs_path, {})
        bm = rs.get("behavior_model") or {}
        if "use_video_features" in bm:
            return bool(bm["use_video_features"])
        if "use_video_features" in rs:
            return bool(rs["use_video_features"])
        return None

    @staticmethod
    def _read_pose_keypoints(project_root: Path) -> list[str]:
        """Recover the trained keypoint names from the project's frame_pose.

        Each keypoint contributes a ``<name>_velocity_x`` column, so the set of
        such columns identifies the body parts the model was trained on.
        """
        fp = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if not fp.exists():
            return []
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
            cols = list(pq.read_schema(fp).names)
        except Exception:
            try:
                import pandas as pd  # noqa: PLC0415
                cols = pd.read_parquet(fp).columns.tolist()
            except Exception:
                return []
        suffix = "_velocity_x"
        return sorted(c[: -len(suffix)] for c in cols if c.endswith(suffix))

    def _auto_resolve_behavior_models(self, project_root: Path) -> dict[str, str]:
        """Map every trained behaviour → its newest model directory.

        Scans ``derived/models/`` for directories that contain a usable
        ``model_state.pkl`` and reads each one's ``run_settings.json`` to
        recover the ``target_behavior`` it was trained for.  When more than
        one directory targets the same behaviour, the most recently modified
        one wins.  ``no_behavior`` models are skipped — they are not run as a
        competitor in the auto path (matching the inference-time
        ``_resolve_competition_model_versions``); the null class is handled
        separately by the competition logic.
        """
        models_root = project_root / "derived" / "models"
        if not models_root.exists():
            return {}

        resolved: dict[str, str] = {}
        best_mtime: dict[str, float] = {}
        for p in sorted(models_root.iterdir()):
            if not p.is_dir() or not p.name.startswith("behavior_model_"):
                continue
            if not (p / "model_state.pkl").exists():
                continue
            settings = read_json(p / "run_settings.json", {})
            bid = str(settings.get("target_behavior", "")).strip()
            if not bid or self._is_no_behavior_label(bid):
                continue
            mtime = p.stat().st_mtime
            if bid not in resolved or mtime > best_mtime.get(bid, 0.0):
                resolved[bid] = p.name
                best_mtime[bid] = mtime
        return resolved

    # ------------------------------------------------------------------
    # Archive listing
    # ------------------------------------------------------------------

    def list_archived(self, project_root: Path) -> list[dict[str, Any]]:
        """Return metadata for all archived snapshots, newest first."""
        archive_dir = project_root / "derived" / ARCHIVE_DIR
        if not archive_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for f in sorted(archive_dir.iterdir(), reverse=True):
            if f.suffix == ".json":
                try:
                    d = read_json(f, {})
                    entries.append({
                        "file": f.name,
                        "model_version": d.get("model_version", "?"),
                        "target_behavior": d.get("target_behavior", "?"),
                        "created_at": d.get("created_at", "?"),
                    })
                except Exception:
                    pass
        return entries

    def load_archived(self, project_root: Path, filename: str) -> WorkflowSnapshot | None:
        """Load a specific archived snapshot by filename."""
        path = project_root / "derived" / ARCHIVE_DIR / filename
        if not path.exists():
            return None
        try:
            data = read_json(path, {})
            return WorkflowSnapshot.from_dict(data) if data else None
        except Exception as exc:
            logger.warning("Failed to load archived snapshot %s: %s", filename, exc)
            return None
