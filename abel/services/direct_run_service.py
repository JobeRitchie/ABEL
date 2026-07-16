"""Direct Use pipeline — replay a trained workflow on new sessions.

Runs the full inference pipeline from raw pose/video to bout outputs
using a frozen WorkflowSnapshot from a source project, without any
active-learning or training steps.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)
from abel.services.context_feature_service import ContextFeatureService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.workflow_snapshot_service import (
    WorkflowSnapshot,
    WorkflowSnapshotService,
)
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml

logger = logging.getLogger("abel")


# Step identifiers (ordered)
STEP_IDS = [
    "pose_clean",
    "pose_features",
    "context_features",
    "representation",
    "inference",
    "temporal_refinement",
    "bout_extraction",
]

STEP_LABELS = {
    "pose_clean": "Cleaning Pose Data",
    "pose_features": "Extracting Pose Features",
    "context_features": "Computing Context Features",
    "representation": "Building Representations",
    "inference": "Running Dense Inference",
    "temporal_refinement": "Temporal Refinement",
    "bout_extraction": "Extracting Behavior Bouts",
}


@dataclass
class DirectRunProgress:
    """Progress state for a direct-run pipeline execution."""

    current_step: str = ""
    current_step_index: int = 0
    total_steps: int = len(STEP_IDS)
    step_progress: float = 0.0  # 0.0–1.0 within current step
    step_message: str = ""
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    completed_steps: list[str] = field(default_factory=list)
    step_timings: dict[str, float] = field(default_factory=dict)
    is_complete: bool = False
    error: str | None = None


class DirectRunService:
    """Orchestrate the full inference pipeline using a workflow snapshot."""

    def __init__(self) -> None:
        self._imports = ImportService()
        self._pose = PoseProcessingService()
        self._context = ContextFeatureService()
        self._repr = BehaviorRepresentationService()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(
        self,
        target_project_root: Path,
        source_project_root: Path,
        snapshot: WorkflowSnapshot,
        progress_cb: Callable[[DirectRunProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the full direct-use pipeline.

        Parameters
        ----------
        target_project_root : Path
            The new project directory (already has videos + pose imported).
        source_project_root : Path
            The original project containing the trained model.
        snapshot : WorkflowSnapshot
            Frozen pipeline configuration from the source project.
        progress_cb : callable, optional
            Called with DirectRunProgress updates.

        Returns
        -------
        dict with status, session_count, bout_count, etc.
        """
        self._cancelled = False
        state = DirectRunProgress()
        reference_timings = dict(snapshot.step_timings) if snapshot.step_timings else {}
        pipeline_start = time.monotonic()

        def _emit(step_id: str, step_idx: int, pct: float, msg: str) -> None:
            state.current_step = step_id
            state.current_step_index = step_idx
            state.step_progress = max(0.0, min(1.0, pct))
            state.step_message = msg
            state.elapsed_seconds = time.monotonic() - pipeline_start
            # Estimate remaining time from reference timings
            remaining = 0.0
            for i, sid in enumerate(STEP_IDS):
                if i < step_idx:
                    continue
                ref = reference_timings.get(sid, 0.0)
                if i == step_idx:
                    remaining += ref * (1.0 - pct)
                else:
                    remaining += ref
            state.estimated_remaining_seconds = max(0.0, remaining)
            if progress_cb:
                progress_cb(state)

        manifest = self._imports.load_manifest(target_project_root)
        if manifest is None:
            state.error = "No import manifest found in target project."
            if progress_cb:
                progress_cb(state)
            return {"status": "error", "error": state.error}

        sessions = manifest.linked_sessions
        if not sessions:
            state.error = "No linked sessions in the target project."
            if progress_cb:
                progress_cb(state)
            return {"status": "error", "error": state.error}

        session_ids = [str(s.session_id) for s in sessions]
        video_by_id = {v.asset_id: v for v in manifest.videos}
        pose_by_id = {p.asset_id: p for p in manifest.poses}
        fps_by_session: dict[str, float] = {}
        for s in sessions:
            v = video_by_id.get(s.video_asset_id)
            fps_by_session[str(s.session_id)] = float(v.fps) if v and v.fps else (snapshot.fps or 30.0)

        # ── Write snapshot settings into the target project's config ──
        # This ensures TR service and other downstream code can find the
        # settings in their canonical locations.
        target_config = target_project_root / "config"
        target_config.mkdir(parents=True, exist_ok=True)
        if snapshot.temporal_refinement_settings:
            write_json(
                target_config / "temporal_refinement_settings.json",
                snapshot.temporal_refinement_settings,
            )
        if snapshot.temporal_review_settings:
            write_json(
                target_config / "temporal_review_settings.json",
                snapshot.temporal_review_settings,
            )
        if snapshot.behavior_definitions:
            import yaml  # noqa: PLC0415
            bd_path = target_config / "behavior_definitions.yaml"
            if not bd_path.exists():
                bd_path.write_text(
                    yaml.dump(
                        {"behaviors": snapshot.behavior_definitions},
                        default_flow_style=False,
                        allow_unicode=True,
                    ),
                    encoding="utf-8",
                )

        # ── Create project.yaml and project_state.json so the target
        #    folder is a valid ABEL project that can be opened later. ──
        # Resolve the authoritative use_video_features for this model so the
        # opened project (Features tab + Run Models) agrees with how the model
        # was trained.  Writing the wrong/absent flag silently disables context
        # features on re-open, which breaks inference for video-trained models.
        use_video = self._resolve_use_video_features(snapshot, source_project_root)
        # Feature-extraction settings: prefer the snapshot, but fall back to the
        # source project's project.yaml so snapshots created before this field
        # existed still propagate the Features-tab settings.
        feature_extraction_src = dict(snapshot.feature_extraction_settings or {})
        if not feature_extraction_src:
            try:
                src_proj = read_yaml(source_project_root / "project.yaml", {}) or {}
                fx = src_proj.get("feature_extraction")
                if isinstance(fx, dict):
                    feature_extraction_src = dict(fx)
            except Exception:
                pass
        project_yaml_path = target_project_root / "project.yaml"
        project_state_path = target_project_root / "project_state.json"
        if not project_yaml_path.exists():
            from datetime import datetime  # noqa: PLC0415
            # Mirror the source's feature_extraction block (the Features tab's
            # single source of truth) and force use_video_features to the
            # resolved value so the checkbox + Run Models both stay correct.
            feature_extraction = dict(feature_extraction_src)
            feature_extraction["use_video_features"] = bool(use_video)
            feature_extraction.pop("context_padding_frames", None)  # feature removed
            write_yaml(project_yaml_path, {
                "schema_version": "0.3.0",
                "project_name": target_project_root.name,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "behavior_model": {
                    # Canonical keys read across the app (schemas.BehaviorModelConfig).
                    "segment_window_frames": snapshot.segment_window_frames,
                    "segment_stride_frames": snapshot.segment_stride_frames,
                    "use_video_features": bool(use_video),
                },
                "feature_extraction": feature_extraction,
                "source_project": snapshot.source_project_path or str(source_project_root),
            })
        if not project_state_path.exists():
            write_json(project_state_path, {"schema_version": "0.3.0"})

        # ── Carry the Features-tab toggle state into the target project ──
        # The Feature Extraction tab persists its settings to THREE places, but
        # only project.yaml's feature_extraction block is mirrored above. Copy
        # the other two so the new project's checkboxes (feature groups +
        # robustness/invariant features) match the source instead of silently
        # reverting to defaults.
        self._carry_feature_settings(source_project_root, target_project_root)

        # NOTE: ROIs are intentionally NOT inherited from the source project.
        # They are pixel coordinates tied to the source's camera/frame geometry
        # and almost never valid on new footage.  The Direct Use tab writes the
        # freshly-drawn ROIs into environment_rois.yaml before this run; if the
        # user skipped that, the tab warns them first.

        # ── Copy ALL models from source project ────────────────────
        # selected_behavior_models maps behavior_id → model_dir_name.
        # We must copy every model directory that will participate in
        # the competitive inference run.
        behavior_models = dict(snapshot.selected_behavior_models or {})
        excluded_ids = list(snapshot.excluded_behavior_ids or [])

        # Robustness: older snapshots (created before multi-behaviour
        # auto-discovery) saved an empty map, which would collapse Direct
        # Use to a single behaviour.  Re-resolve every trained model from
        # the source project so the full competition runs regardless of
        # how stale the snapshot is.
        if not behavior_models:
            behavior_models = WorkflowSnapshotService()._auto_resolve_behavior_models(
                source_project_root
            )
            if behavior_models:
                logger.info(
                    "Snapshot had no behavior map; auto-resolved %d model(s) "
                    "from source project.", len(behavior_models),
                )

        # Last-resort fallback: the legacy single-model approach.
        if not behavior_models and snapshot.target_behavior and snapshot.model_version:
            behavior_models = {
                snapshot.target_behavior: snapshot.model_version,
            }

        all_model_dirs: set[str] = set(behavior_models.values())
        # Also include the legacy model_version so it's always present
        if snapshot.model_version:
            all_model_dirs.add(snapshot.model_version)

        for model_dir_name in all_model_dirs:
            src = source_project_root / "derived" / "models" / model_dir_name
            dst = target_project_root / "derived" / "models" / model_dir_name
            if dst.exists():
                continue
            if src.exists():
                shutil.copytree(src, dst)
            else:
                logger.warning("Model directory not found: %s", src)

        total_sessions = len(session_ids)

        # ── Keypoint aliases ─────────────────────────────────────────
        # New pose files may name their keypoints differently than the model
        # was trained on.  The UI writes a rename map ({data_name: model_name})
        # into the target config; applying it on pose load makes every derived
        # feature column align with the model's expected names.
        keypoint_aliases = read_json(
            target_project_root / "config" / "keypoint_aliases.json", {}
        ) or {}
        if keypoint_aliases:
            logger.info(
                "Applying %d keypoint alias(es) to new pose data.",
                len(keypoint_aliases),
            )

        try:
            # ── Step 1: Pose Cleaning ────────────────────────────────
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}
            for idx, s in enumerate(sessions):
                _emit("pose_clean", 0, idx / total_sessions,
                      f"Cleaning pose {idx + 1}/{total_sessions}")
                if self._cancelled:
                    return {"status": "cancelled"}
                pa = pose_by_id.get(s.pose_asset_id)
                if pa is None:
                    continue
                pp = Path(pa.local_path) if pa.local_path and Path(pa.local_path).exists() else Path(pa.source_path)
                if not pp.exists():
                    continue
                # Write cleaned pose to derived
                clean_dir = target_project_root / "derived" / "pose_clean"
                clean_dir.mkdir(parents=True, exist_ok=True)
                out_path = clean_dir / f"{s.session_id}.parquet"
                if out_path.exists():
                    continue
                try:
                    from abel.models.schemas import PoseSmoothingSettings
                    settings = PoseSmoothingSettings(**snapshot.smoothing_settings) if snapshot.smoothing_settings else PoseSmoothingSettings()
                    pose_data = self._pose.load_and_clean(pp, settings=settings, keypoint_aliases=keypoint_aliases)
                    df = pd.DataFrame({
                        "centroid_x": pose_data.centroid_x,
                        "centroid_y": pose_data.centroid_y,
                    })
                    df.to_parquet(out_path, index=False)
                except Exception as exc:
                    logger.warning("Pose clean failed for %s: %s", s.session_id, exc)
            state.step_timings["pose_clean"] = time.monotonic() - step_start
            state.completed_steps.append("pose_clean")

            # ── Step 2: Pose Features ────────────────────────────────
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}
            for idx, s in enumerate(sessions):
                _emit("pose_features", 1, idx / total_sessions,
                      f"Extracting pose features {idx + 1}/{total_sessions}")
                if self._cancelled:
                    return {"status": "cancelled"}
                pa = pose_by_id.get(s.pose_asset_id)
                if pa is None:
                    continue
                pp = Path(pa.local_path) if pa.local_path and Path(pa.local_path).exists() else Path(pa.source_path)
                if not pp.exists():
                    continue
                sid = str(s.session_id)
                fps = fps_by_session.get(sid, 30.0)
                vid = str(s.video_asset_id)
                animal_id = str(s.subject_id or sid)
                sess_dir = target_project_root / "derived" / "pose_features" / "sessions"
                sess_dir.mkdir(parents=True, exist_ok=True)
                out_path = sess_dir / f"{sid}.parquet"
                if out_path.exists():
                    continue
                try:
                    self._pose.extract_and_save_frame_pose_features(
                        project_root=target_project_root,
                        pose_path=pp,
                        fps=fps,
                        animal_id=animal_id,
                        session_id=sid,
                        video_id=vid,
                        keypoint_aliases=keypoint_aliases,
                    )
                except Exception as exc:
                    logger.warning("Pose features failed for %s: %s", sid, exc)
            # Consolidate
            try:
                self._pose.consolidate_session_files(target_project_root)
            except Exception as exc:
                logger.warning("Pose consolidation failed: %s", exc)
            state.step_timings["pose_features"] = time.monotonic() - step_start
            state.completed_steps.append("pose_features")

            # ── Step 3: Context Features (only when video features enabled) ──
            step_start = time.monotonic()
            _use_video = self._resolve_use_video_features(snapshot, source_project_root)
            if _use_video:
                if self._cancelled:
                    return {"status": "cancelled"}
                from abel.services.context_feature_service import ContextFeatureConfig
                ctx_cfg = self._build_context_config(snapshot)
                for idx, s in enumerate(sessions):
                    _emit("context_features", 2, idx / total_sessions,
                          f"Context features {idx + 1}/{total_sessions}")
                    if self._cancelled:
                        return {"status": "cancelled"}
                    va = video_by_id.get(s.video_asset_id)
                    pa = pose_by_id.get(s.pose_asset_id)
                    if va is None or pa is None:
                        continue
                    vp = Path(va.local_path) if va.local_path and Path(va.local_path).exists() else Path(va.source_path)
                    pp = Path(pa.local_path) if pa.local_path and Path(pa.local_path).exists() else Path(pa.source_path)
                    if not vp.exists() or not pp.exists():
                        continue
                    sid = str(s.session_id)
                    animal_id = str(s.subject_id or sid)
                    sess_dir = target_project_root / "derived" / "context_features" / "sessions"
                    sess_dir.mkdir(parents=True, exist_ok=True)
                    out_path = sess_dir / f"{sid}.parquet"
                    if out_path.exists():
                        continue
                    try:
                        self._context.compute_frame_context(
                            project_root=target_project_root,
                            video_path=vp,
                            pose_path=pp,
                            animal_id=animal_id,
                            session_id=sid,
                            config=ctx_cfg,
                            keypoint_aliases=keypoint_aliases,
                        )
                    except Exception as exc:
                        logger.warning("Context features failed for %s: %s", sid, exc)
                # Consolidate
                try:
                    self._context.consolidate_session_files(target_project_root)
                except Exception as exc:
                    logger.warning("Context consolidation failed: %s", exc)
            else:
                _emit("context_features", 2, 1.0, "Video features disabled — skipping context extraction.")
            state.step_timings["context_features"] = time.monotonic() - step_start
            state.completed_steps.append("context_features")

            # ── Step 4: Build Representations ────────────────────────
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}
            _emit("representation", 3, 0.0, "Building segment representations…")
            repr_cfg = RepresentationConfig(
                window_size_frames=snapshot.segment_window_frames,
                window_stride_frames=snapshot.segment_stride_frames,
                excluded_feature_cols=frozenset(snapshot.excluded_feature_cols),
            )
            pose_feat_path = target_project_root / "derived" / "pose_features" / "frame_pose.parquet"
            ctx_feat_path = (
                target_project_root / "derived" / "context_features" / "frame_context.parquet"
                if _use_video else None
            )
            try:
                self._repr.build(
                    project_root=target_project_root,
                    frame_pose_path=pose_feat_path,
                    frame_context_path=ctx_feat_path,
                    config=repr_cfg,
                    session_ids=set(session_ids),
                    progress_cb=lambda msg: _emit("representation", 3, 0.5, msg),
                )
            except Exception as exc:
                logger.warning("Representation build failed: %s", exc)
            state.step_timings["representation"] = time.monotonic() - step_start
            state.completed_steps.append("representation")
            _emit("representation", 3, 1.0, "Representations built.")

            # ── Step 5: Dense Inference (all behaviours) ────────────
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}
            _emit("inference", 4, 0.0, "Starting dense inference…")
            from abel.temporal_refinement.temporal_refinement_service import (
                TemporalRefinementService,
                TemporalRefinementConfig,
            )
            tr_service = TemporalRefinementService()
            tr_service.set_project(target_project_root)

            # Build inference config from snapshot
            tr_global = (snapshot.temporal_refinement_settings or {}).get("__all__") or {}
            tr_by_beh = (snapshot.temporal_refinement_settings or {}).get("by_behavior") or {}
            # The target_behavior block carries the model-selection settings
            tb_block = tr_by_beh.get("target_behavior") or tr_global

            tr_cfg = TemporalRefinementConfig()
            tr_cfg.selected_behavior_models = dict(behavior_models)
            tr_cfg.excluded_behavior_ids = list(excluded_ids)

            # Apply inference-phase params from the global/target_behavior block
            for key in (
                "inference_step_seconds", "inference_warmup_seconds",
                "inference_parallel_enabled", "inference_max_workers",
                "inhibition_weight", "probability_temperature",
            ):
                val = tb_block.get(key, tr_global.get(key))
                if val is not None:
                    setattr(tr_cfg, key, val)

            # Use "target_behavior" as concept_id — this triggers
            # multi-behaviour competitive inference.
            concept_id = "target_behavior"
            try:
                tr_service.run_temporal_refinement_inference(
                    concept_id=concept_id,
                    sessions=session_ids,
                    config=tr_cfg,
                    force=True,
                    progress_cb=lambda msg: _emit("inference", 4, 0.5, msg),
                )
            except Exception as exc:
                logger.warning("Dense inference failed: %s", exc)
            state.step_timings["inference"] = time.monotonic() - step_start
            state.completed_steps.append("inference")
            _emit("inference", 4, 1.0, "Inference complete.")

            # ── Step 6: Temporal Refinement (per-behaviour postprocess)
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}

            # Resolve per-behaviour postprocess thresholds from
            # temporal_review_settings (user-tuned) falling back to
            # temporal_refinement_settings, then to defaults.
            review = snapshot.temporal_review_settings or {}
            review_all = review.get("__all__") or {}
            review_by_beh = review.get("by_behavior") or {}

            active_behavior_ids = [
                bid for bid in behavior_models
                if bid not in excluded_ids
            ]

            # Propagate the competition inference_dir into each behavior's
            # latest.json so that run_temporal_refinement_postprocess can find
            # the probability traces (inference was run as "target_behavior",
            # not per-behavior).
            tb_latest_path = (
                target_project_root / "derived" / "temporal_refinement"
                / "target_behavior" / "latest.json"
            )
            if tb_latest_path.exists():
                tb_latest = read_json(tb_latest_path, {})
                competition_inference_dir = str(
                    tb_latest.get("inference_dir", "") or ""
                ).strip()
                if competition_inference_dir:
                    for _bid in active_behavior_ids:
                        token = self._safe_name(_bid)
                        bid_latest_path = (
                            target_project_root / "derived" / "temporal_refinement"
                            / token / "latest.json"
                        )
                        bid_latest_path.parent.mkdir(parents=True, exist_ok=True)
                        bid_latest = read_json(bid_latest_path, {}) if bid_latest_path.exists() else {}
                        bid_latest["inference_dir"] = competition_inference_dir
                        write_json(bid_latest_path, bid_latest)

            total_beh = max(len(active_behavior_ids), 1)
            for beh_idx, bid in enumerate(active_behavior_ids):
                if self._cancelled:
                    return {"status": "cancelled"}
                beh_name = self._behavior_display_name(bid, snapshot)
                pct = beh_idx / total_beh
                _emit(
                    "temporal_refinement", 5, pct,
                    f"Post-processing {beh_name} ({beh_idx + 1}/{total_beh})…",
                )
                # Merge threshold hierarchy: defaults ← __all__ ← per-behavior
                beh_review = review_by_beh.get(bid) or {}
                beh_tr = tr_by_beh.get(bid) or {}

                onset = float(
                    beh_review.get("onset_threshold",
                    beh_tr.get("onset_threshold",
                    review_all.get("onset_threshold", 0.5)))
                )
                min_bout = int(
                    beh_review.get("min_bout_duration_frames",
                    beh_tr.get("min_bout_duration_frames",
                    review_all.get("min_bout_duration_frames", 6)))
                )
                merge_gap = int(
                    beh_review.get("merge_gap_frames",
                    beh_tr.get("merge_gap_frames",
                    review_all.get("merge_gap_frames", 3)))
                )

                pp_cfg = TemporalRefinementConfig(
                    onset_threshold=onset,
                    min_bout_duration_frames=min_bout,
                    merge_gap_frames=merge_gap,
                )

                try:
                    tr_service.run_temporal_refinement_postprocess(
                        concept_id=bid,
                        sessions=session_ids,
                        config=pp_cfg,
                        force=True,
                        progress_cb=lambda msg: _emit(
                            "temporal_refinement", 5,
                            (beh_idx + 0.5) / total_beh, msg,
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "Temporal postprocess failed for %s: %s", bid, exc,
                    )

            state.step_timings["temporal_refinement"] = time.monotonic() - step_start
            state.completed_steps.append("temporal_refinement")
            _emit("temporal_refinement", 5, 1.0, "Temporal refinement complete.")

            # ── Step 7: Export bouts for ALL behaviours ──────────────
            step_start = time.monotonic()
            if self._cancelled:
                return {"status": "cancelled"}
            _emit("bout_extraction", 6, 0.0, "Exporting behavior bouts…")
            bout_count = self._export_behavior_bouts(
                target_project_root, active_behavior_ids, snapshot,
            )
            state.step_timings["bout_extraction"] = time.monotonic() - step_start
            state.completed_steps.append("bout_extraction")
            _emit("bout_extraction", 6, 1.0, f"Extracted {bout_count} bouts total.")

            # ── Done ─────────────────────────────────────────────────
            state.is_complete = True
            state.elapsed_seconds = time.monotonic() - pipeline_start
            state.estimated_remaining_seconds = 0.0
            if progress_cb:
                progress_cb(state)

            return {
                "status": "success",
                "session_count": total_sessions,
                "bout_count": bout_count,
                "elapsed_seconds": state.elapsed_seconds,
                "step_timings": dict(state.step_timings),
            }

        except Exception as exc:
            state.error = str(exc)
            if progress_cb:
                progress_cb(state)
            logger.exception("Direct run pipeline failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _export_behavior_bouts(
        self,
        project_root: Path,
        behavior_ids: list[str],
        snapshot: WorkflowSnapshot,
    ) -> int:
        """Collect bout parquets from per-behaviour TR output into derived/behavior_bouts/.

        Iterates over every behaviour that was postprocessed and gathers the
        bout DataFrames into a combined parquet per behaviour and an overall
        merged file.
        """
        tr_root = project_root / "derived" / "temporal_refinement"
        bouts_dir = project_root / "derived" / "behavior_bouts"
        bouts_dir.mkdir(parents=True, exist_ok=True)
        total_bouts = 0
        all_dfs: list[pd.DataFrame] = []

        # Only per-behaviour postprocess output is a valid bout source. The
        # "target_behavior" group-level postprocess thresholds the generic
        # max-over-all-behaviors probability trace, so its bouts represent
        # "any behavior active" rather than a single behavior — folding them in
        # would inflate the total count and emit a misleading
        # target_behavior_bouts.parquet.
        search_ids = list(behavior_ids)
        seen_files: set[str] = set()

        for bid in search_ids:
            token = self._safe_name(bid)
            latest_path = tr_root / token / "latest.json"
            if not latest_path.exists():
                continue
            try:
                latest = read_json(latest_path, {})
                post_dir = str(latest.get("postprocess_dir", "") or "").strip()
                if not post_dir:
                    continue
                pm_path = Path(post_dir) / "postprocess_manifest.json"
                if not pm_path.exists():
                    continue
                pm = read_json(pm_path, {})
                bout_paths = {
                    str(k): str(v)
                    for k, v in (pm.get("bout_paths", {}) or {}).items()
                }
                bid_bouts: list[pd.DataFrame] = []
                for sid, bp in bout_paths.items():
                    bp_key = str(Path(bp).resolve())
                    if bp_key in seen_files:
                        continue
                    seen_files.add(bp_key)
                    if Path(bp).exists():
                        try:
                            df = pd.read_parquet(Path(bp))
                            if not df.empty:
                                if "session_id" not in df.columns:
                                    df["session_id"] = sid
                                if "behavior_id" not in df.columns:
                                    df["behavior_id"] = bid
                                bid_bouts.append(df)
                                total_bouts += len(df)
                        except Exception:
                            pass
                if bid_bouts:
                    merged = pd.concat(bid_bouts, ignore_index=True)
                    merged.to_parquet(
                        bouts_dir / f"{token}_bouts.parquet", index=False,
                    )
                    all_dfs.append(merged)
            except Exception:
                continue

        # Write combined file for all behaviours
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_parquet(
                bouts_dir / "all_behaviors_bouts.parquet", index=False,
            )

        return total_bouts

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _carry_feature_settings(
        source_project_root: Path, target_project_root: Path,
    ) -> None:
        """Mirror the source's feature-selection config into the target project.

        Copies the two config files the Features tab writes besides
        project.yaml:

        * ``config/feature_exclusions.json`` — disabled feature groups
          (per-keypoint kinematics, global movement, oscillation, orientation).
        * ``config/experiment.yaml`` → ``behavior_model.invariant_features`` —
          the robustness toggles (egocentric, body-length norm, relative
          geometry, head direction, joint angles, spine curvature, clip deltas).

        Without this, the new project opens with all of these reverted to their
        defaults rather than the values the source model was configured with.
        """
        src_cfg = source_project_root / "config"
        dst_cfg = target_project_root / "config"
        dst_cfg.mkdir(parents=True, exist_ok=True)

        # feature_exclusions.json — copy wholesale (it is purely feature
        # selection state) when the target doesn't already have one.
        try:
            src_excl = src_cfg / "feature_exclusions.json"
            dst_excl = dst_cfg / "feature_exclusions.json"
            if src_excl.exists() and not dst_excl.exists():
                shutil.copy2(src_excl, dst_excl)
        except Exception as exc:
            logger.warning("Could not carry feature_exclusions.json: %s", exc)

        # experiment.yaml — merge only the invariant_features block so we don't
        # drag along unrelated, possibly project-specific experiment config.
        try:
            src_exp = src_cfg / "experiment.yaml"
            if src_exp.exists():
                src_data = read_yaml(src_exp, {}) or {}
                inv = (src_data.get("behavior_model") or {}).get("invariant_features")
                if isinstance(inv, dict) and inv:
                    dst_exp = dst_cfg / "experiment.yaml"
                    dst_data = read_yaml(dst_exp, {}) or {} if dst_exp.exists() else {}
                    dst_data.setdefault("behavior_model", {})["invariant_features"] = dict(inv)
                    write_yaml(dst_exp, dst_data)
        except Exception as exc:
            logger.warning("Could not carry invariant_features: %s", exc)

    def _behavior_display_name(
        self, bid: str, snapshot: WorkflowSnapshot,
    ) -> str:
        """Look up a human-readable name for a behaviour ID."""
        for b in snapshot.behavior_definitions:
            b_id = b.get("behavior_id", b.get("name", ""))
            if b_id == bid:
                return b.get("name", b.get("short_name", bid))
        return bid

    @staticmethod
    def _resolve_use_video_features(
        snapshot: WorkflowSnapshot, source_project_root: Path,
    ) -> bool:
        """Decide whether to recompute context/video features.

        Honours the snapshot flag first.  For older snapshots created before
        that flag existed (default False), fall back to evidence from the
        source project — the model's ``run_settings.json`` or an existing
        ``frame_context.parquet`` — so a video-trained model never silently
        runs without its context features.
        """
        if bool(getattr(snapshot, "use_video_features", False)):
            return True
        # An explicit "on" captured from the source's Features tab is a
        # deliberate user choice — honour it even when the snapshot's own flag
        # was lost.  Without this, a video-trained project whose model
        # run_settings.json omits the key gets silently downgraded to
        # pose-only, and the new project runs with no video/context features.
        fx = getattr(snapshot, "feature_extraction_settings", None) or {}
        if bool(fx.get("use_video_features")):
            return True
        # Backward-compat: snapshot predates the flag.
        trained = WorkflowSnapshotService()._trained_use_video_features(
            source_project_root, snapshot.model_version,
        )
        if trained is not None:
            return trained
        # Fall back to the source project.yaml's Features-tab settings.
        try:
            src_proj = read_yaml(source_project_root / "project.yaml", {}) or {}
            if bool((src_proj.get("feature_extraction") or {}).get("use_video_features")):
                return True
            if bool((src_proj.get("behavior_model") or {}).get("use_video_features")):
                return True
        except Exception:
            pass
        # Last resort: the source project computed context features at all.
        return (
            source_project_root / "derived" / "context_features"
            / "frame_context.parquet"
        ).exists()

    @staticmethod
    def _build_context_config(snapshot: WorkflowSnapshot):
        """Build a ContextFeatureConfig from the snapshot, ignoring any keys
        that are not valid dataclass fields (e.g. a legacy use_video_features)."""
        from abel.services.context_feature_service import ContextFeatureConfig
        import dataclasses  # noqa: PLC0415
        cfg_dict = dict(snapshot.context_feature_config or {})
        valid = {f.name for f in dataclasses.fields(ContextFeatureConfig)}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid}
        return ContextFeatureConfig(**filtered)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(value).strip()
        ) or "target_behavior"
