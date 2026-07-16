"""Shared, Qt-free preparation of the cacheable Active-Learning inputs.

The supervised pipeline needs three derived artefacts before any model can be
trained:

1. **frame-pose features** (parquet, per session) — :class:`PoseProcessingService`
2. **frame-context features** (optical flow etc., only when video is enabled) —
   :class:`ContextFeatureService`
3. **frame/segment representations** built from (1)+(2) —
   :class:`BehaviorRepresentationService`

Historically all three were produced inside the Active Learning tab's pipeline,
so the first training run paid the full (often multi-minute) cost.  This service
extracts that work into one reusable place so it can be **pre-built during
feature extraction** (the Features tab) and merely *consumed* by Active Learning.

Everything here is content/config-cached: a session whose per-session parquet
already exists is skipped, and the representation build is keyed on a content +
config signature, so re-running is cheap and only genuinely-changed inputs (new
clips, changed window/stride) trigger a rebuild.

The module is intentionally free of any Qt dependency so it is unit-testable and
can run headless.  Progress is reported through the small :class:`PrepObserver`
protocol; the GUI binds it to a live timeline/ETA panel.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from abel.services.behavior_representation_service import (
    BehaviorRepresentationService,
    RepresentationConfig,
)
from abel.services.context_feature_service import (
    ContextFeatureConfig,
    ContextFeatureService,
)
from abel.services.pose_processing_service import PoseProcessingService
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")

# Stage keys — shared with the UI timeline so labels/ETA line up across tabs.
STAGE_PREPROCESS = "preprocess"
STAGE_CONSOLIDATE = "consolidate"
STAGE_REPRESENTATIONS = "representations"


class PrepCancelledError(RuntimeError):
    """Raised when a caller's cancel flag is set mid-prep."""


class PrepInputError(RuntimeError):
    """Raised up front when session inputs (pose/video files) are unreadable."""


class PrepObserver(Protocol):
    """Sink for structured progress.  All methods are optional no-ops."""

    def stage_start(self, key: str, label: str, total_units: int) -> None: ...
    def stage_advance(self, key: str, done_units: int, message: str) -> None: ...
    def stage_done(self, key: str) -> None: ...
    def stage_skip(self, key: str, message: str) -> None: ...
    def log(self, message: str) -> None: ...


class _NullObserver:
    def stage_start(self, key: str, label: str, total_units: int) -> None: ...
    def stage_advance(self, key: str, done_units: int, message: str) -> None: ...
    def stage_done(self, key: str) -> None: ...
    def stage_skip(self, key: str, message: str) -> None: ...
    def log(self, message: str) -> None: ...


@dataclass
class SessionJob:
    session_id: str
    subject_id: str
    pose_path: Path
    video_path: Path | None
    fps: float
    individuals: list[str] = field(default_factory=list)
    """Detected individuals in a multi-animal pose file.  Empty ⇒ legacy
    single-animal extraction path."""
    individual_subject_map: dict[str, str] = field(default_factory=dict)
    """Maps each individual to a real subject identity used as its ``animal_id``."""
    identity_corrections: list[dict] = field(default_factory=list)
    """Identity-swap corrections applied on load (see LinkedSession)."""


@dataclass
class PrepConfig:
    use_video_features: bool = False
    flow_temporal_stride: int = 10
    segment_window_frames: int = 60
    segment_stride_frames: int = 15
    excluded_feature_cols: frozenset[str] = frozenset()
    reuse_cached: bool = True
    advanced_roi_features: bool = True


@dataclass
class WorkerPlan:
    max_workers: int
    intra_session_workers: int
    source: str


@dataclass
class PrepResult:
    n_sessions_processed: int = 0
    n_sessions_reused: int = 0
    n_sessions_skipped: int = 0
    frame_pose_path: Path | None = None
    frame_context_path: Path | None = None
    n_frame_rows: int = 0
    n_segment_rows: int = 0
    gpu_warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def plan_session_workers(
    n_jobs: int,
    *,
    gpu_info: dict | None,
    cpu_count: int | None = None,
    env_workers: str | None = None,
) -> WorkerPlan:
    """Decide session/intra-session worker counts (pure, testable).

    Mirrors the Active Learning pipeline's GPU-aware policy: when GPU optical
    flow is active every session worker serialises on a single GPU lock, so
    spawning one-per-core is wasteful — we cap parallelism to what the hardware
    can actually sustain and spread the remaining cores across intra-session
    frame chunks.
    """
    n_jobs = max(1, int(n_jobs))
    cpu_cap = max(1, (cpu_count if cpu_count is not None else (os.cpu_count() or 1)) - 1)

    requested = 0
    if env_workers:
        try:
            requested = max(1, int(env_workers))
        except ValueError:
            requested = 0

    gpu_info = gpu_info or {}
    gpu_total_mb = float(gpu_info.get("total_mb", 0) or 0)
    gpu_name = gpu_info.get("name", "(none)")
    gpu_backend = gpu_info.get("backend", "cpu")
    uses_gpu_flow = gpu_backend in ("torch", "cv2_cuda")

    if not requested and uses_gpu_flow:
        if gpu_total_mb > 0:
            if gpu_total_mb <= 2048:
                cap = 1
            elif gpu_total_mb <= 4096:
                cap = 2
            elif gpu_total_mb <= 8192:
                cap = min(4, cpu_cap)
            elif gpu_total_mb <= 12288:
                cap = min(6, cpu_cap)
            else:
                cap = min(8, cpu_cap)
            source = f"GPU-adaptive ({gpu_name}, {gpu_total_mb:.0f} MB → {cap} session cap)"
        else:
            cap = min(2, cpu_cap)
            source = f"GPU-safe fallback ({gpu_name}, VRAM unknown → {cap} session cap)"
        max_workers = min(n_jobs, cap)
    elif requested:
        max_workers = min(n_jobs, requested)
        source = "environment override"
    else:
        max_workers = min(n_jobs, cpu_cap)
        source = f"auto (cpu_count-1={cpu_cap})"

    # Ceiling division so no core is left idle to rounding.
    intra = max(1, -(-cpu_cap // max(1, max_workers)))
    return WorkerPlan(max_workers=max_workers, intra_session_workers=intra, source=source)


class FeaturePrepService:
    """Build & cache pose/context/representation artefacts for a project."""

    def __init__(self) -> None:
        self._pose = PoseProcessingService()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def keypoint_aliases(project_root: Path) -> dict[str, str]:
        data = read_json(project_root / "config" / "keypoint_aliases.json", {})
        return {str(k): str(v) for k, v in data.items() if str(k) and str(v)}

    # ── Feature-cache invalidation ────────────────────────────────────
    # The per-session caches embed the inputs they were built from, so they go
    # stale when those inputs change and must be rebuilt — otherwise re-running
    # feature extraction silently reuses the old result:
    #   • pose features depend on the body-part rename map.
    #   • context features additionally depend on the ROI config (target zones,
    #     subject crop, local-motion radius) and the flow stride.
    # We record a signature for each so a change to either is detected and only
    # the affected cache is rebuilt.
    # Bump when the pose feature *schema* (column names / formulas) changes so
    # existing caches are rebuilt into the new, cross-project-compatible format.
    #   v2: canonical (order-independent, sorted) pairwise-distance column names.
    #   v3: optional multi-animal interaction (social_*) columns.
    #   v4: social heading-alignment, directed radial-velocity-toward, and
    #       contact-state (in_contact + duration) columns.
    _POSE_SCHEMA_VERSION = "4"

    @staticmethod
    def _hash(obj: object) -> str:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _alias_signature(cls, aliases: dict[str, str]) -> str:
        return cls._hash(aliases or {})

    @classmethod
    def _pose_signature(cls, project_root: Path, aliases: dict[str, str]) -> str:
        # Fold the social-feature toggle in so enabling/disabling interaction
        # features rebuilds the pose cache (the column set changes), while solo
        # single-animal projects keep the same signature regardless of the flag.
        social = False
        try:
            from abel.models.schemas import InvariantFeatureConfig  # noqa: PLC0415
            social = bool(InvariantFeatureConfig.load_from_project(project_root).enable_social_features)
        except Exception:
            social = False
        return cls._hash({
            "v": cls._POSE_SCHEMA_VERSION,
            "aliases": aliases or {},
            "social": social,
        })

    @classmethod
    def _context_signature(
        cls, project_root: Path, aliases: dict[str, str], config: "PrepConfig",
    ) -> str:
        roi_path = project_root / "config" / "environment_rois.yaml"
        try:
            roi_blob = roi_path.read_text(encoding="utf-8") if roi_path.exists() else ""
        except Exception:
            roi_blob = ""
        return cls._hash({
            "aliases": aliases or {},
            "roi": roi_blob,
            "flow_temporal_stride": int(getattr(config, "flow_temporal_stride", 0) or 0),
            # Toggling advanced ROI features changes the context column set, so
            # it must invalidate the cache — otherwise enabling it would leave
            # the new columns missing until an unrelated ROI edit forced a rebuild.
            "advanced_roi": bool(getattr(config, "advanced_roi_features", True)),
        })

    @staticmethod
    def _alias_sig_path(project_root: Path) -> Path:
        return (
            project_root / "derived" / "pose_features"
            / ".keypoint_alias_signature.json"
        )

    @classmethod
    def _pose_changed(cls, project_root: Path, aliases: dict[str, str]) -> bool:
        """True when the cached pose features were built from a different rename map."""
        # Content check first: if the cached features still carry a body-part
        # name that the current aliases should have renamed, they were built
        # before the rename map was set and are stale — regardless of any
        # recorded signature.  This is the case where aliases were added after
        # the features were first extracted.
        if cls._pose_cache_has_unapplied_aliases(project_root, aliases):
            return True
        path = cls._alias_sig_path(project_root)
        if not path.exists():
            # No record of how the cache was built.  If a pose cache already
            # exists it predates schema tracking and may use the old feature
            # format (e.g. order-dependent distance names), so rebuild once to
            # guarantee a consistent, cross-project-compatible schema.  If there
            # is no cache yet, there's nothing to rebuild.
            return bool(cls.cached_pose_sessions(project_root))
        prev = read_json(path, {}) or {}
        # ``signature`` is the legacy (pose-only) key; it lacks the schema
        # version, so a legacy project's stored value won't match the current
        # signature and will (correctly) trigger a one-time rebuild.
        stored = prev.get("pose", prev.get("signature", ""))
        return str(stored) != cls._pose_signature(project_root, aliases)

    @classmethod
    def _pose_cache_has_unapplied_aliases(
        cls, project_root: Path, aliases: dict[str, str],
    ) -> bool:
        """True when ``frame_pose.parquet`` still contains a keypoint that the
        current alias map renames (i.e. the rename was never applied)."""
        if not aliases:
            return False
        fp = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if not fp.exists():
            return False
        try:
            import pyarrow.parquet as pq
            cols = set(pq.read_schema(fp).names)
        except Exception:
            return False
        suffix = "_velocity_x"
        kps = {c[: -len(suffix)] for c in cols if c.endswith(suffix)}
        from abel.services.pose_processing_service import normalize_bodypart_name
        norm_kps = {normalize_bodypart_name(k) for k in kps}
        for src, dst in aliases.items():
            ns, nd = normalize_bodypart_name(src), normalize_bodypart_name(dst)
            if ns != nd and ns in norm_kps:
                return True
        return False

    @classmethod
    def _context_changed(
        cls, project_root: Path, aliases: dict[str, str], config: "PrepConfig",
    ) -> bool:
        """True when cached context features were built from different ROIs/renames."""
        path = cls._alias_sig_path(project_root)
        if not path.exists():
            return False
        prev = read_json(path, {}) or {}
        if "context" not in prev:
            # Legacy signature predates context tracking.  Fall back to mtimes:
            # the context cache is stale if the ROI config is newer than it.
            return cls._roi_newer_than_context_cache(project_root)
        return str(prev.get("context", "")) != cls._context_signature(
            project_root, aliases, config
        )

    @staticmethod
    def _roi_newer_than_context_cache(project_root: Path) -> bool:
        roi_path = project_root / "config" / "environment_rois.yaml"
        ctx_dir = project_root / "derived" / "context_features" / "sessions"
        if not roi_path.exists() or not ctx_dir.exists():
            return False
        ctx_files = list(ctx_dir.glob("*.parquet"))
        if not ctx_files:
            return False
        newest_ctx = max(f.stat().st_mtime for f in ctx_files)
        return roi_path.stat().st_mtime > newest_ctx

    @classmethod
    def _write_signatures(
        cls, project_root: Path, aliases: dict[str, str], config: "PrepConfig",
    ) -> None:
        path = cls._alias_sig_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {
            "pose": cls._pose_signature(project_root, aliases),
            "context": cls._context_signature(project_root, aliases, config),
        })

    @classmethod
    def invalidate_caches(cls, project_root: Path) -> None:
        """Mark the feature caches stale so the next ``prepare`` rebuilds them.

        Called when body parts are renamed (which affects both pose and context
        features).  Writes sentinel signatures that can never equal a real one,
        forcing a full re-extraction without destroying any files up front.
        """
        path = cls._alias_sig_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"pose": "stale", "context": "stale"})

    # Directories under ``derived/`` that hold *only* generated feature caches.
    # Deleting them forces the next ``prepare`` to rebuild every stage from the
    # source pose/video — the nuclear option when stale caches are being reused.
    _CACHE_DIRS = ("pose_features", "context_features", "representations")

    @classmethod
    def clear_feature_caches(cls, project_root: Path) -> dict[str, object]:
        """Delete all cached feature artefacts so the next run rebuilds from scratch.

        Removes the pose-feature (.npz kinematic windows, per-session and
        monolithic parquet, summaries, alias signature), context-feature, and
        representation caches.  Source pose/video files and project config are
        untouched.  Returns a summary dict with the directories removed, the
        number of files deleted, and total bytes freed.
        """
        import shutil  # noqa: PLC0415

        derived = project_root / "derived"
        removed: list[str] = []
        n_files = 0
        n_bytes = 0
        for name in cls._CACHE_DIRS:
            d = derived / name
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file():
                    n_files += 1
                    try:
                        n_bytes += f.stat().st_size
                    except OSError:
                        pass
            try:
                shutil.rmtree(d)
                removed.append(name)
            except OSError as exc:
                logger.warning("Failed to remove cache dir %s: %s", d, exc)
        return {"removed": removed, "n_files": n_files, "n_bytes": n_bytes}

    @staticmethod
    def _parquet_rows(path: Path) -> int:
        """Row count from the parquet footer only (no data read)."""
        if not path.exists():
            return 0
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            try:
                import pandas as pd
                return int(len(pd.read_parquet(path, columns=[])))
            except Exception:
                return 0

    @staticmethod
    def cached_pose_sessions(project_root: Path) -> set[str]:
        d = project_root / "derived" / "pose_features" / "sessions"
        return {f.stem for f in d.glob("*.parquet")} if d.exists() else set()

    @staticmethod
    def cached_context_sessions(project_root: Path) -> set[str]:
        d = project_root / "derived" / "context_features" / "sessions"
        return {f.stem for f in d.glob("*.parquet")} if d.exists() else set()

    @staticmethod
    def _preflight_inputs(jobs: list[SessionJob], config: PrepConfig) -> None:
        """Fail fast when a session's source files are gone.

        Extraction opens each pose/video file deep inside a worker thread, so a
        single missing file used to surface as an opaque mid-run crash after the
        other sessions had already been processed.  Videos in particular are the
        common casualty: projects registered against an external drive keep
        working until the cache is cleared, because pose features only ever read
        the (project-local) pose file.  Check everything before doing any work
        and report the whole list at once.
        """
        missing: list[str] = []
        for job in jobs:
            label = f"{job.subject_id or '?'} ({job.session_id})"
            if not job.pose_path or not Path(job.pose_path).exists():
                missing.append(f"  {label}: pose file not found — {job.pose_path}")
            if config.use_video_features:
                if not job.video_path:
                    missing.append(f"  {label}: no video linked (video context is ON)")
                elif not Path(job.video_path).exists():
                    missing.append(f"  {label}: video not found — {job.video_path}")
        if not missing:
            return
        raise PrepInputError(
            f"{len(missing)} session input file(s) are missing, so feature "
            f"preparation cannot run:\n" + "\n".join(missing) +
            "\n\nRe-link or restore these files (Data Import → session list), or "
            "deselect the affected sessions. If the videos live on an external "
            "drive, make sure it is connected."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        project_root: Path,
        jobs: list[SessionJob],
        config: PrepConfig,
        *,
        observer: PrepObserver | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> PrepResult:
        """Ensure pose/context/representation caches exist for ``jobs``.

        Sessions whose per-session feature parquet already exists are skipped
        when ``config.reuse_cached`` is set.  Returns a :class:`PrepResult` with
        counts, output paths and per-stage timings.
        """
        obs: PrepObserver = observer or _NullObserver()
        result = PrepResult()
        aliases = self.keypoint_aliases(project_root)

        def _check_cancel() -> None:
            if cancel_flag and cancel_flag[0]:
                raise PrepCancelledError("PREP_CANCELLED_BY_USER")

        # ── Decide which sessions actually need (re)building ──────────────
        # Pose features go stale when body parts are renamed; context features
        # additionally go stale when the ROI config changes.  Discard only the
        # cache whose inputs changed so an ROI edit doesn't force a (needless)
        # full pose re-extraction, and vice-versa.
        pose_changed = self._pose_changed(project_root, aliases)
        ctx_changed = self._context_changed(project_root, aliases, config)
        reuse = config.reuse_cached
        if reuse and pose_changed:
            obs.log(
                "Pose feature inputs changed (body-part renames or an updated "
                "feature format) — rebuilding pose features for a consistent, "
                "cross-project-compatible schema."
            )
        if reuse and ctx_changed:
            obs.log("ROI configuration changed — rebuilding context features for the new ROIs.")
        cached_pose = (
            self.cached_pose_sessions(project_root) if reuse and not pose_changed else set()
        )
        cached_ctx = (
            self.cached_context_sessions(project_root) if reuse and not ctx_changed else set()
        )

        pending: list[SessionJob] = []
        for job in jobs:
            sid = str(job.session_id)
            pose_ok = sid in cached_pose
            ctx_ok = (not config.use_video_features) or sid in cached_ctx
            if pose_ok and ctx_ok:
                result.n_sessions_reused += 1
            else:
                pending.append(job)

        # ── Stage 1: per-session pose (+ context) extraction ─────────────
        feat_label = "pose + context features" if config.use_video_features else "pose features"
        if pending:
            self._preflight_inputs(pending, config)
            obs.stage_start(STAGE_PREPROCESS, f"Extract {feat_label}", len(pending))
            t0 = time.monotonic()
            self._extract_sessions(
                project_root, pending, config, aliases, obs, result, _check_cancel,
                cached_pose=cached_pose, cached_ctx=cached_ctx,
            )
            result.timings["preprocess"] = time.monotonic() - t0
            obs.stage_done(STAGE_PREPROCESS)
        else:
            obs.stage_skip(STAGE_PREPROCESS, f"All {len(jobs)} session(s) already cached.")

        _check_cancel()

        # ── Stage 2: consolidate per-session parquet into monolithic ─────
        obs.stage_start(STAGE_CONSOLIDATE, "Consolidate feature caches", 1)
        t0 = time.monotonic()
        PoseProcessingService.consolidate_session_files(project_root)
        if config.use_video_features:
            ContextFeatureService.consolidate_session_files(project_root)
        # Guarantee the monolithic files exist even when every session was cached.
        pose_mono = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if not pose_mono.exists():
            PoseProcessingService.consolidate_session_files(project_root)
        ctx_mono = project_root / "derived" / "context_features" / "frame_context.parquet"
        if config.use_video_features and not ctx_mono.exists():
            ContextFeatureService.consolidate_session_files(project_root)
        result.timings["consolidate"] = time.monotonic() - t0
        obs.stage_advance(STAGE_CONSOLIDATE, 1, "Feature caches consolidated.")
        obs.stage_done(STAGE_CONSOLIDATE)

        result.frame_pose_path = pose_mono
        # Only treat context as available when the monolithic file actually
        # exists — e.g. video was enabled but no selected session had a video.
        result.frame_context_path = (
            ctx_mono if (config.use_video_features and ctx_mono.exists()) else None
        )

        _check_cancel()

        # ── Stage 3: build (or reuse) frame/segment representations ───────
        obs.stage_start(STAGE_REPRESENTATIONS, "Build representations", 1)
        t0 = time.monotonic()
        repr_svc = BehaviorRepresentationService()
        # Build the full project-wide cache (session_ids=None).  ``ensure_only``
        # avoids re-reading the multi-GB cache just to hand back dataframes the
        # prep step does not need — we read row counts from the parquet footers.
        frame_df, segment_df = repr_svc.build(
            project_root=project_root,
            frame_pose_path=pose_mono,
            frame_context_path=result.frame_context_path,
            config=RepresentationConfig(
                window_size_frames=int(config.segment_window_frames),
                window_stride_frames=int(config.segment_stride_frames),
                excluded_feature_cols=frozenset(config.excluded_feature_cols),
            ),
            session_ids=None,
            progress_cb=lambda msg: obs.log(msg),
            ensure_only=True,
        )
        repr_dir = project_root / "derived" / "representations"
        result.n_frame_rows = (
            int(len(frame_df)) if len(frame_df)
            else self._parquet_rows(repr_dir / "frame_features.parquet")
        )
        result.n_segment_rows = (
            int(len(segment_df)) if len(segment_df)
            else self._parquet_rows(repr_dir / "segment_features.parquet")
        )
        result.timings["representations"] = time.monotonic() - t0
        obs.stage_advance(
            STAGE_REPRESENTATIONS, 1,
            f"Representations ready: {result.n_segment_rows} segment row(s).",
        )
        obs.stage_done(STAGE_REPRESENTATIONS)

        # Record the inputs these caches were built from so a later rename or
        # ROI edit is detected and triggers a targeted rebuild.
        self._write_signatures(project_root, aliases, config)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_sessions(
        self,
        project_root: Path,
        jobs: list[SessionJob],
        config: PrepConfig,
        aliases: dict[str, str],
        obs: PrepObserver,
        result: PrepResult,
        check_cancel,
        *,
        cached_pose: set[str] | None = None,
        cached_ctx: set[str] | None = None,
    ) -> None:
        # A session lands here when *either* its pose or context cache is stale.
        # Skip the part that is still valid so e.g. an ROI-only change rebuilds
        # context without re-running the (unaffected) pose extraction.
        cached_pose = cached_pose or set()
        cached_ctx = cached_ctx or set()
        gpu_info: dict = {}
        try:
            from abel.utils.gpu_optical_flow import gpu_summary
            gpu_info = gpu_summary()
        except Exception as exc:  # pragma: no cover - probe is best-effort
            logger.warning("GPU summary probe failed: %s", exc)

        plan = plan_session_workers(
            len(jobs),
            gpu_info=gpu_info,
            env_workers=os.environ.get("ABEL_PREPROCESS_WORKERS", "").strip() or None,
        )
        obs.log(
            f"Preprocessing {len(jobs)} session(s): {plan.max_workers} session worker(s) "
            f"× {plan.intra_session_workers} chunk worker(s) — {plan.source}."
        )

        warn_lock = threading.Lock()

        # Project-level invariance/social toggles.  Loaded once here (not baked
        # into the per-frame default) so multi-animal jobs honor the social flag.
        # The single-animal path keeps its existing default-config behavior.
        from abel.models.schemas import InvariantFeatureConfig  # noqa: PLC0415
        invariant_cfg = InvariantFeatureConfig.load_from_project(project_root)

        def _collect_warning(msg: str) -> None:
            with warn_lock:
                result.gpu_warnings.append(msg)

        def _process_one(job: SessionJob) -> str:
            sid = str(job.session_id)
            # Per-individual animal_id mapping (used by BOTH pose and context so
            # their frame tables share join keys). Empty for single-animal jobs.
            animal_ids = {
                ind: (job.individual_subject_map.get(ind) or f"{job.subject_id or sid}:{ind}")
                for ind in job.individuals
            } if job.individuals else {}
            if sid not in cached_pose:
                if job.individuals:
                    # Multi-animal: one row-set per individual (distinct animal_id),
                    # plus inter-animal social_* columns when enabled.
                    self._pose.extract_and_save_frame_pose_features_multi(
                        project_root=project_root,
                        pose_path=job.pose_path,
                        fps=job.fps,
                        session_id=job.session_id,
                        video_id=job.session_id,
                        individual_animal_ids=animal_ids,
                        invariant_config=invariant_cfg,
                        keypoint_aliases=aliases,
                        enable_social_features=invariant_cfg.enable_social_features,
                        identity_corrections=list(job.identity_corrections or []),
                    )
                else:
                    self._pose.extract_and_save_frame_pose_features(
                        project_root=project_root,
                        pose_path=job.pose_path,
                        fps=job.fps,
                        animal_id=job.subject_id,
                        session_id=job.session_id,
                        video_id=job.session_id,
                        keypoint_aliases=aliases,
                    )
            if (
                config.use_video_features
                and job.video_path is not None
                and sid not in cached_ctx
            ):
                ctx_cfg = ContextFeatureConfig(
                    flow_temporal_stride=int(config.flow_temporal_stride),
                    advanced_roi_features=bool(config.advanced_roi_features),
                )
                if job.individuals:
                    # Per-individual context so animal_id matches the pose table.
                    ContextFeatureService().compute_frame_context_multi(
                        project_root=project_root,
                        video_path=job.video_path,
                        pose_path=job.pose_path,
                        individual_animal_ids=animal_ids,
                        session_id=job.session_id,
                        roi_subject_id=job.subject_id,
                        config=ctx_cfg,
                        intra_session_workers=plan.intra_session_workers,
                        warning_cb=_collect_warning,
                        keypoint_aliases=aliases,
                        identity_corrections=list(job.identity_corrections or []),
                    )
                else:
                    ContextFeatureService().compute_frame_context(
                        project_root=project_root,
                        video_path=job.video_path,
                        pose_path=job.pose_path,
                        animal_id=job.subject_id,
                        session_id=job.session_id,
                        config=ctx_cfg,
                        intra_session_workers=plan.intra_session_workers,
                        warning_cb=_collect_warning,
                        keypoint_aliases=aliases,
                    )
            return str(job.session_id)

        done = 0
        with cf.ThreadPoolExecutor(max_workers=plan.max_workers) as executor:
            futures = {executor.submit(_process_one, job): job for job in jobs}
            try:
                for future in cf.as_completed(futures):
                    check_cancel()
                    sid = future.result()
                    done += 1
                    result.n_sessions_processed += 1
                    obs.stage_advance(
                        STAGE_PREPROCESS, done,
                        f"Processed session {done}/{len(jobs)}: {sid}.",
                    )
            except PrepCancelledError:
                for fut in futures:
                    fut.cancel()
                raise

        if result.gpu_warnings:
            n_oom = sum(1 for w in result.gpu_warnings if "OOM" in w)
            n_timeout = sum(1 for w in result.gpu_warnings if "timed out" in w)
            parts = []
            if n_oom:
                parts.append(f"{n_oom} GPU out-of-memory event(s) — fell back to CPU")
            if n_timeout:
                parts.append(f"{n_timeout} GPU lock timeout(s) — fell back to CPU")
            obs.log("⚠ " + ("; ".join(parts) or f"{len(result.gpu_warnings)} GPU issue(s)") +
                    ". Results are valid (CPU fallback) but slower than expected.")
