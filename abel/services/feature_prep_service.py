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


@dataclass
class PrepConfig:
    use_video_features: bool = False
    flow_temporal_stride: int = 10
    segment_window_frames: int = 60
    segment_stride_frames: int = 15
    excluded_feature_cols: frozenset[str] = frozenset()
    reuse_cached: bool = True


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

    # ── Keypoint-rename cache invalidation ────────────────────────────
    # The per-session pose/context caches embed the body-part names that were
    # in force when they were built.  When the user renames body parts (Data
    # Import → Rename Body Parts), those caches become stale, so we record the
    # alias map each build ran under and force a rebuild when it changes —
    # otherwise re-running feature extraction would silently reuse the old
    # names.
    @staticmethod
    def _alias_signature(aliases: dict[str, str]) -> str:
        payload = json.dumps(aliases or {}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _alias_sig_path(project_root: Path) -> Path:
        return (
            project_root / "derived" / "pose_features"
            / ".keypoint_alias_signature.json"
        )

    @classmethod
    def _aliases_changed(cls, project_root: Path, aliases: dict[str, str]) -> bool:
        """True when the cached features were built under a different rename map.

        Returns ``False`` when no signature has been recorded yet (a project
        built before this guard existed) so existing caches aren't needlessly
        discarded — the signature is written after the next build.
        """
        path = cls._alias_sig_path(project_root)
        if not path.exists():
            return False
        prev = read_json(path, {}) or {}
        return str(prev.get("signature", "")) != cls._alias_signature(aliases)

    @classmethod
    def _write_alias_signature(cls, project_root: Path, aliases: dict[str, str]) -> None:
        path = cls._alias_sig_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"signature": cls._alias_signature(aliases)})

    @classmethod
    def invalidate_caches(cls, project_root: Path) -> None:
        """Mark the feature caches stale so the next ``prepare`` rebuilds them.

        Called when body parts are renamed.  Writes a sentinel signature that
        can never equal a real one, which forces a full re-extraction under the
        new names without destroying any existing files up front.
        """
        path = cls._alias_sig_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"signature": "stale"})

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
        # A body-part rename invalidates every cached pose/context parquet (the
        # column names changed), so ignore the cache when the alias map differs
        # from the one the caches were built under.
        aliases_changed = self._aliases_changed(project_root, aliases)
        reuse = config.reuse_cached and not aliases_changed
        if aliases_changed:
            obs.log(
                "Body-part renames changed since the last build — rebuilding all "
                "feature caches so the new names apply."
            )
        cached_pose = self.cached_pose_sessions(project_root) if reuse else set()
        cached_ctx = self.cached_context_sessions(project_root) if reuse else set()

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
            obs.stage_start(STAGE_PREPROCESS, f"Extract {feat_label}", len(pending))
            t0 = time.monotonic()
            self._extract_sessions(
                project_root, pending, config, aliases, obs, result, _check_cancel
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

        # Record the rename map these caches were built under so a later rename
        # is detected and triggers a rebuild.
        self._write_alias_signature(project_root, aliases)

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
    ) -> None:
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

        def _collect_warning(msg: str) -> None:
            with warn_lock:
                result.gpu_warnings.append(msg)

        def _process_one(job: SessionJob) -> str:
            self._pose.extract_and_save_frame_pose_features(
                project_root=project_root,
                pose_path=job.pose_path,
                fps=job.fps,
                animal_id=job.subject_id,
                session_id=job.session_id,
                video_id=job.session_id,
                keypoint_aliases=aliases,
            )
            if config.use_video_features and job.video_path is not None:
                ContextFeatureService().compute_frame_context(
                    project_root=project_root,
                    video_path=job.video_path,
                    pose_path=job.pose_path,
                    animal_id=job.subject_id,
                    session_id=job.session_id,
                    config=ContextFeatureConfig(
                        flow_temporal_stride=int(config.flow_temporal_stride)
                    ),
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
