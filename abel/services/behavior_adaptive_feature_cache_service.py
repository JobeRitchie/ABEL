"""Phase 1 behavior-adaptive feature caching for benchmark workflows.

This service builds and caches multi-scale segment feature tables from already cached
frame-level pose/context features. It does not modify baseline representation outputs.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from abel.storage.file_store import read_json, write_json


@dataclass
class MultiScaleFeatureCacheConfig:
    scales_sec: list[float]
    fps: float = 30.0
    regenerate: bool = False
    include_window_periodicity: bool = False
    session_ids: list[str] | None = None
    parallel_workers: int = 0
    use_process_pool: bool = True


def _summarize_group_segments_task(
    group_df: pd.DataFrame,
    feature_cols: list[str],
    animal_id: str,
    session_id: str,
    window_frames: int,
    stride_frames: int,
    include_periodicity: bool,
) -> list[dict[str, float | str | int]]:
    work = group_df.sort_values("frame").reset_index(drop=True)
    n = len(work)
    if n < window_frames:
        return []

    rows: list[dict[str, float | str | int]] = []
    for start in range(0, n - window_frames + 1, stride_frames):
        end = start + window_frames
        window = work.iloc[start:end]
        seg_id = (
            f"seg_{animal_id}_{session_id}_"
            f"{int(window['frame'].iloc[0])}_{int(window['frame'].iloc[-1])}"
        )
        out: dict[str, float | str | int] = {
            "segment_id": seg_id,
            "start_frame": int(window["frame"].iloc[0]),
            "end_frame": int(window["frame"].iloc[-1]),
            "animal_id": str(animal_id),
            "session_id": str(session_id),
        }
        for col in feature_cols:
            arr = window[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = float(np.mean(arr))
            out[f"{col}_std"] = float(np.std(arr))
            out[f"{col}_median"] = float(np.median(arr))
            out[f"{col}_max"] = float(np.max(arr))
            out[f"{col}_p10"] = float(np.percentile(arr, 10))
            out[f"{col}_p90"] = float(np.percentile(arr, 90))
            out[f"{col}_energy"] = float(np.sum(arr * arr) / max(1, len(arr)))
            if include_periodicity:
                centered = arr - np.mean(arr)
                if len(centered) >= 8 and np.var(centered) > 1e-10:
                    fft = np.fft.rfft(centered)
                    out[f"{col}_periodicity"] = float(np.max(np.abs(fft[1:])) if len(fft) > 1 else 0.0)
                else:
                    out[f"{col}_periodicity"] = 0.0
        rows.append(out)
    return rows


class BehaviorAdaptiveFeatureCacheService:
    """Build cacheable multi-scale segment features for benchmarking."""

    def _analysis_root(self, project_root: Path) -> Path:
        return project_root / "derived" / "analysis"

    def _cache_root(self, project_root: Path) -> Path:
        return self._analysis_root(project_root) / "benchmarks" / "feature_cache"

    @staticmethod
    def _scale_key(scale_sec: float) -> str:
        text = f"{float(scale_sec):.3f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    @staticmethod
    def _zscore_by_group(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        # Delegate to the single canonical implementation so all paths use the
        # same (vectorised, single-row-group-safe) standardisation.
        from abel.services.behavior_representation_service import BehaviorRepresentationService

        return BehaviorRepresentationService._zscore_by_group(df, feature_cols)

    @staticmethod
    def _segment_summary(
        window_df: pd.DataFrame,
        feature_cols: list[str],
        segment_id: str,
        include_periodicity: bool,
    ) -> dict[str, float | str | int]:
        out: dict[str, float | str | int] = {
            "segment_id": segment_id,
            "start_frame": int(window_df["frame"].iloc[0]),
            "end_frame": int(window_df["frame"].iloc[-1]),
            "animal_id": str(window_df["animal_id"].iloc[0]),
            "session_id": str(window_df["session_id"].iloc[0]),
        }
        for col in feature_cols:
            arr = window_df[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = float(np.mean(arr))
            out[f"{col}_std"] = float(np.std(arr))
            out[f"{col}_median"] = float(np.median(arr))
            out[f"{col}_max"] = float(np.max(arr))
            out[f"{col}_p10"] = float(np.percentile(arr, 10))
            out[f"{col}_p90"] = float(np.percentile(arr, 90))
            out[f"{col}_energy"] = float(np.sum(arr * arr) / max(1, len(arr)))
            if include_periodicity:
                centered = arr - np.mean(arr)
                if len(centered) >= 8 and np.var(centered) > 1e-10:
                    fft = np.fft.rfft(centered)
                    out[f"{col}_periodicity"] = float(np.max(np.abs(fft[1:])) if len(fft) > 1 else 0.0)
                else:
                    out[f"{col}_periodicity"] = 0.0
        return out

    @staticmethod
    def _numeric_feature_columns(frame_df: pd.DataFrame) -> list[str]:
        excluded = {"frame", "animal_id", "session_id", "video_id"}
        cols: list[str] = []
        for c in frame_df.columns:
            if c in excluded:
                continue
            if pd.api.types.is_numeric_dtype(frame_df[c]):
                cols.append(c)
        return cols

    def _load_merged_frame_features(self, project_root: Path, session_ids: list[str] | None = None) -> pd.DataFrame:
        # Prefer the canonical, already-z-scored representation cache.  It is
        # content-keyed (see BehaviorRepresentationService.build) and is the same
        # frame table the trainer uses, so reusing it avoids a redundant raw
        # reload + re-z-score on every adaptive-cache build.  z-scoring is
        # per-(animal_id, session_id), so filtering to a session subset keeps the
        # values correct.
        repr_frame = project_root / "derived" / "representations" / "frame_features.parquet"
        if repr_frame.exists():
            frame_df = pd.read_parquet(repr_frame)
            if session_ids:
                keep = {str(s) for s in session_ids}
                frame_df = frame_df[frame_df["session_id"].astype(str).isin(keep)].copy()
            feature_cols = self._numeric_feature_columns(frame_df)
            if feature_cols:
                return frame_df

        pose_path = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        ctx_path = project_root / "derived" / "context_features" / "frame_context.parquet"

        # Fall back to per-session directory when the monolithic file is absent.
        pose_sessions_dir = project_root / "derived" / "pose_features" / "sessions"
        ctx_sessions_dir = project_root / "derived" / "context_features" / "sessions"

        if pose_path.exists():
            pose_df = pd.read_parquet(pose_path)
        elif pose_sessions_dir.exists():
            parts = [pd.read_parquet(f) for f in sorted(pose_sessions_dir.glob("*.parquet"))]
            if not parts:
                raise FileNotFoundError(
                    "No frame pose feature files found. Run the full pipeline once to build them."
                )
            pose_df = pd.concat(parts, ignore_index=True)
        else:
            raise FileNotFoundError(
                "Missing frame pose feature cache. Expected derived/pose_features/frame_pose.parquet "
                "or per-session files under derived/pose_features/sessions/. "
                "Run the full pipeline once to build them."
            )

        ctx_df: pd.DataFrame | None = None
        if ctx_path.exists():
            ctx_df = pd.read_parquet(ctx_path)
        elif ctx_sessions_dir.exists():
            ctx_parts = [pd.read_parquet(f) for f in sorted(ctx_sessions_dir.glob("*.parquet"))]
            if ctx_parts:
                ctx_df = pd.concat(ctx_parts, ignore_index=True)

        if session_ids:
            keep = {str(s) for s in session_ids}
            pose_df = pose_df[pose_df["session_id"].astype(str).isin(keep)].copy()
            if ctx_df is not None:
                ctx_df = ctx_df[ctx_df["session_id"].astype(str).isin(keep)].copy()
        join_cols = ["frame", "animal_id", "session_id"]
        if ctx_df is not None and not ctx_df.empty:
            frame_df = pose_df.merge(ctx_df, on=join_cols, how="inner")
        else:
            frame_df = pose_df.copy()
        feature_cols = self._numeric_feature_columns(frame_df)
        if not feature_cols:
            raise ValueError("No numeric columns available in merged frame pose/context features.")
        return self._zscore_by_group(frame_df, feature_cols)

    def _build_scale_segments(
        self,
        frame_df: pd.DataFrame,
        scale_sec: float,
        fps: float,
        include_periodicity: bool,
        parallel_workers: int = 0,
        use_process_pool: bool = True,
        progress_cb: Callable[[str], None] | None = None,
        include_posture_deltas: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        window_frames = max(4, int(round(float(scale_sec) * float(fps))))
        stride_frames = max(1, int(round(window_frames * 0.25)))

        feature_cols = self._numeric_feature_columns(frame_df)

        grouped = list(frame_df.groupby(["animal_id", "session_id"]))
        n_groups = len(grouped)

        # ---- Vectorised path (GPU-preferred, vectorised-CPU fallback) --------
        # Eliminates the per-window Python loop by computing all window
        # statistics simultaneously.  When a CUDA GPU is present the work
        # runs on the GPU; otherwise a vectorised NumPy path is used which
        # is still substantially faster than the legacy per-window loop.
        from abel.utils.gpu_feature_ops import build_segment_df_fast, gpu_available

        use_vectorised = True  # always prefer vectorised path
        if use_vectorised:
            backend = "GPU (CUDA)" if gpu_available() else "vectorised CPU"
            if progress_cb is not None:
                progress_cb(
                    f"Phase 1 cache: using {backend} for {scale_sec:.3f}s segment window "
                    f"({n_groups} group(s))."
                )
            dfs: list[pd.DataFrame] = []
            for idx_group, ((animal_id, session_id), grp) in enumerate(grouped, start=1):
                seg_df = build_segment_df_fast(
                    grp,
                    feature_cols,
                    str(animal_id),
                    str(session_id),
                    window_frames,
                    stride_frames,
                    include_periodicity=include_periodicity,
                    include_posture_deltas=include_posture_deltas,
                )
                if not seg_df.empty:
                    dfs.append(seg_df)
                if progress_cb is not None and (idx_group % 5 == 0 or idx_group == n_groups):
                    total_segs = sum(len(d) for d in dfs)
                    progress_cb(
                        f"Phase 1 cache: {scale_sec:.3f}s window processed "
                        f"{idx_group}/{n_groups} sessions (segments={total_segs})."
                    )
            segment_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        meta = {
            "scale_sec": float(scale_sec),
            "window_frames": int(window_frames),
            "stride_frames": int(stride_frames),
            "n_rows": int(len(segment_df)),
            "n_groups": int(len(grouped)),
            "feature_columns": feature_cols,
        }
        return segment_df, meta

    def get_or_build_multiscale_cache(
        self,
        project_root: Path,
        config: MultiScaleFeatureCacheConfig,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        cache_root = self._cache_root(project_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        from abel.models.schemas import InvariantFeatureConfig
        posture_deltas = InvariantFeatureConfig.load_from_project(project_root).enable_clipwise_deltas

        index_path = cache_root / "index.json"
        index = read_json(index_path, {"scales": {}, "version": "phase1"})

        frame_df: pd.DataFrame | None = None
        results: dict[str, Any] = {"scales": {}, "index_path": str(index_path)}

        # When session_ids is provided (filtered/subset run), use a separate
        # cache subdirectory keyed by a short hash of the sorted session list.
        # This prevents a subset run from overwriting the full-dataset cache
        # and avoids serving stale full-dataset cache entries to subset runs.
        import hashlib as _hashlib
        if config.session_ids:
            _sid_hash = _hashlib.sha1(
                ",".join(sorted(str(s) for s in config.session_ids)).encode()
            ).hexdigest()[:12]
            scale_cache_root = cache_root / f"subset_{_sid_hash}"
            scale_cache_root.mkdir(parents=True, exist_ok=True)
        else:
            scale_cache_root = cache_root

        for scale_sec in sorted({float(s) for s in config.scales_sec if float(s) > 0.0}):
            scale_key = self._scale_key(scale_sec)
            seg_path = scale_cache_root / f"segment_features_scale_{scale_key}.parquet"
            meta_path = scale_cache_root / f"segment_features_scale_{scale_key}.meta.json"
            if progress_cb is not None:
                progress_cb(f"Phase 1 cache: preparing {scale_sec:.3f}s segment window.")

            should_build = config.regenerate or (not seg_path.exists()) or (not meta_path.exists())
            if should_build:
                if frame_df is None:
                    frame_df = self._load_merged_frame_features(project_root, session_ids=config.session_ids)
                if progress_cb is not None:
                    n_sessions_loaded = int(frame_df["session_id"].nunique()) if not frame_df.empty else 0
                    progress_cb(
                        f"Phase 1 cache: building {scale_sec:.3f}s segment features "
                        f"from {len(frame_df):,} frame rows ({n_sessions_loaded} session(s))."
                    )
                segment_df, meta = self._build_scale_segments(
                    frame_df,
                    scale_sec=scale_sec,
                    fps=config.fps,
                    include_periodicity=bool(config.include_window_periodicity),
                    parallel_workers=int(config.parallel_workers or 0),
                    use_process_pool=bool(config.use_process_pool),
                    progress_cb=progress_cb,
                    include_posture_deltas=posture_deltas,
                )
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                segment_df.to_parquet(seg_path, index=False)
                write_json(meta_path, meta)
                if progress_cb is not None:
                    progress_cb(
                        f"Phase 1 cache: wrote {int(meta.get('n_rows', 0))} segments ({scale_sec:.3f}s window)."
                    )
            else:
                meta = read_json(meta_path, {})
                if progress_cb is not None:
                    progress_cb(
                        f"Phase 1 cache: reused existing segment-feature cache ({scale_sec:.3f}s window)."
                    )

            index.setdefault("scales", {})[scale_key] = {
                "scale_sec": float(scale_sec),
                "segment_features": str(seg_path),
                "meta": str(meta_path),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            results["scales"][scale_key] = index["scales"][scale_key]

        write_json(index_path, index)
        return results
