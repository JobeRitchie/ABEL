"""Build frame and segment-level representations from pose and context features."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.services.provenance_service import ProvenanceService
from abel.storage.file_store import atomic_write_parquet, write_json


# Statistic suffixes the segment builder appends to each frame-level series.
# Frame-level columns carry none of these; segment-level ones always do.
_STAT_SUFFIXES = (
    "mean", "std", "median", "min", "max", "p10", "p90",
    "energy", "periodicity", "trend", "delta",
)


def canonical_distance_name(col: str) -> str:
    """Return the canonical (sorted-endpoint) spelling of a pairwise-distance column.

    Pairwise inter-keypoint distances are symmetric, so ``dist_a_to_b`` and
    ``dist_b_to_a`` (and their ``_norm`` variants) denote the same quantity; the
    canonical name sorts the two endpoints.  Non-distance columns — and ROI/target
    distances such as ``*_to_target_dist`` / ``*_to_roi_*`` that don't parse as a
    keypoint pair — are returned unchanged.  Centralised here so feature extraction,
    representation building, and model-vs-data alignment at inference all agree on
    one spelling.

    Handles both frame-level names (``dist_a_to_b``, ``dist_a_to_b_norm``) and
    segment-level ones, which append a statistic (``dist_a_to_b_norm_mean``).
    The statistic must be split off before sorting the endpoints, or it is
    swept into the second endpoint and sorted with it — turning
    ``dist_nose_to_left_ear_mean`` into ``dist_left_ear_mean_to_nose``, a name
    no table has, which then silently reindexes to a fill value.
    """
    stat = ""
    for suffix in _STAT_SUFFIXES:
        if col.endswith(f"_{suffix}"):
            col, stat = col[: -(len(suffix) + 1)], f"_{suffix}"
            break
    norm = col.endswith("_norm")
    core = col[: -len("_norm")] if norm else col
    if not core.startswith("dist_"):
        return col + ("_norm" if norm else "") + stat
    parts = core[len("dist_") :].split("_to_")
    if len(parts) != 2:
        return col + ("_norm" if norm else "") + stat
    a, b = parts
    return "dist_" + "_to_".join(sorted((a, b))) + ("_norm" if norm else "") + stat


def align_model_feature_columns(
    model_cols: list[str], data_cols: set[str],
) -> tuple[list[str], list[bool]]:
    """Map a model's feature columns onto the data columns that supply them.

    Returns ``(source_cols, nan_fill)``, both aligned 1:1 with ``model_cols``.
    ``source_cols[i]`` is the data column to read for ``model_cols[i]``; when it
    is absent from the data, the slot is filled with NaN if ``nan_fill[i]`` and
    0.0 otherwise.

    Three cases, in order:

    * **Present.** The data has the column under the model's own name.
    * **Spelled the other way round.** The data has this symmetric distance
      under the opposite endpoint ordering.  If the model expects *only* the one
      spelling, it reads the data's column — the same measurement, renamed.  If
      the model expects *both* spellings, the pair is double-named: in training
      exactly one of the two held the value and the other was NaN (the training
      set was assembled across two extractor eras with opposite conventions), so
      the surplus slot is NaN-filled to reproduce that.  Copying the value into
      both would present a combination the model never saw.
    * **Genuinely absent.** Filled with 0.0, as before.

    Never NaN-fill by assuming 0.0 is harmless here: these features are z-scored,
    so 0.0 asserts an *average* value rather than an absent measurement.
    """
    data = set(data_cols)
    model_set = set(model_cols)
    source: list[str] = []
    nan_fill: list[bool] = []
    for col in model_cols:
        if col in data:
            source.append(col)
            nan_fill.append(False)
            continue
        canon = canonical_distance_name(col)
        if canon != col and canon in data:
            if canon in model_set:
                # Double-named pair — the model already reads the value under
                # the canonical name, so this slot was the NaN one in training.
                source.append(col)
                nan_fill.append(True)
            else:
                source.append(canon)
                nan_fill.append(False)
            continue
        source.append(col)
        nan_fill.append(False)
    return source, nan_fill


# Label tokens that denote the negative / "not this behaviour" class.  A model's
# label_map pairs the target behaviour's id with one of these.
_NO_BEHAVIOR_TOKENS = frozenset({
    "no_behavior", "no_behaviour", "nobehavior", "nobehaviour",
})


def normalize_label_token(label: Any) -> str:
    """Lower-case, punctuation-collapsed form of a label for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")


def is_no_behavior_label(label: Any) -> bool:
    """True if *label* denotes the negative / no-behaviour class."""
    return normalize_label_token(label) in _NO_BEHAVIOR_TOKENS


def resolve_target_class_index(
    label_map: dict[Any, Any] | None, target_behavior: str,
) -> int | None:
    """Column index into ``predict_proba`` for *target_behavior*'s positive class.

    ``label_map`` maps a class index to its stored label (a behaviour id/name, or
    a ``no_behavior`` sentinel).  Resolution order:

    1. **Exact match** — the target behaviour's id (or name) is stored under some
       class index.  Tolerant of punctuation/case via :func:`normalize_label_token`.
    2. **Binary positive-class fallback** — the map has exactly one class that is
       not a ``no_behavior`` label.  That class *is* the behaviour's positive
       class even when its stored id differs from ``target_behavior``.  This is
       precisely the imported-model case: a model copied from another project
       keeps the *source* project's behaviour UUID in its ``label_map``, so an id
       match fails, yet the single non-``no_behavior`` class is unambiguously the
       target.  Selecting it here is what keeps active-learning inference and
       dense temporal refinement scoring the *same*, correct class.

    Returns ``None`` when neither rule applies (a genuinely multi-class map with
    no id match), leaving the caller to choose a last resort.

    This is the single source of truth for target-class selection; every scorer
    (active-learning run-models, dense temporal refinement) must go through it so
    they can never again disagree on which class column is "the behaviour".
    """
    if not isinstance(label_map, dict) or not label_map:
        return None
    items: list[tuple[int, str]] = []
    for k, v in label_map.items():
        try:
            items.append((int(k), str(v)))
        except (TypeError, ValueError):
            continue
    if not items:
        return None

    target = str(target_behavior)
    for idx, label in items:
        if label == target:
            return idx
    ntarget = normalize_label_token(target)
    if ntarget:
        for idx, label in items:
            if normalize_label_token(label) == ntarget:
                return idx

    positives = [idx for idx, label in items if not is_no_behavior_label(label)]
    if len(positives) == 1:
        return positives[0]
    return None


@dataclass
class RepresentationConfig:
    window_size_frames: int = 60
    window_stride_frames: int = 15
    model_version: str = "behavior_repr_v1"
    # v2: clip-wise _delta is now an edge-band average (mean of last k − mean of
    # first k frames) instead of last-frame − first-frame.
    # v3: pairwise-distance columns are canonicalised (``dist_a_to_b`` /
    # ``dist_b_to_a`` merged onto the sorted name) so mixed-order pose exports no
    # longer produce duplicate, half-populated "dead" distance columns.
    # Bumping the version invalidates the content/config-hash representation cache
    # so segment features are rebuilt with the current feature definitions.
    feature_version: str = "representation_v3"
    # DEPRECATED / no-op: feature exclusions are applied at training time, not
    # baked into the representation, so the cache stays valid across exclusion
    # changes.  Retained only for call-site compatibility.  See
    # ActiveLearningTrainerService for where exclusions are actually applied.
    excluded_feature_cols: frozenset[str] = field(default_factory=frozenset)


class BehaviorRepresentationService:
    """Create normalized frame-level and summary segment-level features."""

    def __init__(self) -> None:
        self._provenance = ProvenanceService()

    @staticmethod
    def _parquet_content_sig(path: Path) -> list:
        """Cheap content/structure signature of a parquet file.

        Reads only the footer: row count, sorted column names, and a digest of
        per-column statistics (min / max / null-count / value-count aggregated
        across row groups).  All of this comes from the parquet footer, so it is
        still cheap (no data pages are read) and independent of file mtime or
        byte-level compression — re-saving identical data yields an identical
        signature.  The statistics digest closes a correctness gap: re-extracting
        features with the *same* schema and row count but *different values*
        (e.g. a smoothing/units change, or a pose re-export) now invalidates the
        cache, whereas row-count + column-names alone would silently reuse stale
        segment/training features built from the old values.
        """
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            md = pf.metadata
            names = sorted(pf.schema_arrow.names)
            return [int(md.num_rows), names, BehaviorRepresentationService._parquet_stats_digest(md)]
        except Exception:
            try:
                return [int(path.stat().st_size), []]
            except OSError:
                return [0, []]

    @staticmethod
    def _parquet_stats_digest(md: object) -> str:
        """Footer-only digest of per-column statistics; '' when unavailable.

        Aggregates each column's statistics across all row groups (min-of-mins,
        max-of-maxes, summed null/value counts) so the digest is invariant to
        how the writer chunked the data into row groups — an identical re-save
        produces the same digest — while a genuine change in values shifts a
        min/max/null-count and therefore the digest.
        """
        import hashlib  # noqa: PLC0415

        try:
            per_col: dict[str, list] = {}
            for rg in range(md.num_row_groups):  # type: ignore[attr-defined]
                row_group = md.row_group(rg)  # type: ignore[attr-defined]
                for ci in range(row_group.num_columns):
                    col = row_group.column(ci)
                    st = getattr(col, "statistics", None)
                    if st is None:
                        continue
                    name = col.path_in_schema
                    mn = st.min if st.has_min_max else None
                    mx = st.max if st.has_min_max else None
                    nulls = int(st.null_count) if st.has_null_count else -1
                    nvals = int(getattr(st, "num_values", 0) or 0)
                    agg = per_col.get(name)
                    if agg is None:
                        per_col[name] = [mn, mx, nulls, nvals]
                    else:
                        if mn is not None:
                            agg[0] = mn if agg[0] is None else (mn if mn < agg[0] else agg[0])
                        if mx is not None:
                            agg[1] = mx if agg[1] is None else (mx if mx > agg[1] else agg[1])
                        agg[2] = agg[2] + nulls if agg[2] >= 0 and nulls >= 0 else -1
                        agg[3] += nvals
            if not per_col:
                return ""
            payload = "|".join(
                f"{name}:{per_col[name][0]!r}:{per_col[name][1]!r}:{per_col[name][2]}:{per_col[name][3]}"
                for name in sorted(per_col)
            )
            return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=16).hexdigest()
        except Exception:
            return ""

    @classmethod
    def _source_signature(
        cls, frame_pose_path: Path | None, frame_context_path: Path | None
    ) -> dict[str, list]:
        """Content signature of all source pose/context feature files.

        Prefers the per-session directories (the format written by the parallel
        preprocessing stage); falls back to the legacy monolithic files.
        """
        sig: dict[str, list] = {}
        pose_dir = (frame_pose_path.parent / "sessions") if frame_pose_path else None
        ctx_dir = (frame_context_path.parent / "sessions") if frame_context_path else None

        used_dir = False
        for tag, d in (("pose", pose_dir), ("ctx", ctx_dir)):
            if d is not None and d.exists():
                used_dir = True
                for f in sorted(d.glob("*.parquet")):
                    sig[f"{tag}/{f.name}"] = cls._parquet_content_sig(f)

        if not used_dir:
            for tag, f in (("pose", frame_pose_path), ("ctx", frame_context_path)):
                if f is not None and f.exists():
                    sig[f"{tag}/{f.name}"] = cls._parquet_content_sig(f)
        return sig

    @staticmethod
    def _config_signature(config: "RepresentationConfig") -> dict:
        """Signature of the representation config that is baked into the cache."""
        # NOTE: ``excluded_feature_cols`` is deliberately NOT part of the cache
        # signature.  Feature exclusions are applied at *training* time (see
        # ActiveLearningTrainerService), so the representation always contains
        # all features and the cache stays valid regardless of which features a
        # user chooses to exclude downstream.
        return {
            "window_size_frames": int(config.window_size_frames),
            "window_stride_frames": int(config.window_stride_frames),
            "feature_version": str(config.feature_version),
            "model_version": str(config.model_version),
        }

    @staticmethod
    def _canonicalize_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Merge non-canonical pairwise-distance columns onto their sorted name.

        Pairwise inter-keypoint distances are symmetric, so ``dist_a_to_b`` and
        ``dist_b_to_a`` denote the same quantity (likewise their ``_norm``
        variants).  Pose files that list keypoints in different orders
        historically produced both spellings; concatenating such sessions yields
        two half-populated columns per pair with complementary NaNs, each of
        which can look "dead" (e.g. its delta/trend collapses).  Pose extraction
        now emits the sorted (canonical) name, but older or mixed caches still
        contain both spellings.

        This collapses every distance column onto its canonical (sorted) name,
        combining values so the merged column is fully populated across sessions,
        and drops the redundant duplicate(s).  Non-distance columns and
        ROI/target distances (``*_to_target_dist`` / ``*_to_roi_*``) are left
        untouched.
        """
        def _canonical(col: str) -> str | None:
            canon = canonical_distance_name(col)
            return canon if canon != col or col.startswith("dist_") else None

        groups: dict[str, list[str]] = {}
        for col in df.columns:
            canon = _canonical(col)
            if canon is None:
                continue
            bucket = groups.setdefault(canon, [])
            # Canonical spelling (if present) takes precedence in the merge.
            bucket.insert(0, col) if col == canon else bucket.append(col)

        drop: list[str] = []
        for canon, cols in groups.items():
            if len(cols) == 1 and cols[0] == canon:
                continue  # already canonical with no duplicate spelling
            merged = None
            for c in cols:
                merged = df[c] if merged is None else merged.combine_first(df[c])
            df[canon] = merged
            drop.extend(c for c in cols if c != canon)
        if drop:
            df = df.drop(columns=drop)
        return df

    ZSCORE_STATS_FILENAME = "zscore_stats.parquet"

    @classmethod
    def _zscore_by_group_with_stats(
        cls, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-(animal_id, session_id) standardisation, returning the stats too.

        Vectorised via ``groupby(...).transform`` instead of a per-group Python
        loop with ``.loc`` assignment — same result, dramatically faster on the
        ~5M-frame / 93-session tables this runs on.  The per-group mean/std are
        deterministic and small, so they are returned for persistence and reuse
        (e.g. Direct-Use inference on new data).
        """
        out = df.copy()
        if not feature_cols:
            return out, pd.DataFrame(columns=["animal_id", "session_id"])

        grp = out.groupby(["animal_id", "session_id"])[feature_cols]
        mu = grp.transform("mean")
        # std() returns NaN for single-row groups; treat both NaN and 0 as 1 to
        # avoid producing NaN/Inf in the scaled features.
        sigma = grp.transform("std").fillna(1.0).replace(0.0, 1.0)
        out[feature_cols] = (out[feature_cols] - mu) / sigma

        mu_g = grp.mean()
        sigma_g = grp.std().fillna(1.0).replace(0.0, 1.0)
        stats = (
            mu_g.add_suffix("__mean")
            .join(sigma_g.add_suffix("__std"))
            .reset_index()
        )
        return out, stats

    @classmethod
    def _zscore_by_group(cls, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out, _ = cls._zscore_by_group_with_stats(df, feature_cols)
        return out

    @staticmethod
    def _segment_summary(window_df: pd.DataFrame, feature_cols: list[str], segment_id: str) -> dict[str, float | str | int]:
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

            centered = arr - np.mean(arr)
            if len(centered) >= 8 and np.var(centered) > 1e-10:
                fft = np.fft.rfft(centered)
                out[f"{col}_periodicity"] = float(np.max(np.abs(fft[1:])) if len(fft) > 1 else 0.0)
            else:
                out[f"{col}_periodicity"] = 0.0
        return out

    def build(
        self,
        project_root: Path,
        frame_pose_path: Path,
        frame_context_path: Path | None,
        config: RepresentationConfig | None = None,
        session_ids: set[str] | None = None,
        progress_cb: Callable[[str], None] | None = None,
        ensure_only: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # ``ensure_only`` skips loading the (potentially multi-GB) cached frame
        # and segment parquet on a full cache hit — the caller only wants the
        # cache to *exist* (e.g. pre-building during feature extraction), not
        # the dataframes.  Returns empty frames in that case.
        config = config or RepresentationConfig()

        def _progress(msg: str) -> None:
            if progress_cb is not None:
                progress_cb(msg)

        # Fast path: return cached outputs when the full representation cache
        # already exists.  For full (non-filtered) runs the cache is returned
        # as-is.  For subset runs (session_ids provided) the cache is loaded
        # and filtered in memory — much faster than re-deriving features from
        # raw pose/context data, and avoids overwriting the full cache with a
        # partial result.
        #
        # The segment window/stride are baked into the cached segment features,
        # so a config change requires a manual cache clear
        # (delete derived/representations/).  This is intentional and consistent
        # with how other derived artefacts work in ABEL.
        out_dir = project_root / "derived" / "representations"
        _frame_cached = out_dir / "frame_features.parquet"
        _seg_cached = out_dir / "segment_features.parquet"
        _manifest_cached = out_dir / "representations.manifest.json"

        # Invalidate the representation cache only when the *content* of the
        # underlying feature files has changed, or when the representation
        # config (window/stride/feature version/exclusions) differs from what
        # produced the cache.  We deliberately do NOT key on file mtime:
        # re-extracting features re-saves the parquet files (bumping mtime) even
        # when the data is equivalent, which previously forced a full reload +
        # re-z-score + segment rebuild on every run.  The content signature is
        # read from the parquet footer (row counts + column names), so genuine
        # changes (new sessions, ROI re-extract that changes context columns,
        # changed feature set) still invalidate correctly.
        cache_signature = {
            "sources": self._source_signature(frame_pose_path, frame_context_path),
            "config": self._config_signature(config),
        }
        if _manifest_cached.exists():
            try:
                _meta = json.loads(_manifest_cached.read_text(encoding="utf-8"))
            except Exception:
                _meta = {}
            if _meta.get("cache_signature") != cache_signature:
                _progress(
                    "Representation: cache is stale (source content or config "
                    "changed) — clearing and rebuilding..."
                )
                _frame_cached.unlink(missing_ok=True)
                _seg_cached.unlink(missing_ok=True)
                _manifest_cached.unlink(missing_ok=True)
        elif _frame_cached.exists():
            # Frame cache present but no signature manifest (older cache format):
            # rebuild once so future runs get a fast content-signature hit.
            _progress(
                "Representation: cache present without signature manifest — "
                "rebuilding once to record content signature..."
            )
            _frame_cached.unlink(missing_ok=True)
            _seg_cached.unlink(missing_ok=True)

        if _frame_cached.exists() and _seg_cached.exists() and _manifest_cached.exists():
            if not session_ids:
                if ensure_only:
                    _progress("Representation: cache hit — already prepared (skipping load).")
                    return pd.DataFrame(), pd.DataFrame()
                _progress("Representation: cache hit — loading existing frame and segment features...")
                return pd.read_parquet(_frame_cached), pd.read_parquet(_seg_cached)
            else:
                # Subset fast-path: load full cache then filter to requested sessions.
                # z-scoring was done per (animal_id, session_id) so independent
                # sessions remain correctly normalised after filtering.
                keep = {str(s) for s in session_ids}
                _progress(
                    f"Representation: cache hit (subset) — filtering {len(keep)} session(s) "
                    "from existing cached features..."
                )
                frame_sub = pd.read_parquet(_frame_cached)
                seg_sub = pd.read_parquet(_seg_cached)
                frame_sub = frame_sub[frame_sub["session_id"].astype(str).isin(keep)].copy()
                seg_sub = seg_sub[seg_sub["session_id"].astype(str).isin(keep)].copy()
                _progress(
                    f"Representation: subset filter applied — "
                    f"frame_rows={len(frame_sub)}, segment_rows={len(seg_sub)}."
                )
                # Detect stale cache: find sessions that are present in the
                # frame cache with enough frames to produce at least one segment
                # window but are entirely absent from the segment cache.  This
                # catches both the "all segments missing" case and the more
                # subtle "partially stale" case where new sessions were added
                # after the segment cache was last built.
                _seg_sessions: set[str] = (
                    set(seg_sub["session_id"].astype(str).unique())
                    if not seg_sub.empty
                    else set()
                )
                _frame_counts = (
                    frame_sub.groupby("session_id").size()
                    if not frame_sub.empty
                    else pd.Series(dtype=int)
                )
                _eligible_sessions = {
                    str(s) for s, n in _frame_counts.items()
                    if n >= config.window_size_frames
                }
                _missing_eligible = _eligible_sessions - _seg_sessions
                if _missing_eligible and len(frame_sub) > 0:
                    # The segment cache is stale (e.g. previously overwritten
                    # with a different session subset, or new sessions added
                    # after the last build).  Remove the stale files and rebuild
                    # the full cache so future runs are fast.
                    _progress(
                        f"Representation: segment cache is stale — "
                        f"{len(_missing_eligible)} session(s) have frames but no segments "
                        f"({', '.join(sorted(_missing_eligible)[:5])}"
                        f"{'…' if len(_missing_eligible) > 5 else ''}). "
                        "Clearing cache and rebuilding..."
                    )
                    _seg_cached.unlink(missing_ok=True)
                    _manifest_cached.unlink(missing_ok=True)
                    # Rebuild the full (unfiltered) cache, then filter.
                    full_frame, full_seg = self.build(
                        project_root=project_root,
                        frame_pose_path=frame_pose_path,
                        frame_context_path=frame_context_path,
                        config=config,
                        session_ids=None,
                        progress_cb=progress_cb,
                    )
                    frame_sub = full_frame[full_frame["session_id"].astype(str).isin(keep)].copy()
                    seg_sub = full_seg[full_seg["session_id"].astype(str).isin(keep)].copy()
                    _progress(
                        f"Representation: rebuilt and filtered — "
                        f"frame_rows={len(frame_sub)}, segment_rows={len(seg_sub)}."
                    )
                return frame_sub, seg_sub

        _progress("Representation: loading frame-level pose/context features...")

        _keep_set = {str(s) for s in session_ids} if session_ids else None
        pose_sessions_dir = frame_pose_path.parent / "sessions"
        ctx_sessions_dir = frame_context_path.parent / "sessions" if frame_context_path is not None else None

        # Per-session files path: each session is stored independently so only
        # the needed sessions are read.  This avoids loading the entire
        # (potentially very large) monolithic parquet when only a subset of
        # sessions is required, and is also the output format written by the
        # parallel preprocessing stage.
        if pose_sessions_dir.exists() or (ctx_sessions_dir is not None and ctx_sessions_dir.exists()):
            available_pose = {f.stem for f in pose_sessions_dir.glob("*.parquet")} if pose_sessions_dir.exists() else set()
            available_ctx = {f.stem for f in ctx_sessions_dir.glob("*.parquet")} if (ctx_sessions_dir is not None and ctx_sessions_dir.exists()) else set()

            load_pose = (_keep_set & available_pose) if _keep_set else available_pose
            load_ctx = (_keep_set & available_ctx) if _keep_set else available_ctx

            pose_parts: list[pd.DataFrame] = []
            for sid in sorted(load_pose):
                pf = pose_sessions_dir / f"{sid}.parquet"
                if pf.exists():
                    pose_parts.append(pd.read_parquet(pf))

            # Any sessions in _keep_set that are not in the per-session dir may
            # still be in the legacy monolithic file (e.g. older projects).
            if _keep_set and frame_pose_path.exists():
                missing_from_dir = _keep_set - available_pose
                if missing_from_dir:
                    try:
                        legacy_pose = pd.read_parquet(frame_pose_path)
                        legacy_pose = legacy_pose[legacy_pose["session_id"].astype(str).isin(missing_from_dir)]
                        if not legacy_pose.empty:
                            pose_parts.append(legacy_pose)
                    except Exception:
                        pass

            ctx_parts: list[pd.DataFrame] = []
            for sid in sorted(load_ctx):
                cf_ = ctx_sessions_dir / f"{sid}.parquet"
                if cf_.exists():
                    ctx_parts.append(pd.read_parquet(cf_))

            if _keep_set and frame_context_path is not None and frame_context_path.exists():
                missing_ctx_from_dir = _keep_set - available_ctx
                if missing_ctx_from_dir:
                    try:
                        legacy_ctx = pd.read_parquet(frame_context_path)
                        legacy_ctx = legacy_ctx[legacy_ctx["session_id"].astype(str).isin(missing_ctx_from_dir)]
                        if not legacy_ctx.empty:
                            ctx_parts.append(legacy_ctx)
                    except Exception:
                        pass

            pose_df = pd.concat(pose_parts, ignore_index=True) if pose_parts else pd.DataFrame()
            ctx_df = pd.concat(ctx_parts, ignore_index=True) if ctx_parts else pd.DataFrame()
            _progress(
                f"Representation: loaded {len(load_pose)} pose session file(s) and "
                f"{len(load_ctx)} context session file(s) "
                f"(pose_rows={len(pose_df)}, ctx_rows={len(ctx_df)})."
            )
        else:
            # Legacy: load monolithic files and filter to required sessions.
            if not frame_pose_path.exists() or (frame_context_path is not None and not frame_context_path.exists()):
                raise ValueError(
                    "Frame feature files not found.  Re-run feature extraction to regenerate them, "
                    "or clear the representation cache and try again."
                )
            pose_df = pd.read_parquet(frame_pose_path)
            ctx_df = pd.read_parquet(frame_context_path) if frame_context_path is not None else pd.DataFrame()
            if _keep_set:
                pose_df = pose_df[pose_df["session_id"].astype(str).isin(_keep_set)].copy()
                ctx_df = ctx_df[ctx_df["session_id"].astype(str).isin(_keep_set)].copy()
                _progress(
                    f"Representation: filtered to {len(_keep_set)} selected session(s); "
                    f"pose_rows={len(pose_df)}, ctx_rows={len(ctx_df)}."
                )

        join_cols = ["frame", "animal_id", "session_id"]
        _progress("Representation: merging pose and context frame tables...")
        frame_df = pose_df.merge(ctx_df, on=join_cols, how="inner") if not ctx_df.empty else pose_df.copy()

        # Collapse symmetric pairwise-distance duplicates (dist_a_to_b /
        # dist_b_to_a) onto the canonical sorted name before any statistics are
        # computed, so mixed-order pose exports don't leave half-populated "dead"
        # distance columns downstream.
        _n_cols_before = frame_df.shape[1]
        frame_df = self._canonicalize_distance_columns(frame_df)
        if frame_df.shape[1] != _n_cols_before:
            _progress(
                f"Representation: canonicalised distance columns "
                f"({_n_cols_before - frame_df.shape[1]} duplicate spelling(s) merged)."
            )

        # Feature exclusions are NOT applied here — neither the project-level
        # config/feature_exclusions.json NOR the per-run ``excluded_feature_cols``.
        # The representation builder must compute statistics for ALL available
        # features so the cache is independent of feature-selection choices and
        # downstream consumers (trainer, evaluation) can decide what to include.
        # Applying exclusions at the frame level also created a circular
        # dependency: dead features were excluded → never got windowed stats →
        # stayed "dead" even after the underlying data was fixed.  The trainer
        # applies all exclusions at the segment level instead.
        excluded = {"frame", "animal_id", "session_id", "video_id"}

        feature_cols = [
            c
            for c in frame_df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(frame_df[c])
        ]
        if not feature_cols:
            raise ValueError(
                "No numeric feature columns available after merging pose and context features."
            )
        _progress(f"Representation: z-scoring {len(feature_cols)} numeric feature columns by session...")
        frame_df, zscore_stats = self._zscore_by_group_with_stats(frame_df, feature_cols)

        from abel.utils.gpu_feature_ops import build_segment_df_fast, gpu_available
        from abel.models.schemas import InvariantFeatureConfig

        posture_deltas = InvariantFeatureConfig.load_from_project(project_root).enable_clipwise_deltas
        backend = "GPU (CUDA)" if gpu_available() else "vectorised CPU"
        grouped = list(frame_df.groupby(["animal_id", "session_id"]))
        n_groups = len(grouped)
        _progress(
            f"Representation: building segments via {backend} ({n_groups} group(s))"
            + (" with clip-wise deltas" if posture_deltas else "")
            + "..."
        )
        dfs: list[pd.DataFrame] = []
        for idx_group, ((animal_id, session_id), grp) in enumerate(grouped, start=1):
            n = len(grp)
            if n < config.window_size_frames:
                if idx_group % 2 == 0 or idx_group == n_groups:
                    _progress(
                        f"Representation: group {idx_group}/{n_groups} ({session_id}) skipped; "
                        f"frames={n} < window={config.window_size_frames}."
                    )
                continue
            seg_df = build_segment_df_fast(
                grp,
                feature_cols,
                str(animal_id),
                str(session_id),
                config.window_size_frames,
                config.window_stride_frames,
                include_periodicity=True,
                include_posture_deltas=posture_deltas,
            )
            if not seg_df.empty:
                dfs.append(seg_df)
            if idx_group % 2 == 0 or idx_group == n_groups:
                total_segs = sum(len(d) for d in dfs)
                _progress(
                    f"Representation: processed group {idx_group}/{n_groups} ({session_id}); "
                    f"segments_so_far={total_segs}."
                )

        segment_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        _progress(
            f"Representation: completed frame_rows={len(frame_df)}, segment_rows={len(segment_df)}."
        )

        # Only persist the canonical cache for full (non-filtered) runs.  Subset
        # runs (session_ids provided) must not overwrite the full-dataset cache
        # so that subsequent full runs remain valid.
        if not session_ids:
            out_dir.mkdir(parents=True, exist_ok=True)
            frame_out = out_dir / "frame_features.parquet"
            seg_out = out_dir / "segment_features.parquet"
            # Atomic writes: a run interrupted mid-write (e.g. app closed) must
            # never leave a truncated, footerless parquet at the canonical path,
            # which would break every downstream reader.
            atomic_write_parquet(frame_df, frame_out, index=False)
            atomic_write_parquet(segment_df, seg_out, index=False)
            # Persist the per-(animal_id, session_id) z-score statistics so that
            # downstream consumers (Direct-Use inference, refinement) can reuse
            # the exact training-time baseline instead of recomputing it.
            try:
                if zscore_stats is not None and not zscore_stats.empty:
                    atomic_write_parquet(
                        zscore_stats, out_dir / self.ZSCORE_STATS_FILENAME, index=False,
                    )
            except Exception:
                pass

            prov = self._provenance.make_provenance(
                project_root=project_root,
                model_version=config.model_version,
                feature_version=config.feature_version,
                config={"representation_config": config.__dict__, "feature_columns": feature_cols},
            )
            write_json(
                out_dir / "representations.manifest.json",
                {
                    "frame_features": str(frame_out),
                    "segment_features": str(seg_out),
                    "feature_columns": feature_cols,
                    "provenance": prov.model_dump(mode="json"),
                    # Content/config signature for mtime-independent cache reuse.
                    "cache_signature": cache_signature,
                },
            )

        return frame_df, segment_df
