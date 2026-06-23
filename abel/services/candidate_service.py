"""Candidate generation service.

Ranks pose feature segments for downstream active learning.

Pipeline position:
    Pose Features → Behavior Representations
    → Candidate Generation ← here
    → Review → Temporal Refinement
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from abel.models.schemas import CandidateSegment, CandidateWindow
from abel.services.provenance_service import ProvenanceService
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")


@dataclass
class SegmentCandidateGenerationConfig:
    top_k: int = 300
    mode: str = "uncertainty"  # prototype | uncertainty | novelty | low_probability | random_absent
    target_behavior_id: str | None = None
    uncertainty_threshold: float = 0.0
    prediction_threshold: float = 0.0
    model_version: str = "behavior_model_v1"
    feature_version: str = "representation_v1"
    hard_negative_ratio: float = 0.3
    query_size: int = 100
    low_prob_max_prob: float = 0.25
    random_absent_max_prob: float = 0.35
    random_seed: int | None = None
    sample_window_frames: int = 60
    examples_per_session: int = 0  # 0 = unlimited; positive = keep at most this many candidates per session
    selected_session_ids: list[str] = field(default_factory=list)
    all_behavior_aware: bool = True
    all_behavior_competition_margin: float = 0.05
    allow_co_occurring_behaviors: bool = False
    enable_weighted_queue_scoring: bool = False
    enable_uncertainty_sampling: bool = True
    enable_expert_disagreement: bool = False
    enable_diversity_sampling: bool = False
    diversity_mode: str = "distance_to_reviewed"  # distance_to_reviewed | clustering_quota
    enable_confound_sampling: bool = False
    enable_hard_negative_mining: bool = False
    exploration_fraction: float = 0.15
    min_false_positive_for_hard_negative: int = 8
    queue_weight_candidate: float = 0.35
    queue_weight_uncertainty: float = 0.20
    queue_weight_disagreement: float = 0.15
    queue_weight_diversity: float = 0.10
    queue_weight_confound: float = 0.10
    queue_weight_hard_negative: float = 0.07
    queue_weight_exploration: float = 0.03


@dataclass
class SegmentCandidateGenerationResult:
    n_segments_loaded: int = 0
    n_segments_ranked: int = 0
    n_segments_selected: int = 0
    candidates: list[CandidateSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    success: bool = False


class CandidateGenerationService:
    """Ranks pose-feature segments for active-learning review."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._provenance = ProvenanceService()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def _external_windows_path(self) -> Path | None:
        if not self._project_root:
            return None
        return self._project_root / "derived" / "review_tables" / "external_window_candidates.json"

    def save_candidates(self, candidates: list[CandidateWindow]) -> None:
        """Persist a list of candidate windows to JSON."""
        if not self._project_root:
            return
        path = self._project_root / "derived" / "review_tables" / "candidate_windows.json"
        write_json(
            path,
            {
                "n_candidates_selected": len(candidates),
                "candidates": [c.model_dump(mode="json") for c in candidates],
            },
        )
        logger.info("Candidate windows saved: %d -> %s", len(candidates), path)

    def load_candidates(self) -> list[CandidateWindow]:
        # Segment-native path: UI consumers still use CandidateWindow shape, but
        # data source is always candidate_segments.json.
        if not self._project_root:
            return []
        path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        raw = read_json(path, {"candidates": [], "config": {}})
        target_behavior = str(((raw.get("config") or {}).get("target_behavior_id") or "")).strip() or None
        rows = self._segment_to_window_candidates(self.load_segment_candidates(), target_behavior)
        merged: dict[str, CandidateWindow] = {str(c.window_id): c for c in rows}
        for ext in self.load_external_window_candidates():
            merged[str(ext.window_id)] = ext
        return list(merged.values())

    def has_candidates(self) -> bool:
        if not self._project_root:
            return False
        path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        ext_path = self._external_windows_path()
        return path.exists() or (ext_path.exists() if ext_path else False)

    def load_external_window_candidates(self) -> list[CandidateWindow]:
        path = self._external_windows_path()
        if path is None:
            return []
        raw = read_json(path, {"candidates": []})
        out: list[CandidateWindow] = []
        for item in list(raw.get("candidates", [])):
            try:
                out.append(CandidateWindow.model_validate(item))
            except Exception:
                continue
        return out

    def remove_external_candidates_by_source(self, source: str) -> int:
        """Remove all persisted external window candidates whose source matches *source*.

        Returns the number of candidates removed.
        """
        path = self._external_windows_path()
        if path is None or not source:
            return 0
        existing = self.load_external_window_candidates()
        kept = [c for c in existing if (c.source or "") != source]
        removed = len(existing) - len(kept)
        if removed:
            write_json(
                path,
                {
                    "updated_at": datetime.utcnow().isoformat(),
                    "candidates": [c.model_dump(mode="json") for c in kept],
                },
            )
        return removed

    def upsert_external_window_candidates(self, candidates: list[CandidateWindow]) -> int:
        """Add or update external window candidates persisted for cross-tab visibility."""
        path = self._external_windows_path()
        if path is None or not candidates:
            return 0

        existing = self.load_external_window_candidates()
        by_id: dict[str, CandidateWindow] = {str(c.window_id): c for c in existing}
        before = len(by_id)
        for cand in candidates:
            by_id[str(cand.window_id)] = cand

        write_json(
            path,
            {
                "updated_at": datetime.utcnow().isoformat(),
                "candidates": [c.model_dump(mode="json") for c in by_id.values()],
            },
        )
        return max(0, len(by_id) - before)

    def clear_clip_paths(self, session_id: str | None = None) -> int:
        """Clear persisted clip_path links from saved candidates.

        Args:
            session_id: when provided, only candidates for this session are updated.

        Returns:
            Number of candidate rows whose clip_path was cleared.
        """
        # Segment candidates do not persist clip paths; no-op for segment-native flow.
        _ = session_id
        return 0

    def clear_candidates(self) -> bool:
        """Delete the entire persisted candidate list for the current project.

        This is the unconditional wipe used by the Candidate Generation tab
        before regenerating.  The Clip tab's "Clear Candidates" button uses
        :meth:`clear_candidate_queue` instead, which preserves reviewed windows.
        """
        if not self._project_root:
            return False
        path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        ext_path = self._external_windows_path()
        removed = False
        if path.exists():
            path.unlink(missing_ok=True)
            removed = True
            logger.info("Cleared persisted candidates: %s", path)
        if ext_path is not None and ext_path.exists():
            ext_path.unlink(missing_ok=True)
            removed = True
            logger.info("Cleared persisted external window candidates: %s", ext_path)
        return removed

    def reviewed_segment_ids(self) -> set[str]:
        """Return segment IDs that carry any reviewer label.

        A segment is "reviewed" once it has any decision recorded in
        ``reviewer_labels.parquet`` (accept / reject / relabel / etc.).  These
        identify windows the user has already worked on, so clearing the
        candidate queue must keep them.
        """
        if not self._project_root:
            return set()
        path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        if not path.exists():
            return set()
        try:
            labels = pd.read_parquet(path, columns=["segment_id"])
        except Exception:
            try:
                labels = pd.read_parquet(path)
            except Exception:
                return set()
        if "segment_id" not in labels.columns:
            return set()
        return {
            str(s).strip()
            for s in labels["segment_id"].tolist()
            if str(s).strip()
        }

    def clear_candidate_queue(self, preserve_reviewed: bool = True) -> dict:
        """Clear the pending candidate queue shown in the Clip tab.

        Removes candidate windows from both the persisted segment queue
        (``candidate_segments.json``) and the external-window queue.  When
        *preserve_reviewed* is True (the default), windows the user has already
        reviewed — those whose ``segment_id`` appears in
        ``reviewer_labels.parquet`` — are kept so reviewed clips/windows are
        never lost.  This method never touches reviewer labels or rendered clip
        files under ``derived/clips/``.

        Returns ``{"removed": int, "kept_reviewed": int}``.
        """
        if not self._project_root:
            return {"removed": 0, "kept_reviewed": 0}

        reviewed = self.reviewed_segment_ids() if preserve_reviewed else set()
        removed = 0
        kept = 0

        def _prune(path: Path, id_key: str) -> None:
            nonlocal removed, kept
            if path is None or not path.exists():
                return
            raw = read_json(path, {"candidates": []})
            cands = list(raw.get("candidates", []))
            if not cands:
                path.unlink(missing_ok=True)
                return
            keep = [c for c in cands if str(c.get(id_key, "")).strip() in reviewed]
            removed += len(cands) - len(keep)
            kept += len(keep)
            if keep:
                raw["candidates"] = keep
                write_json(path, raw)
            else:
                path.unlink(missing_ok=True)

        # Persisted segment queue is keyed by segment_id; external windows use
        # window_id, which equals segment_id for segment-derived windows.
        _prune(
            self._project_root / "derived" / "review_tables" / "candidate_segments.json",
            "segment_id",
        )
        _prune(self._external_windows_path(), "window_id")

        logger.info(
            "Cleared candidate queue: removed=%d, kept_reviewed=%d (preserve_reviewed=%s)",
            removed, kept, preserve_reviewed,
        )
        return {"removed": removed, "kept_reviewed": kept}

    def generate_segment_candidates(
        self,
        config: SegmentCandidateGenerationConfig,
        segment_df: pd.DataFrame | None = None,
    ) -> SegmentCandidateGenerationResult:
        result = SegmentCandidateGenerationResult()
        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        repr_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        pred_path = self._project_root / "derived" / "models" / config.model_version / "segment_predictions.parquet"
        unc_path = self._project_root / "derived" / "models" / config.model_version / "segment_uncertainty.parquet"
        labels_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"

        if segment_df is not None:
            seg_df = segment_df.copy()
        elif repr_path.exists():
            seg_df = pd.read_parquet(repr_path)
        else:
            result.warnings.append(f"Missing segment features: {repr_path}")
            return result
        result.n_segments_loaded = int(len(seg_df))
        if seg_df.empty:
            result.warnings.append("No segment rows found.")
            return result

        if pred_path.exists():
            pred_df = pd.read_parquet(pred_path)
        elif {"prediction_prob"}.issubset(set(seg_df.columns)):
            pred_df = seg_df[["segment_id", "prediction_prob"]].copy()
        else:
            result.warnings.append("Missing segment predictions.")
            return result

        if unc_path.exists():
            unc_df = pd.read_parquet(unc_path)
        elif {"uncertainty_score"}.issubset(set(seg_df.columns)):
            unc_df = seg_df[["segment_id", "uncertainty_score"]].copy()
        else:
            unc_df = pd.DataFrame({"segment_id": seg_df["segment_id"], "uncertainty_score": np.zeros(len(seg_df))})

        merged = seg_df.merge(pred_df[["segment_id", "prediction_prob"]], on="segment_id", how="left")
        merged = merged.merge(unc_df[["segment_id", "uncertainty_score"]], on="segment_id", how="left")

        selected_session_ids = {
            str(s).strip()
            for s in list(getattr(config, "selected_session_ids", []) or [])
            if str(s).strip()
        }
        if selected_session_ids and "session_id" in merged.columns:
            merged = merged[merged["session_id"].astype(str).isin(selected_session_ids)].reset_index(drop=True)
            if merged.empty:
                result.warnings.append(
                    "No segment rows remain after applying selected session scope."
                )
                return result

        # If seg_df already contains these columns, pandas can suffix merged names
        # (e.g., prediction_prob_x / prediction_prob_y). Coalesce to canonical names.
        if "prediction_prob" not in merged.columns:
            pred_candidates = [c for c in ["prediction_prob_y", "prediction_prob_x"] if c in merged.columns]
            if pred_candidates:
                series = merged[pred_candidates[0]].copy()
                for col in pred_candidates[1:]:
                    series = series.fillna(merged[col])
                merged["prediction_prob"] = series
            else:
                merged["prediction_prob"] = 0.0

        if "uncertainty_score" not in merged.columns:
            unc_candidates = [c for c in ["uncertainty_score_y", "uncertainty_score_x"] if c in merged.columns]
            if unc_candidates:
                series = merged[unc_candidates[0]].copy()
                for col in unc_candidates[1:]:
                    series = series.fillna(merged[col])
                merged["uncertainty_score"] = series
            else:
                merged["uncertainty_score"] = 0.0

        merged["prediction_prob"] = merged["prediction_prob"].fillna(0.0)
        merged["uncertainty_score"] = merged["uncertainty_score"].fillna(0.0)
        if bool(config.all_behavior_aware):
            merged = self._attach_all_behavior_competition(
                merged,
                current_model_version=config.model_version,
                co_occurring=bool(config.allow_co_occurring_behaviors),
            )
        else:
            merged["other_behavior_max_prob"] = 0.0
            merged["other_behavior_mean_prob"] = 0.0
            merged["other_behavior_support"] = 0.0
            merged["exclusivity_margin"] = merged["prediction_prob"]
            merged["exclusivity_uncertainty"] = 0.0

        feedback = pd.DataFrame(columns=["segment_id", "review_label"])
        if labels_path.exists():
            feedback = pd.read_parquet(labels_path)

        ranked = self._rank_segments(
            merged,
            feedback=feedback,
            config=config,
        )

        ranked = ranked[
            (ranked["prediction_prob"] >= config.prediction_threshold)
            & (ranked["uncertainty_score"] >= config.uncertainty_threshold)
        ]
        before_non_overlap = int(len(ranked))
        ranked = self._suppress_overlapping_segments(ranked)
        dropped = before_non_overlap - int(len(ranked))
        if dropped > 0:
            result.warnings.append(
                f"Suppressed {dropped} overlapping segment(s); kept highest-probability clips only."
            )

        per_session_cap = max(0, int(getattr(config, "examples_per_session", 0)))
        if per_session_cap > 0 and "session_id" in ranked.columns:
            before_cap = int(len(ranked))
            ranked = self._cap_candidates_per_session(ranked, per_session_cap)
            dropped_by_cap = before_cap - int(len(ranked))
            if dropped_by_cap > 0:
                result.warnings.append(
                    f"Applied per-session candidate cap ({per_session_cap}); dropped {dropped_by_cap} segment(s)."
                )

        result.n_segments_ranked = int(len(ranked))

        top = ranked if int(config.top_k) <= 0 else ranked.head(int(config.top_k))
        result.candidates = self._to_candidate_segments(
            top,
            model_version=config.model_version,
            feature_version=config.feature_version,
            target_behavior_id=config.target_behavior_id,
        )
        result.n_segments_selected = len(result.candidates)
        result.success = True
        return result

    def generate_random_absent_candidates(
        self,
        config: SegmentCandidateGenerationConfig,
    ) -> SegmentCandidateGenerationResult:
        result = SegmentCandidateGenerationResult()
        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        window = max(8, int(config.sample_window_frames))
        selected_session_ids = {
            str(s).strip()
            for s in list(getattr(config, "selected_session_ids", []) or [])
            if str(s).strip()
        }
        session_meta = self._load_sampling_session_metadata(window, selected_session_ids)
        if not session_meta:
            result.warnings.append("No sessions with enough frames for random sampling.")
            return result

        excluded = self._accepted_intervals_from_reviewer_labels()
        selected: dict[str, list[tuple[int, int]]] = {sid: [] for sid in session_meta.keys()}

        rng = np.random.default_rng(config.random_seed)
        session_ids = sorted(session_meta.keys())
        rows: list[dict[str, object]] = []

        if int(config.top_k) <= 0:
            rough_capacity = sum(max(1, int(v["n_frames"]) // window) for v in session_meta.values())
            target = max(1, int(rough_capacity))
        else:
            target = max(1, int(config.top_k))

        # Spread clips across subjects first, then randomize per-subject session choice.
        subject_to_sessions: dict[str, list[str]] = {}
        for sid in session_ids:
            subject = str(session_meta[sid].get("animal_id") or sid)
            subject_to_sessions.setdefault(subject, []).append(sid)

        subject_ids = sorted(subject_to_sessions.keys())
        rng.shuffle(subject_ids)
        n_subjects = len(subject_ids)
        base_per_subject = target // n_subjects
        extra = target % n_subjects
        per_subject_quota: dict[str, int] = {
            subject: base_per_subject + (1 if idx < extra else 0)
            for idx, subject in enumerate(subject_ids)
        }

        selected_by_subject: dict[str, int] = {subject: 0 for subject in subject_ids}

        max_attempts = max(4000, target * 100)
        attempts = 0

        while len(rows) < target and attempts < max_attempts:
            attempts += 1
            # Only pick from subjects that haven't yet met their quota.
            eligible_subjects = [
                subject
                for subject in subject_ids
                if selected_by_subject.get(subject, 0) < per_subject_quota[subject]
            ]
            if not eligible_subjects:
                break

            subject = str(rng.choice(eligible_subjects))
            sid = str(rng.choice(subject_to_sessions[subject]))
            meta = session_meta[sid]
            n_frames = int(meta["n_frames"])

            max_start = max(0, n_frames - window)
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            end = min(n_frames - 1, start + window - 1)

            if self._interval_overlaps_any(start, end, excluded.get(sid, [])):
                continue
            if self._interval_overlaps_any(start, end, selected.get(sid, [])):
                continue

            selected.setdefault(sid, []).append((start, end))
            selected_by_subject[subject] = int(selected_by_subject.get(subject, 0) + 1)
            rows.append(
                {
                    "segment_id": f"rand_{sid}_{start}_{end}",
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "video_id": str(meta["video_id"]),
                    "animal_id": str(meta["animal_id"]),
                    "session_id": sid,
                    "prediction_prob": 0.0,
                    "uncertainty_score": 0.0,
                    "rank_score": float(rng.random()),
                }
            )

        ranked = pd.DataFrame(rows)
        per_session_cap = max(0, int(getattr(config, "examples_per_session", 0)))
        if per_session_cap > 0 and not ranked.empty and "session_id" in ranked.columns:
            before_cap = int(len(ranked))
            ranked = self._cap_candidates_per_session(ranked, per_session_cap)
            dropped_by_cap = before_cap - int(len(ranked))
            if dropped_by_cap > 0:
                result.warnings.append(
                    f"Applied per-session candidate cap ({per_session_cap}); dropped {dropped_by_cap} sampled window(s)."
                )

        result.n_segments_loaded = int(sum(int(v["n_frames"]) for v in session_meta.values()))
        result.n_segments_ranked = int(len(ranked))

        if ranked.empty:
            result.warnings.append("Unable to sample random absent windows after exclusions.")
            return result

        ranked = ranked.sort_values("rank_score", ascending=False)
        result.candidates = self._to_candidate_segments(
            ranked,
            model_version=config.model_version,
            feature_version=config.feature_version,
            target_behavior_id=config.target_behavior_id,
        )
        result.n_segments_selected = len(result.candidates)
        if result.n_segments_selected < target:
            result.warnings.append(
                f"Sampled {result.n_segments_selected}/{target} windows after excluding accepted intervals."
            )
        result.success = True
        return result

    @staticmethod
    def _suppress_overlapping_segments(df: pd.DataFrame) -> pd.DataFrame:
        """Drop temporal overlaps per session/video, keeping highest-probability rows.

        Uses a sorted endpoint list with binary search for O(n log n) overlap
        detection instead of O(n²) linear scan.
        """
        if df.empty:
            return df.copy()

        required = {"start_frame", "end_frame", "prediction_prob"}
        if not required.issubset(set(df.columns)):
            return df.copy()

        group_cols = [c for c in ["session_id", "video_id"] if c in df.columns]
        if not group_cols:
            grouped = [(None, df)]
        else:
            grouped = list(df.groupby(group_cols, sort=False))

        keep_index: list[int] = []

        sort_cols = ["prediction_prob"]
        ascending = [False]
        if "rank_score" in df.columns:
            sort_cols.append("rank_score")
            ascending.append(False)
        if "uncertainty_score" in df.columns:
            sort_cols.append("uncertainty_score")
            ascending.append(False)

        from bisect import bisect_left, insort

        for _, group in grouped:
            ordered = group.sort_values(sort_cols, ascending=ascending)
            # Maintain a sorted list of (end_frame, start_frame) for accepted
            # intervals.  For a new segment [s, e) to overlap an accepted
            # interval [s', e'), we need s < e' AND s' < e.  Sorting by
            # end_frame lets us use bisect to skip intervals whose end_frame
            # is <= s (they can't overlap) and then only check a small window.
            accepted_ends: list[tuple[int, int]] = []  # sorted by end_frame
            starts = ordered["start_frame"].to_numpy(dtype=int)
            ends = ordered["end_frame"].to_numpy(dtype=int)
            indices = ordered.index.to_numpy()

            for i in range(len(starts)):
                s, e = int(starts[i]), int(ends[i])
                # Find first accepted interval whose end_frame > s
                pos = bisect_left(accepted_ends, (s + 1,))
                overlaps = False
                for j in range(pos, len(accepted_ends)):
                    ae, a_s = accepted_ends[j]
                    if a_s >= e:
                        continue
                    # ae > s (guaranteed by bisect) and a_s < e → overlap
                    overlaps = True
                    break
                if not overlaps:
                    keep_index.append(int(indices[i]))
                    insort(accepted_ends, (e, s))

        kept = df.loc[keep_index].copy()
        if "rank_score" in kept.columns:
            return kept.sort_values("rank_score", ascending=False)
        return kept.sort_values("prediction_prob", ascending=False)

    @staticmethod
    def _cap_candidates_per_session(df: pd.DataFrame, max_per_session: int) -> pd.DataFrame:
        """Keep at most ``max_per_session`` rows for each session, preserving rank order."""
        if df.empty or max_per_session <= 0 or "session_id" not in df.columns:
            return df.copy()

        if "rank_score" in df.columns:
            ranked = df.sort_values("rank_score", ascending=False)
        else:
            ranked = df.sort_values("prediction_prob", ascending=False)

        capped = ranked.groupby("session_id", sort=False, group_keys=False).head(int(max_per_session))
        if "rank_score" in capped.columns:
            return capped.sort_values("rank_score", ascending=False)
        return capped.sort_values("prediction_prob", ascending=False)

    def save_segment_candidates(
        self,
        result: SegmentCandidateGenerationResult,
        config: SegmentCandidateGenerationConfig,
    ) -> None:
        if not self._project_root:
            return
        out_path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        write_json(
            out_path,
            {
                "generated_at": datetime.utcnow().isoformat(),
                "config": config.__dict__,
                "n_segments_loaded": result.n_segments_loaded,
                "n_segments_ranked": result.n_segments_ranked,
                "n_segments_selected": result.n_segments_selected,
                "candidates": [c.model_dump(mode="json") for c in result.candidates],
                "warnings": result.warnings,
            },
        )
        self._write_queue_composition_diagnostics(result, config)

    def _write_queue_composition_diagnostics(
        self,
        result: SegmentCandidateGenerationResult,
        config: SegmentCandidateGenerationConfig,
    ) -> None:
        if not self._project_root:
            return
        analysis_dir = self._project_root / "derived" / "analysis" / "diagnostics" / "queue"
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = analysis_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        reasons = [str(c.selection_reason or "unknown") for c in result.candidates]
        reason_counts = pd.Series(reasons).value_counts().to_dict() if reasons else {}

        payload = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "n_selected": int(result.n_segments_selected),
            "config": config.__dict__,
            "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
            "reason_fraction": {
                str(k): float(v) / max(1.0, float(result.n_segments_selected))
                for k, v in reason_counts.items()
            },
        }
        write_json(run_dir / "queue_composition.json", payload)
        write_json(
            analysis_dir / "latest.json",
            {
                "run_id": run_id,
                "queue_composition": str(run_dir / "queue_composition.json"),
            },
        )

        try:
            import matplotlib.pyplot as plt

            if reason_counts:
                labels = list(reason_counts.keys())
                values = [int(reason_counts[k]) for k in labels]
                fig, ax = plt.subplots(figsize=(8.0, 4.6))
                ax.bar(labels, values)
                ax.set_title("Active Learning Queue Composition")
                ax.set_ylabel("Selected windows")
                ax.set_xlabel("Primary selection driver")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(run_dir / "queue_composition.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

    def sync_segment_candidates_to_windows(self) -> int:
        # Deprecated in segment-native flow.
        return len(self.load_segment_candidates())

    def load_segment_candidates(self) -> list[CandidateSegment]:
        if not self._project_root:
            return []
        path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        raw = read_json(path, {"candidates": []})
        rows: list[CandidateSegment] = []
        for item in raw.get("candidates", []):
            try:
                rows.append(CandidateSegment.model_validate(item))
            except Exception:
                continue
        return rows

    def remove_segment_candidates(self, segment_ids: list[str]) -> int:
        """Remove persisted segment candidates by segment_id.

        Returns the number of removed candidate rows.
        """
        if not self._project_root:
            return 0
        if not segment_ids:
            return 0

        path = self._project_root / "derived" / "review_tables" / "candidate_segments.json"
        if not path.exists():
            return 0

        raw = read_json(path, {"candidates": []})
        candidates = list(raw.get("candidates", []))
        if not candidates:
            return 0

        remove_ids = {str(x) for x in segment_ids}
        kept = [
            item
            for item in candidates
            if str(item.get("segment_id", "")) not in remove_ids
        ]

        removed = len(candidates) - len(kept)
        if removed <= 0:
            return 0

        raw["candidates"] = kept
        write_json(path, raw)
        logger.info("Removed %d segment candidates missing clips from %s", removed, path)
        return removed

    @staticmethod
    def _min_distance_to_reviewed(feats: np.ndarray, reviewed_feats: np.ndarray) -> np.ndarray:
        """Compute min Euclidean distance to reviewed features without large 3D allocations."""
        if feats.shape[0] == 0 or reviewed_feats.shape[0] == 0:
            return np.zeros(feats.shape[0], dtype=float)

        feats32 = np.asarray(feats, dtype=np.float32)
        reviewed32 = np.asarray(reviewed_feats, dtype=np.float32)
        reviewed_norm = np.sum(reviewed32 * reviewed32, axis=1)

        # Keep temporary pairwise blocks bounded to avoid O(N*M*D) memory spikes.
        target_pairs = 2_000_000
        chunk_size = max(128, min(4096, target_pairs // max(1, reviewed32.shape[0])))

        min_d = np.empty(feats32.shape[0], dtype=np.float32)
        for start in range(0, feats32.shape[0], chunk_size):
            stop = min(feats32.shape[0], start + chunk_size)
            chunk = feats32[start:stop]
            chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
            sq = np.clip(chunk_norm + reviewed_norm[None, :] - 2.0 * (chunk @ reviewed32.T), 0.0, None)
            min_d[start:stop] = np.sqrt(np.min(sq, axis=1))

        return min_d.astype(float)

    @staticmethod
    def _rank_segments(
        df: pd.DataFrame,
        feedback: pd.DataFrame,
        config: SegmentCandidateGenerationConfig | None = None,
        mode: str = "uncertainty",
        hard_negative_ratio: float = 0.3,
        low_prob_max_prob: float = 0.25,
        random_absent_max_prob: float = 0.35,
        random_seed: int | None = None,
        all_behavior_aware: bool = True,
        all_behavior_competition_margin: float = 0.05,
    ) -> pd.DataFrame:
        out = df.copy()

        if config is None:
            config = SegmentCandidateGenerationConfig(
                mode=mode,
                hard_negative_ratio=float(hard_negative_ratio),
                low_prob_max_prob=float(low_prob_max_prob),
                random_absent_max_prob=float(random_absent_max_prob),
                random_seed=random_seed,
                all_behavior_aware=bool(all_behavior_aware),
                all_behavior_competition_margin=float(all_behavior_competition_margin),
            )

        mode = str(config.mode or "uncertainty")
        hard_negative_ratio = float(config.hard_negative_ratio)
        low_prob_max_prob = float(config.low_prob_max_prob)
        random_absent_max_prob = float(config.random_absent_max_prob)
        random_seed = config.random_seed
        all_behavior_aware = bool(config.all_behavior_aware)
        all_behavior_competition_margin = float(config.all_behavior_competition_margin)

        if "other_behavior_max_prob" not in out.columns:
            out["other_behavior_max_prob"] = 0.0
        if "other_behavior_mean_prob" not in out.columns:
            out["other_behavior_mean_prob"] = 0.0
        if "other_behavior_support" not in out.columns:
            out["other_behavior_support"] = 0.0
        if "exclusivity_margin" not in out.columns:
            out["exclusivity_margin"] = out["prediction_prob"] - out["other_behavior_max_prob"]
        if "exclusivity_uncertainty" not in out.columns:
            out["exclusivity_uncertainty"] = 1.0 - np.clip(np.abs(out["exclusivity_margin"]), 0.0, 1.0)

        if bool(all_behavior_aware):
            margin = float(np.clip(float(all_behavior_competition_margin), 0.0, 1.0))
            shifted = np.clip(
                out["exclusivity_uncertainty"].to_numpy(dtype=float) + margin,
                0.0,
                1.0,
            )
            # Behavior-aware uncertainty: ambiguous windows are those where a competing
            # behavior model scores similarly to the target behavior.
            out["behavior_aware_uncertainty"] = np.maximum(
                out["uncertainty_score"].to_numpy(dtype=float),
                shifted,
            )
        else:
            out["behavior_aware_uncertainty"] = out["uncertainty_score"].to_numpy(dtype=float)

        if not feedback.empty and "review_label" in feedback.columns:
            fb = feedback.groupby("segment_id")["review_label"].agg(lambda s: list(s))

            def _feedback_score(segment_id: str) -> float:
                labels = fb.get(segment_id, [])
                if not labels:
                    return 0.0
                pos = sum(1 for x in labels if str(x) not in {"ambiguous", "boundary_error"} and not str(x).startswith("not_"))
                neg = sum(1 for x in labels if str(x).startswith("not_"))
                amb = sum(1 for x in labels if str(x) in {"ambiguous", "boundary_error"})
                return float((pos - neg) / max(1, pos + neg + amb))

            out["feedback_score"] = out["segment_id"].map(_feedback_score)
        else:
            out["feedback_score"] = 0.0

        # Feature-space distance used for diversity-aware scoring.
        numeric_cols = [
            c
            for c in out.columns
            if c
            not in {
                "segment_id",
                "start_frame",
                "end_frame",
                "animal_id",
                "session_id",
                "video_id",
                "prediction_prob",
                "uncertainty_score",
                "rank_score",
            }
            and pd.api.types.is_numeric_dtype(out[c])
        ]
        if len(numeric_cols) > 80:
            numeric_cols = numeric_cols[:80]

        reviewed_lookup: set[str] = set()
        reviewed_negative_lookup: set[str] = set()
        if not feedback.empty and "segment_id" in feedback.columns:
            reviewed_lookup = set(feedback["segment_id"].astype(str).tolist())
            if "review_label" in feedback.columns:
                reviewed_negative_lookup = set(
                    feedback.loc[
                        feedback["review_label"].astype(str).str.startswith("not_"),
                        "segment_id",
                    ].astype(str).tolist()
                )

        out["candidate_score"] = np.clip(out["prediction_prob"].to_numpy(dtype=float), 0.0, 1.0)
        out["disagreement_score"] = np.clip(out["exclusivity_uncertainty"].to_numpy(dtype=float), 0.0, 1.0)
        out["confound_margin_score"] = np.clip(
            1.0 - np.abs(out["exclusivity_margin"].to_numpy(dtype=float)),
            0.0,
            1.0,
        )

        if numeric_cols:
            feats = out[numeric_cols].to_numpy(dtype=float)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            if reviewed_lookup:
                reviewed_mask = out["segment_id"].astype(str).isin(reviewed_lookup).to_numpy(dtype=bool)
                reviewed_feats = feats[reviewed_mask]
            else:
                reviewed_feats = np.zeros((0, feats.shape[1]), dtype=float)

            if config.enable_diversity_sampling and reviewed_feats.shape[0] > 0 and feats.shape[0] > 0:
                # distance_to_reviewed default: prioritize windows far from reviewed examples.
                d = CandidateGenerationService._min_distance_to_reviewed(feats, reviewed_feats)
                d = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-9)
                out["diversity_score"] = d
            elif config.enable_diversity_sampling and str(config.diversity_mode) == "clustering_quota" and feats.shape[0] > 4:
                try:
                    from sklearn.cluster import MiniBatchKMeans

                    k = int(np.clip(np.sqrt(len(feats)), 4, 24))
                    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
                    labels = km.fit_predict(feats)
                    centers = km.cluster_centers_
                    dist_center = np.sqrt(np.sum((feats - centers[labels]) ** 2, axis=1))
                    # High score for sparse/edge regions to promote coverage.
                    density = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy(dtype=float)
                    raw_div = np.clip(dist_center, 0.0, None) + 0.2 * (1.0 / np.maximum(1.0, density))
                    out["diversity_score"] = (raw_div - np.min(raw_div)) / (np.max(raw_div) - np.min(raw_div) + 1e-9)
                except Exception:
                    out["diversity_score"] = 0.0
            else:
                out["diversity_score"] = 0.0
        else:
            out["diversity_score"] = 0.0

        if config.enable_hard_negative_mining and reviewed_negative_lookup:
            neg_mask = out["segment_id"].astype(str).isin(reviewed_negative_lookup)
            fp_count = int(np.sum(neg_mask))
            if fp_count >= int(max(1, config.min_false_positive_for_hard_negative)):
                out["hard_negative_priority"] = np.where(
                    neg_mask,
                    np.clip(out["prediction_prob"].to_numpy(dtype=float), 0.0, 1.0),
                    np.clip(-out["feedback_score"].to_numpy(dtype=float), 0.0, 1.0),
                )
            else:
                out["hard_negative_priority"] = 0.0
        else:
            out["hard_negative_priority"] = 0.0

        rng = np.random.default_rng(random_seed)
        if float(config.exploration_fraction) > 0:
            if "session_id" in out.columns:
                session_counts = out["session_id"].astype(str).value_counts()
                session_weight = out["session_id"].astype(str).map(
                    lambda s: 1.0 / max(1, int(session_counts.get(s, 1)))
                ).to_numpy(dtype=float)
            else:
                session_weight = np.ones(len(out), dtype=float)
            noise = rng.random(len(out))
            explore = 0.7 * noise + 0.3 * session_weight
            out["exploration_bonus"] = np.clip(explore, 0.0, 1.0)
        else:
            out["exploration_bonus"] = 0.0

        if mode == "prototype":
            out["rank_score"] = (
                out["prediction_prob"]
                + 0.10 * out["feedback_score"]
                + 0.20 * out["exclusivity_uncertainty"]
            )
        elif mode == "novelty":
            novelty = 1.0 - out["prediction_prob"].to_numpy(dtype=float)
            out["rank_score"] = novelty + out["behavior_aware_uncertainty"]
        elif mode == "low_probability":
            prob = out["prediction_prob"].to_numpy(dtype=float)
            unc = out["behavior_aware_uncertainty"].to_numpy(dtype=float)
            low_prob_bonus = np.clip(1.0 - prob, 0.0, 1.0)
            confident_negative_bonus = np.clip(1.0 - unc, 0.0, 1.0)
            reviewed_negative_bonus = np.clip(-out["feedback_score"].to_numpy(dtype=float), 0.0, 1.0)
            absent_mask = (prob <= float(np.clip(low_prob_max_prob, 0.0, 1.0))).astype(float)
            competitor_bonus = np.clip(out["other_behavior_max_prob"].to_numpy(dtype=float) - prob, 0.0, 1.0)
            out["rank_score"] = (
                low_prob_bonus
                + 0.35 * confident_negative_bonus
                + 0.25 * reviewed_negative_bonus
                + 0.40 * absent_mask
                + 0.20 * competitor_bonus
            )
        elif mode == "random_absent":
            prob = out["prediction_prob"].to_numpy(dtype=float)
            max_prob = float(np.clip(random_absent_max_prob, 0.0, 1.0))
            eligible = prob <= max_prob
            if not bool(np.any(eligible)):
                eligible = np.ones(len(out), dtype=bool)
            rng = np.random.default_rng(random_seed)
            out["rank_score"] = -1.0
            out.loc[eligible, "rank_score"] = rng.random(int(np.sum(eligible)))
        else:
            out["rank_score"] = (
                out["behavior_aware_uncertainty"]
                + 0.5 * out["prediction_prob"]
                + 0.2 * out["feedback_score"]
                + 0.2 * out["exclusivity_uncertainty"]
            )

        # Weighted queue scoring is fully opt-in; baseline ranking remains unchanged otherwise.
        if bool(config.enable_weighted_queue_scoring) and mode != "random_absent":
            w_candidate = float(max(0.0, config.queue_weight_candidate))
            w_unc = float(max(0.0, config.queue_weight_uncertainty if config.enable_uncertainty_sampling else 0.0))
            w_dis = float(max(0.0, config.queue_weight_disagreement if config.enable_expert_disagreement else 0.0))
            w_div = float(max(0.0, config.queue_weight_diversity if config.enable_diversity_sampling else 0.0))
            w_conf = float(max(0.0, config.queue_weight_confound if config.enable_confound_sampling else 0.0))
            w_hn = float(max(0.0, config.queue_weight_hard_negative if config.enable_hard_negative_mining else 0.0))
            w_exp = float(max(0.0, config.queue_weight_exploration * max(0.0, float(config.exploration_fraction))))

            raw = (
                w_candidate * out["candidate_score"].to_numpy(dtype=float)
                + w_unc * out["behavior_aware_uncertainty"].to_numpy(dtype=float)
                + w_dis * out["disagreement_score"].to_numpy(dtype=float)
                + w_div * out["diversity_score"].to_numpy(dtype=float)
                + w_conf * out["confound_margin_score"].to_numpy(dtype=float)
                + w_hn * out["hard_negative_priority"].to_numpy(dtype=float)
                + w_exp * out["exploration_bonus"].to_numpy(dtype=float)
            )
            denom = max(1e-9, w_candidate + w_unc + w_dis + w_div + w_conf + w_hn + w_exp)
            out["final_priority_score"] = raw / denom
            out["rank_score"] = out["final_priority_score"]

            comp_cols = [
                "candidate_score",
                "behavior_aware_uncertainty",
                "disagreement_score",
                "diversity_score",
                "confound_margin_score",
                "hard_negative_priority",
                "exploration_bonus",
            ]
            comp_arr = out[comp_cols].to_numpy(dtype=float)
            max_idx = np.argmax(comp_arr, axis=1)
            reason_map = {
                0: "candidate",
                1: "uncertainty",
                2: "disagreement",
                3: "diversity",
                4: "confound_boundary",
                5: "hard_negative",
                6: "exploration",
            }
            out["selection_reason"] = [reason_map.get(int(i), "candidate") for i in max_idx]
        else:
            out["final_priority_score"] = out["rank_score"].to_numpy(dtype=float)
            out["selection_reason"] = "baseline"

        if hard_negative_ratio > 0 and mode in {"prototype", "novelty", "uncertainty"}:
            hard_mask = (out["prediction_prob"] > 0.6) & (out["feedback_score"] < 0)
            out.loc[hard_mask, "rank_score"] += hard_negative_ratio
            out.loc[hard_mask, "final_priority_score"] += hard_negative_ratio

        return out.sort_values("rank_score", ascending=False)

    def _attach_all_behavior_competition(
        self,
        merged: pd.DataFrame,
        current_model_version: str,
        co_occurring: bool = False,
    ) -> pd.DataFrame:
        """Attach cross-behavior competition features using available model predictions.

        This keeps candidate IDs unchanged while making ranking aware of
        behavior competition.  When *co_occurring* is True, high scores from
        peer behavior models do **not** penalise ranking because overlapping
        behaviors are expected.
        """
        if not self._project_root or merged.empty or "segment_id" not in merged.columns:
            out = merged.copy()
            out["other_behavior_max_prob"] = 0.0
            out["other_behavior_mean_prob"] = 0.0
            out["other_behavior_support"] = 0.0
            out["exclusivity_margin"] = out.get("prediction_prob", 0.0)
            out["exclusivity_uncertainty"] = 0.0
            return out

        models_root = self._project_root / "derived" / "models"
        if not models_root.exists():
            out = merged.copy()
            out["other_behavior_max_prob"] = 0.0
            out["other_behavior_mean_prob"] = 0.0
            out["other_behavior_support"] = 0.0
            out["exclusivity_margin"] = out["prediction_prob"]
            out["exclusivity_uncertainty"] = 1.0 - np.clip(np.abs(out["prediction_prob"]), 0.0, 1.0)
            return out

        peers: list[pd.Series] = []
        # Collect peer prediction parquet paths first, then read in parallel.
        peer_paths: list[tuple[str, Path]] = []
        for model_dir in sorted(models_root.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name == str(current_model_version):
                continue
            if not model_dir.name.startswith("behavior_model_"):
                continue
            pred_path = model_dir / "segment_predictions.parquet"
            if pred_path.exists():
                peer_paths.append((model_dir.name, pred_path))

        def _read_peer(item: tuple[str, Path]) -> pd.Series | None:
            name, path = item
            try:
                pred_df = pd.read_parquet(path, columns=["segment_id", "prediction_prob"])
                if pred_df.empty:
                    return None
                return pred_df.set_index("segment_id")["prediction_prob"].rename(f"peer_prob__{name}")
            except Exception:
                return None

        if peer_paths:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, len(peer_paths))) as pool:
                for result in pool.map(_read_peer, peer_paths):
                    if result is not None:
                        peers.append(result)

        out = merged.copy()
        if not peers:
            out["other_behavior_max_prob"] = 0.0
            out["other_behavior_mean_prob"] = 0.0
            out["other_behavior_support"] = 0.0
            out["exclusivity_margin"] = out["prediction_prob"]
            out["exclusivity_uncertainty"] = 1.0 - np.clip(np.abs(out["prediction_prob"]), 0.0, 1.0)
            return out

        peer_df = pd.concat(peers, axis=1)
        joined = out[["segment_id"]].merge(peer_df, left_on="segment_id", right_index=True, how="left")
        peer_cols = [c for c in joined.columns if c.startswith("peer_prob__")]
        peer_vals = joined[peer_cols].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            peer_max = np.nanmax(peer_vals, axis=1)
            peer_mean = np.nanmean(peer_vals, axis=1)
        peer_max = np.where(np.isnan(peer_max), 0.0, peer_max)
        peer_mean = np.where(np.isnan(peer_mean), 0.0, peer_mean)
        peer_support = np.sum(np.nan_to_num(peer_vals, nan=0.0) >= 0.5, axis=1).astype(float)

        out["other_behavior_max_prob"] = peer_max
        out["other_behavior_mean_prob"] = peer_mean
        out["other_behavior_support"] = peer_support
        if co_occurring:
            # In co-occurring mode, peer behavior scores do not reduce
            # exclusivity — a segment genuinely expressing multiple behaviors
            # is not ambiguous, so we keep margin at prediction_prob and
            # exclusivity_uncertainty at zero.
            out["exclusivity_margin"] = out["prediction_prob"].to_numpy(dtype=float)
            out["exclusivity_uncertainty"] = 0.0
        else:
            out["exclusivity_margin"] = out["prediction_prob"].to_numpy(dtype=float) - peer_max
            out["exclusivity_uncertainty"] = 1.0 - np.clip(np.abs(out["exclusivity_margin"].to_numpy(dtype=float)), 0.0, 1.0)
        return out

    def _to_candidate_segments(
        self,
        ranked: pd.DataFrame,
        model_version: str,
        feature_version: str,
        target_behavior_id: str | None = None,
    ) -> list[CandidateSegment]:
        assert self._project_root is not None

        pose_cols = [c for c in ranked.columns if "forepaw" in c or "nose_oscillation" in c or "nose_autocorr" in c or "nose_movement_freq" in c or c.endswith("_speed_mean")]
        ctx_cols = [
            c
            for c in ranked.columns
            if "TMT" in c
            or "target" in c
            or "bedding" in c
            or "substrate" in c
            or "flow_" in c
        ]
        prov = self._provenance.make_provenance(
            project_root=self._project_root,
            model_version=model_version,
            feature_version=feature_version,
            config={"rows": int(len(ranked)), "columns": list(ranked.columns)},
        )

        # Pre-extract numpy arrays for hot columns to avoid per-row attribute lookups.
        seg_ids = ranked["segment_id"].astype(str).to_numpy()
        starts = ranked["start_frame"].to_numpy(dtype=int)
        ends = ranked["end_frame"].to_numpy(dtype=int)
        vid_ids = ranked["video_id"].astype(str).to_numpy() if "video_id" in ranked.columns else None
        anim_ids = ranked["animal_id"].astype(str).to_numpy() if "animal_id" in ranked.columns else None
        sess_ids = ranked["session_id"].astype(str).to_numpy() if "session_id" in ranked.columns else None
        pred_probs = ranked["prediction_prob"].to_numpy(dtype=float)
        unc_scores = ranked["uncertainty_score"].to_numpy(dtype=float)

        cand_scores = ranked["candidate_score"].to_numpy(dtype=float) if "candidate_score" in ranked.columns else np.zeros(len(ranked))
        ba_unc = ranked["behavior_aware_uncertainty"].to_numpy(dtype=float) if "behavior_aware_uncertainty" in ranked.columns else unc_scores
        disag = ranked["disagreement_score"].to_numpy(dtype=float) if "disagreement_score" in ranked.columns else np.zeros(len(ranked))
        div_sc = ranked["diversity_score"].to_numpy(dtype=float) if "diversity_score" in ranked.columns else np.zeros(len(ranked))
        conf_sc = ranked["confound_margin_score"].to_numpy(dtype=float) if "confound_margin_score" in ranked.columns else np.zeros(len(ranked))
        hn_sc = ranked["hard_negative_priority"].to_numpy(dtype=float) if "hard_negative_priority" in ranked.columns else np.zeros(len(ranked))
        exp_sc = ranked["exploration_bonus"].to_numpy(dtype=float) if "exploration_bonus" in ranked.columns else np.zeros(len(ranked))
        final_sc = (ranked["final_priority_score"].to_numpy(dtype=float) if "final_priority_score" in ranked.columns
                    else ranked["rank_score"].to_numpy(dtype=float) if "rank_score" in ranked.columns
                    else np.zeros(len(ranked)))
        sel_reasons = ranked["selection_reason"].astype(str).to_numpy() if "selection_reason" in ranked.columns else None

        pose_arr = ranked[pose_cols].to_numpy(dtype=float) if pose_cols else None
        ctx_arr = ranked[ctx_cols].to_numpy(dtype=float) if ctx_cols else None

        bid = str(target_behavior_id) if target_behavior_id else None

        out: list[CandidateSegment] = []
        for i in range(len(ranked)):
            pose = dict(zip(pose_cols, pose_arr[i].tolist())) if pose_arr is not None else {}
            ctx = dict(zip(ctx_cols, ctx_arr[i].tolist())) if ctx_arr is not None else {}
            out.append(
                CandidateSegment(
                    segment_id=seg_ids[i],
                    start_frame=int(starts[i]),
                    end_frame=int(ends[i]),
                    video_id=vid_ids[i] if vid_ids is not None else "unknown_video",
                    animal_id=anim_ids[i] if anim_ids is not None else "unknown_animal",
                    session_id=sess_ids[i] if sess_ids is not None else "unknown_session",
                    prediction_prob=float(pred_probs[i]),
                    uncertainty_score=float(unc_scores[i]),
                    behavior_id=bid,
                    pose_features=pose,
                    context_features=ctx,
                    score_components={
                        "candidate_score": float(cand_scores[i]),
                        "uncertainty_score": float(ba_unc[i]),
                        "disagreement_score": float(disag[i]),
                        "diversity_score": float(div_sc[i]),
                        "confound_margin_score": float(conf_sc[i]),
                        "hard_negative_priority": float(hn_sc[i]),
                        "exploration_bonus": float(exp_sc[i]),
                        "final_priority_score": float(final_sc[i]),
                    },
                    final_priority_score=float(final_sc[i]),
                    selection_reason=sel_reasons[i] if sel_reasons is not None else "baseline",
                    model_version=model_version,
                    feature_version=feature_version,
                    provenance=prov,
                )
            )
        return out

    def _load_sampling_session_metadata(
        self,
        min_window_frames: int,
        selected_session_ids: set[str] | None = None,
    ) -> dict[str, dict[str, object]]:
        assert self._project_root is not None

        out: dict[str, dict[str, object]] = {}
        summaries_path = self._project_root / "derived" / "pose_features" / "summaries.json"
        manifest_path = self._project_root / "derived" / "review_tables" / "import_manifest.json"

        summary_rows = read_json(summaries_path, {"summaries": []}).get("summaries", [])
        frames_by_session = {
            str(item.get("session_id", "")): int(item.get("n_frames", 0) or 0)
            for item in summary_rows
            if str(item.get("session_id", ""))
        }

        manifest = read_json(manifest_path, {}) if manifest_path.exists() else {}
        videos = {str(v.get("asset_id", "")): v for v in manifest.get("videos", [])}
        linked = manifest.get("linked_sessions", [])
        selected = {str(s).strip() for s in (selected_session_ids or set()) if str(s).strip()}

        for sess in linked:
            sid = str(sess.get("session_id", "")).strip()
            if not sid:
                continue
            if selected and sid not in selected:
                continue
            video_id = str(sess.get("video_asset_id", "")).strip()
            video = videos.get(video_id, {})
            n_frames = int(frames_by_session.get(sid) or video.get("frame_count") or 0)
            if n_frames < int(min_window_frames):
                continue
            out[sid] = {
                "n_frames": n_frames,
                "animal_id": str(sess.get("subject_id") or sid),
                "video_id": video_id or sid,
            }

        # Fallback: summaries without manifest entries.
        for sid, n_frames in frames_by_session.items():
            if sid in out or n_frames < int(min_window_frames):
                continue
            if selected and sid not in selected:
                continue
            out[sid] = {
                "n_frames": int(n_frames),
                "animal_id": sid,
                "video_id": sid,
            }

        return out

    def _accepted_intervals_from_reviewer_labels(self) -> dict[str, list[tuple[int, int]]]:
        assert self._project_root is not None

        path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        out: dict[str, list[tuple[int, int]]] = {}
        if not path.exists():
            return out

        try:
            labels = pd.read_parquet(path)
        except Exception:
            return out

        if labels.empty or "segment_id" not in labels.columns or "review_label" not in labels.columns:
            return out

        for segment_id, review_label in labels[["segment_id", "review_label"]].itertuples(index=False, name=None):
            label = str(review_label or "").strip()
            if not label or label in {"ambiguous", "boundary_error"} or label.startswith("not_"):
                continue
            parsed = self._segment_interval_from_id(str(segment_id))
            if parsed is None:
                continue
            sid, start, end = parsed
            out.setdefault(sid, []).append((start, end))

        for sid, intervals in list(out.items()):
            out[sid] = self._merge_intervals(intervals)
        return out

    @staticmethod
    def _segment_interval_from_id(segment_id: str) -> tuple[str, int, int] | None:
        text = str(segment_id or "").strip()
        if not text:
            return None
        parts = text.split("_")
        if len(parts) < 4:
            return None
        try:
            end = int(parts[-1])
            start = int(parts[-2])
        except ValueError:
            return None

        sid_idx = -1
        for i, token in enumerate(parts):
            if token == "session" and i + 1 < len(parts):
                sid_idx = i
                break
        if sid_idx < 0:
            return None
        sid = "_".join(parts[sid_idx : sid_idx + 2])
        if not sid.startswith("session_"):
            return None
        return sid, int(start), int(end)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        cleaned = sorted((min(int(s), int(e)), max(int(s), int(e))) for s, e in intervals)
        merged: list[tuple[int, int]] = [cleaned[0]]
        for s, e in cleaned[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def _interval_overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
        s0 = int(min(start, end))
        e0 = int(max(start, end))
        return any(max(0, min(e0, e1) - max(s0, s1) + 1) > 0 for s1, e1 in intervals)

    @staticmethod
    def _segment_to_window_candidates(
        segments: list[CandidateSegment],
        target_behavior_id: str | None = None,
    ) -> list[CandidateWindow]:
        out: list[CandidateWindow] = []
        behavior_value = str(target_behavior_id or "").strip() or None
        for seg in segments:
            # Use the segment's own behavior_id if present, otherwise fall back to target.
            seg_behavior = str(getattr(seg, "behavior_id", "") or "").strip() or None
            effective_behavior = seg_behavior or behavior_value
            # Preserve segment identity as clip/window id for downstream label joins.
            out.append(
                CandidateWindow(
                    window_id=seg.segment_id,
                    session_id=seg.session_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    behavior_id=effective_behavior,
                    seed_similarity_score=1.0 - float(seg.uncertainty_score),
                    total_score=float(0.7 * seg.prediction_prob + 0.3 * (1.0 - seg.uncertainty_score)),
                    clip_path=None,
                    selection_reason=getattr(seg, "selection_reason", "") or "",
                )
            )
        return out
