"""Validation service: model-quality overview, quiz assembly, and metrics.

This service powers the Validation tab.  It is deliberately self-contained and
file-backed (mirroring the other project-scoped services) so it can be reused by
the UI and exercised directly in tests.

Responsibilities
----------------
* ``model_overview`` — aggregate per-behavior model quality + label/bout counts.
* ``assemble_run`` — build a blind quiz (a :class:`ValidationRun`) by sampling
  clips from four categories (prior-accepted, unreviewed-positive bouts, clearly
  negative regions, and near-threshold fringe) and extracting the needed video
  clips on the fly.
* ``compute_metrics`` / ``suggestions`` — user-vs-machine and inter-rater
  (user-vs-user) metrics plus rule-based improvement guidance.
* persistence of runs/answers under ``derived/validation/`` and an opt-in
  write-back of reviewer answers into training labels.

Data sources are reused, never recomputed: probability traces and bouts come
from the temporal-refinement outputs, model metrics from ``derived/models`` /
``derived/evaluation``, and prior decisions from the review tables.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from abel.models.schemas import (
    CandidateWindow,
    ReviewDecisionType,
    ReviewerLabelRecord,
    ValidationAnswerRecord,
    ValidationClipRecord,
    ValidationRun,
    ValidationSettings,
)
from abel.services.behavior_service import BehaviorService
from abel.services.candidate_service import CandidateGenerationService
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.services.preprocessing_service import ClipExtractionConfig, ClipExtractionService
from abel.services.review_service import ReviewService
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml
from abel.temporal_refinement.bout_postprocess import (
    binary_trace_to_intervals,
    merge_close_bouts,
    remove_short_bouts,
    threshold_probabilities,
)

logger = logging.getLogger("abel")

NO_BEHAVIOR_ID = "no_behavior"
_DEFAULT_THRESHOLD = 0.65
_DEFAULT_MIN_BOUT = 8
_DEFAULT_MERGE_GAP = 4


class GridCellSpec(NamedTuple):
    """One bout selected for a Behavior Grid cell."""

    session_id: str
    bout_start: int
    bout_end: int
    mean_prob: float


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return safe or "target_behavior"


def _is_no_behavior(token: str) -> bool:
    norm = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(token or "").strip()).strip("_")
    return norm in {"no_behavior", "no_behaviour", "nobehavior", "nobehaviour"}


class ValidationService:
    """Assembles validation quizzes and computes reviewer/model agreement metrics."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._reviews = ReviewService()
        self._candidates = CandidateGenerationService()
        self._clips = ClipExtractionService()
        self._behaviors = BehaviorService()
        self._pose = PoseProcessingService()
        self._traces_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._competition_ids: set[str] | None | str = "unset"

    # ------------------------------------------------------------------
    # Project wiring
    # ------------------------------------------------------------------
    def set_project(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._reviews.set_project(self._project_root)
        self._candidates.set_project(self._project_root)
        self._clips.set_project(self._project_root)
        self._behaviors.set_project(self._project_root)
        self._traces_cache = {}
        self._competition_ids = "unset"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _root(self) -> Path:
        if self._project_root is None:
            raise RuntimeError("ValidationService has no project set.")
        return self._project_root

    def _validation_dir(self) -> Path:
        return self._root() / "derived" / "validation"

    def _runs_dir(self) -> Path:
        return self._validation_dir() / "runs"

    def _answers_dir(self) -> Path:
        return self._validation_dir() / "answers"

    def _clips_dir(self) -> Path:
        return self._validation_dir() / "clips"

    def _active_run_path(self) -> Path:
        return self._validation_dir() / "active_run.json"

    def _settings_path(self) -> Path:
        return self._root() / "config" / "validation_settings.yaml"

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def load_settings(self) -> ValidationSettings:
        if self._project_root is None:
            return ValidationSettings()
        raw = read_yaml(self._settings_path(), {})
        if not raw:
            return ValidationSettings()
        try:
            return ValidationSettings.model_validate(raw)
        except Exception:
            return ValidationSettings()

    def save_settings(self, settings: ValidationSettings) -> None:
        if self._project_root is None:
            return
        write_yaml(self._settings_path(), settings.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Active behaviors helper
    # ------------------------------------------------------------------
    def _competition_behavior_ids(self) -> set[str] | None:
        """Behavior ids that took part in the dense-inference competition, or None.

        Behaviors excluded from temporal refinement have no probability column or
        thresholds, so they must be excluded from the validation test.  Returns the
        set listed under the inference manifest's ``competition.behavior_models``,
        or ``None`` when there is no competition manifest (don't filter).
        """
        if self._competition_ids != "unset":
            return self._competition_ids  # type: ignore[return-value]
        result: set[str] | None = None
        latest = self._read_latest("target_behavior")
        inference_dir = str(latest.get("inference_dir", "") or "").strip()
        if inference_dir:
            man = read_json(Path(inference_dir) / "inference_manifest.json", {})
            comp = man.get("competition") or {}
            models = comp.get("behavior_models") or {}
            if models:
                result = {str(k) for k in models}
        self._competition_ids = result
        return result

    def _active_behaviors(self) -> list[tuple[str, str]]:
        """Return (behavior_id, name) for every active behavior in the competition.

        Excludes ``no_behavior`` (the negative class) and any behavior that was
        excluded from temporal refinement (no thresholds → cannot be tested).
        Used for quiz sampling, overlap, and the overview.
        """
        self._behaviors.set_project(self._root())
        competition = self._competition_behavior_ids()
        out: list[tuple[str, str]] = []
        for b in self._behaviors.behaviors:
            bid = str(b.behavior_id or "").strip()
            name = str(b.name or bid).strip() or bid
            if not bid or _is_no_behavior(bid) or not getattr(b, "is_active", True):
                continue
            if competition is not None and bid not in competition:
                continue
            out.append((bid, name))
        return out

    def _overview_behaviors(self) -> list[tuple[str, str]]:
        """Active behaviors plus ``no_behavior`` (its own model in the overview)."""
        self._behaviors.set_project(self._root())
        out = list(self._active_behaviors())
        for b in self._behaviors.behaviors:
            bid = str(b.behavior_id or "").strip()
            if bid and _is_no_behavior(bid) and getattr(b, "is_active", True):
                out.append((bid, str(b.name or "No Behavior").strip() or "No Behavior"))
                break
        return out

    # ==================================================================
    # Overview
    # ==================================================================
    def model_overview(self) -> list[dict[str, Any]]:
        """One summary row per active behavior model."""
        rows: list[dict[str, Any]] = []
        snapshot_models = self._behavior_model_map()
        label_counts, total_labels = self._label_counts_by_behavior()
        bout_counts = self._bout_counts_by_behavior()
        overlap = self.overlap_analysis().get("per_behavior", {})

        for bid, name in self._overview_behaviors():
            model_dir = self._resolve_model_dir(bid, name, snapshot_models.get(bid, ""))
            model_version = model_dir.name if model_dir is not None else ""
            metrics = self._read_model_metrics(model_dir)
            pos = label_counts.get(bid, 0)
            # One-vs-rest: a behavior's negatives are every label that isn't it.
            neg = max(0, total_labels - pos) if total_labels else 0
            row = {
                "behavior_id": bid,
                "behavior_name": name,
                "model_version": model_version or "—",
                "frame_f1": metrics.get("frame_f1"),
                "frame_precision": metrics.get("frame_precision"),
                "frame_recall": metrics.get("frame_recall"),
                "pr_auc": metrics.get("pr_auc"),
                "n_train": metrics.get("n_train"),
                "n_val": metrics.get("n_val"),
                "calibration": metrics.get("calibration"),
                "last_trained": metrics.get("last_trained"),
                "n_positive_labels": pos,
                "n_negative_labels": neg,
                "n_bouts": bout_counts.get(bid, 0),
                "overlap_fraction": (overlap.get(bid) or {}).get("overlap_fraction"),
            }
            row["quality"] = self._quality_badge(row.get("frame_f1"))
            rows.append(row)
        return rows

    @staticmethod
    def _quality_badge(f1: float | None) -> str:
        if f1 is None:
            return "unknown"
        try:
            v = float(f1)
        except (TypeError, ValueError):
            return "unknown"
        if v >= 0.80:
            return "good"
        if v >= 0.60:
            return "fair"
        return "poor"

    def _behavior_model_map(self) -> dict[str, str]:
        """Map behavior_id -> model_version from the workflow snapshot when present."""
        snap_path = self._root() / "derived" / "workflow_snapshot.json"
        raw = read_json(snap_path, {})
        sbm = raw.get("selected_behavior_models") or {}
        return {str(k): str(v) for k, v in sbm.items() if str(v).strip()}

    def _resolve_model_dir(self, bid: str, name: str, snapshot_version: str) -> Path | None:
        """Locate a behavior's model directory under derived/models.

        Tries, in order: the snapshot-recorded version, then the conventional
        ``behavior_model_<SafeName>`` directory built from the behavior name and
        finally from its id.  Returns the first existing directory.
        """
        models_root = self._root() / "derived" / "models"
        candidates: list[str] = []
        if snapshot_version:
            candidates.append(snapshot_version)
        for token in (name, bid):
            safe = _safe_name(token)
            candidates.append(f"behavior_model_{safe}")
            candidates.append(safe)
        seen: set[str] = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            mdir = models_root / cand
            if (mdir / "metrics.json").exists() or (mdir / "model_card.yaml").exists():
                return mdir
        return None

    def _read_model_metrics(self, model_dir: Path | None) -> dict[str, Any]:
        """Read a behavior model's validation metrics from its model directory.

        Falls back to the aggregate derived/evaluation/model_metrics.json only
        when no per-behavior model directory is available.
        """
        out: dict[str, Any] = {}

        def _f(v: Any) -> float | None:
            try:
                f = float(v)
                return f if np.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        if model_dir is not None:
            metrics = read_json(model_dir / "metrics.json", {})
            if metrics:
                out["frame_f1"] = _f(metrics.get("f1"))
                out["frame_precision"] = _f(metrics.get("precision"))
                out["frame_recall"] = _f(metrics.get("recall"))
                out["pr_auc"] = _f(metrics.get("pr_auc"))
                out["n_train"] = metrics.get("n_train")
                out["n_val"] = metrics.get("n_val")
            card = read_yaml(model_dir / "model_card.yaml", {})
            if card:
                out["calibration"] = card.get("calibration_method")
                prov = card.get("provenance") or {}
                out["last_trained"] = prov.get("timestamp")

        if "frame_f1" not in out:
            # Aggregate evaluation fallback (single-model projects only).
            agg_path = self._root() / "derived" / "evaluation" / "model_metrics.json"
            if agg_path.exists():
                try:
                    text = agg_path.read_text(encoding="utf-8")
                    import re  # noqa: PLC0415

                    text = re.sub(r"\bNaN\b", "null", text)
                    agg = json.loads(text)
                    fl = agg.get("frame_level") or {}
                    out.setdefault("frame_f1", _f(fl.get("f1")))
                    out.setdefault("frame_precision", _f(fl.get("precision")))
                    out.setdefault("frame_recall", _f(fl.get("recall")))
                    out.setdefault("pr_auc", _f(fl.get("pr_auc")))
                except Exception:
                    pass
        return out

    def _label_counts_by_behavior(self) -> tuple[dict[str, int], int]:
        """Count labels per behavior (``no_behavior`` is its own key), plus the grand total.

        Combines accepted review decisions and reviewer segment labels so the
        overview reflects all human labeling.  The total drives one-vs-rest
        negative counts (a behavior's negatives are every label that isn't it).
        """
        counts: dict[str, int] = defaultdict(int)

        def _key(label: str) -> str:
            return NO_BEHAVIOR_ID if _is_no_behavior(label) else label

        for d in self._reviews.load_decisions():
            label = str(d.behavior_label or "").strip()
            if d.decision == ReviewDecisionType.ACCEPT and label:
                counts[_key(label)] += 1
        for rec in self._reviews.load_segment_labels():
            label = str(rec.review_label or "").strip()
            if label:
                counts[_key(label)] += 1
        total = sum(counts.values())
        return dict(counts), total

    def _bout_counts_by_behavior(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for bid, _ in self._active_behaviors():
            total = 0
            for _sid, df in self._load_traces(bid).items():
                total += len(self._detect_bouts(bid, df))
            counts[bid] = total
        return counts

    # ==================================================================
    # Behavior overlap (simultaneous detections)
    # ==================================================================
    def overlap_analysis(self) -> dict[str, Any]:
        """Measure how often behaviors are flagged at the same frames.

        ABEL has no hard rule preventing two behaviors from being detected at the
        same time, so bouts can overlap.  Using each behavior's own threshold/bout
        settings we build per-frame "active" masks from the competition trace and
        report, per behavior, the fraction of its detected frames shared with any
        other behavior, plus the worst-offending behavior pairs.  High overlap
        suggests thresholds are too lax or behavior inhibition is too weak.
        """
        empty = {"overall_overlap_rate": None, "per_behavior": {}, "pairs": []}
        if self._project_root is None:
            return empty
        behaviors = self._active_behaviors()
        if len(behaviors) < 2:
            return empty
        ids = [bid for bid, _ in behaviors]
        sess_traces = self._load_traces(ids[0])
        if not sess_traces:
            return empty

        active_frames: dict[str, int] = defaultdict(int)
        overlap_frames: dict[str, int] = defaultdict(int)
        pair_frames: dict[tuple[str, str], int] = defaultdict(int)
        total_any = 0
        total_multi = 0

        for _sid, df in sess_traces.items():
            if df.empty:
                continue
            n = int(df["frame"].max()) + 1
            if n <= 0:
                continue
            actives: dict[str, np.ndarray] = {}
            for bid in ids:
                arr = np.zeros(n, dtype=bool)
                for s, e, _m in self._detect_bouts(bid, df):
                    if 0 <= s <= e < n:
                        arr[s : e + 1] = True
                actives[bid] = arr
            stack = np.vstack([actives[bid] for bid in ids])
            per_frame = stack.sum(axis=0)
            any_mask = per_frame >= 1
            multi_mask = per_frame >= 2
            total_any += int(any_mask.sum())
            total_multi += int(multi_mask.sum())
            for bid in ids:
                active_frames[bid] += int(actives[bid].sum())
                overlap_frames[bid] += int((actives[bid] & multi_mask).sum())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ov = int((actives[ids[i]] & actives[ids[j]]).sum())
                    if ov:
                        pair_frames[(ids[i], ids[j])] += ov

        per_behavior = {
            bid: {
                "active_frames": active_frames[bid],
                "overlap_frames": overlap_frames[bid],
                "overlap_fraction": (
                    overlap_frames[bid] / active_frames[bid] if active_frames[bid] else None
                ),
            }
            for bid in ids
        }
        pairs = []
        for (a, b), ov in sorted(pair_frames.items(), key=lambda kv: kv[1], reverse=True):
            pairs.append({
                "a": a,
                "b": b,
                "overlap_frames": ov,
                "frac_a": (ov / active_frames[a]) if active_frames[a] else None,
                "frac_b": (ov / active_frames[b]) if active_frames[b] else None,
            })
        return {
            "overall_overlap_rate": (total_multi / total_any) if total_any else None,
            "per_behavior": per_behavior,
            "pairs": pairs,
        }

    def _assign_coactive_labels(self, clips: list[ValidationClipRecord]) -> None:
        """Populate each clip's ``coactive_labels`` = behaviors flagged at its center frame.

        Used to detect ambiguous clips where the model asserts two or more behaviors
        at once, so they can be excluded from user-vs-machine scoring.
        """
        if self._project_root is None:
            return
        ids = [bid for bid, _ in self._active_behaviors()]
        if not ids or not clips:
            for c in clips:
                c.coactive_labels = []
            return
        sess_traces = self._load_traces(ids[0])
        by_session: dict[str, list[ValidationClipRecord]] = defaultdict(list)
        for c in clips:
            by_session[c.session_id].append(c)
        for sid, recs in by_session.items():
            df = sess_traces.get(sid)
            if df is None or df.empty:
                for c in recs:
                    c.coactive_labels = []
                continue
            intervals = {bid: self._detect_bouts(bid, df) for bid in ids}
            for c in recs:
                center = (int(c.start_frame) + int(c.end_frame)) // 2
                c.coactive_labels = [
                    bid for bid in ids
                    if any(s <= center <= e for s, e, _m in intervals[bid])
                ]

    def apply_inhibition(self, behavior_a: str, behavior_b: str, weight: float | None = None) -> float:
        """Add mutual suppression between two behaviors for temporal refinement.

        Writes a symmetric weight into ``__all__.suppression_matrix`` of
        ``config/temporal_refinement_settings.json`` (the same file the Temporal
        Refinement tab reads), so the next dense-inhibition run will down-weight
        each behavior by the other's probability.  Returns the weight applied.
        """
        path = self._root() / "config" / "temporal_refinement_settings.json"
        raw = read_json(path, {})
        all_settings = dict(raw.get("__all__", {}) or {})
        matrix = {k: dict(v) for k, v in (all_settings.get("suppression_matrix", {}) or {}).items()}
        if weight is None:
            weight = float(all_settings.get("inhibition_weight", 0.20) or 0.20)
        weight = float(max(0.0, min(0.5, weight)))
        for suppressed, suppressor in ((behavior_a, behavior_b), (behavior_b, behavior_a)):
            row = dict(matrix.get(suppressed, {}))
            row[suppressor] = weight
            matrix[suppressed] = row
        all_settings["suppression_matrix"] = matrix
        raw["__all__"] = all_settings
        write_json(path, raw)
        logger.info("Applied mutual inhibition %.2f between %s and %s.", weight, behavior_a, behavior_b)
        return weight

    # ==================================================================
    # Trace / bout access
    # ==================================================================
    def _read_latest(self, dir_token: str) -> dict[str, Any]:
        path = self._root() / "derived" / "temporal_refinement" / dir_token / "latest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _latest_json(self, bid: str) -> dict[str, Any]:
        """Return the temporal run record for a behavior.

        Prefers a behavior-specific run directory, then falls back to the
        shared ``target_behavior`` competition run (whose trace carries a
        ``prob_<behavior_id>`` column for every behavior).
        """
        latest = self._read_latest(_safe_name(bid))
        if latest:
            return latest
        return self._read_latest("target_behavior")

    def _threshold_for(self, bid: str) -> tuple[float, int, int]:
        """Resolve (onset_threshold, min_bout_frames, merge_gap_frames) for a behavior."""
        onset, min_bout, merge_gap = _DEFAULT_THRESHOLD, _DEFAULT_MIN_BOUT, _DEFAULT_MERGE_GAP
        # 1. Settings file written by the Temporal Review tab (per-behavior preferred).
        raw = read_json(self._root() / "config" / "temporal_review_settings.json", {})
        by_behavior = (raw.get("by_behavior") or {}).get(bid) or {}
        defaults = raw.get("__all__") or {}
        merged = {**defaults, **by_behavior}
        # 2. A behavior-specific postprocess manifest records the values actually used.
        #    (Skip the shared target_behavior run so we don't override per-behavior thresholds.)
        own_latest = self._read_latest(_safe_name(bid))
        post_dir = str(own_latest.get("postprocess_dir", "") or "").strip()
        if post_dir:
            man = read_json(Path(post_dir) / "postprocess_manifest.json", {})
            merged = {**merged, **(man.get("postprocess") or {})}
        onset = float(merged.get("onset_threshold", onset))
        min_bout = int(merged.get("min_bout_duration_frames", min_bout))
        merge_gap = int(merged.get("merge_gap_frames", merge_gap))
        return onset, min_bout, merge_gap

    def _load_traces(self, bid: str) -> dict[str, pd.DataFrame]:
        """Return {session_id: trace_df} for a behavior, each with frame + probability columns.

        Traces are cached per inference directory: in competition mode every
        behavior shares one trace file (with a ``prob_<id>`` column each), so this
        avoids re-reading the same parquet for each behavior.
        """
        latest = self._latest_json(bid)
        inference_dir = str(latest.get("inference_dir", "") or "").strip()
        if not inference_dir:
            return {}
        if inference_dir in self._traces_cache:
            return self._traces_cache[inference_dir]
        man = read_json(Path(inference_dir) / "inference_manifest.json", {})
        trace_paths = man.get("trace_paths") or {}
        out: dict[str, pd.DataFrame] = {}
        for sid, p in trace_paths.items():
            path = Path(str(p))
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "frame" not in df.columns:
                continue
            out[str(sid)] = df.sort_values("frame").reset_index(drop=True)
        self._traces_cache[inference_dir] = out
        return out

    @staticmethod
    def _prob_column(df: pd.DataFrame, bid: str) -> str | None:
        for col in (f"prob_{bid}", f"prob_{_safe_name(bid)}", "probability"):
            if col in df.columns:
                return col
        return None

    def _detect_bouts(self, bid: str, df: pd.DataFrame) -> list[tuple[int, int, float]]:
        """Return [(start_frame, end_frame, mean_prob)] positive bouts for one session."""
        col = self._prob_column(df, bid)
        if col is None:
            return []
        onset, min_bout, merge_gap = self._threshold_for(bid)
        probs = df[col].to_numpy(dtype=float)
        frames = df["frame"].to_numpy(dtype=int)
        binary = threshold_probabilities(probs, onset)
        binary = merge_close_bouts(binary, merge_gap)
        binary = remove_short_bouts(binary, min_bout)
        out: list[tuple[int, int, float]] = []
        for i, j in binary_trace_to_intervals(binary):
            mean_p = float(np.nanmean(probs[i : j + 1])) if j >= i else 0.0
            out.append((int(frames[i]), int(frames[j]), round(mean_p, 4)))
        return out

    def _detect_negative_runs(self, bid: str, df: pd.DataFrame) -> list[tuple[int, int, float]]:
        """Return clearly-negative runs (mean prob well below threshold)."""
        col = self._prob_column(df, bid)
        if col is None:
            return []
        onset, min_bout, _ = self._threshold_for(bid)
        lo = max(0.0, onset - 0.2)
        probs = df[col].to_numpy(dtype=float)
        frames = df["frame"].to_numpy(dtype=int)
        binary = (probs < lo).astype(np.uint8)
        binary = remove_short_bouts(binary, min_bout)
        out: list[tuple[int, int, float]] = []
        for i, j in binary_trace_to_intervals(binary):
            mean_p = float(np.nanmean(probs[i : j + 1])) if j >= i else 0.0
            out.append((int(frames[i]), int(frames[j]), round(mean_p, 4)))
        return out

    def _session_fps(self, sid: str) -> float:
        manifest = self._imports.load_manifest(self._root())
        if manifest is None:
            return 30.0
        # Video assets often carry session_id=None; resolve via the linked session.
        video_by_id = {v.asset_id: v for v in manifest.videos}
        for session in manifest.linked_sessions:
            if str(session.session_id) == str(sid):
                video = video_by_id.get(session.video_asset_id)
                if video and video.fps:
                    return float(video.fps)
                break
        # Fallback: a direct match if any video does carry the session id.
        for v in manifest.videos:
            if str(v.session_id) == str(sid) and v.fps:
                return float(v.fps)
        return 30.0

    def _session_subject_map(self) -> dict[str, str]:
        """Map ``session_id -> subject_id`` from the import manifest (best effort).

        Sessions without a resolvable subject are omitted; callers fall back to
        treating each such session as its own subject.
        """
        manifest = self._imports.load_manifest(self._root())
        if manifest is None:
            return {}
        out: dict[str, str] = {}
        for session in manifest.linked_sessions:
            sid = str(session.session_id or "").strip()
            sub = str(getattr(session, "subject_id", "") or "").strip()
            if sid and sub:
                out[sid] = sub
        return out

    # ==================================================================
    # Quiz assembly
    # ==================================================================
    def assemble_run(
        self,
        settings: ValidationSettings,
        progress_callback: Any = None,
    ) -> ValidationRun:
        """Sample clips across categories, extract videos, and persist a new run."""
        active = self._active_behaviors()
        if not active:
            raise RuntimeError("No active behaviors to validate. Define behaviors first.")

        weights = {
            "prior_accepted": max(0.0, settings.prop_prior_accepted),
            "unreviewed_positive": max(0.0, settings.prop_unreviewed_positive),
            "negative": max(0.0, settings.prop_negative),
            "fringe": max(0.0, settings.prop_fringe),
        }
        total_w = sum(weights.values()) or 1.0
        n_total = max(1, int(settings.n_total_clips))
        quotas = {k: int(round(n_total * w / total_w)) for k, w in weights.items()}

        records: list[ValidationClipRecord] = []
        records += self._sample_prior_accepted(quotas["prior_accepted"], active)
        records += self._sample_positive(
            quotas["unreviewed_positive"], active, settings, fringe=False
        )
        records += self._sample_negative(quotas["negative"], active, settings)
        records += self._sample_positive(quotas["fringe"], active, settings, fringe=True)

        # Deduplicate by clip_id (first category wins).
        seen: set[str] = set()
        deduped: list[ValidationClipRecord] = []
        for rec in records:
            if rec.clip_id in seen:
                continue
            seen.add(rec.clip_id)
            deduped.append(rec)
        records = deduped

        # Tag clips where the model flags two or more behaviors simultaneously.
        self._assign_coactive_labels(records)

        # Extract any missing clip videos.
        self._extract_clips(records, settings, progress_callback)

        # Keep only clips that have a playable video file.
        records = [r for r in records if r.clip_path and Path(r.clip_path).exists()]

        import random  # noqa: PLC0415

        random.shuffle(records)

        run = ValidationRun(
            run_id=uuid.uuid4().hex[:12],
            created_at=datetime.utcnow(),
            config=settings.model_dump(mode="json"),
            clips=records,
        )
        self._save_run(run)
        write_json(self._active_run_path(), {"run_id": run.run_id})
        self._prune_orphan_clips()
        logger.info("Validation run assembled: %s (%d clips)", run.run_id, len(records))
        return run

    def _prune_orphan_clips(self) -> None:
        """Delete cached validation clips not referenced by any saved run.

        Keeps the validation clip cache tied to live tests and clears stale clips
        left by deleted (or pre-update) runs.  Only files under
        ``derived/validation/clips`` are touched.
        """
        clips_root = self._clips_dir()
        if not clips_root.exists():
            return
        referenced: set[str] = set()
        for run in self.list_runs():
            for clip in run.clips:
                if clip.clip_path:
                    try:
                        referenced.add(str(Path(clip.clip_path).resolve()))
                    except Exception:
                        referenced.add(str(clip.clip_path))
        for path in clips_root.rglob("*.mp4"):
            try:
                if str(path.resolve()) not in referenced:
                    path.unlink(missing_ok=True)
            except Exception:
                pass

    def _balanced_quota(self, total: int, n_groups: int) -> list[int]:
        if n_groups <= 0 or total <= 0:
            return [0] * max(0, n_groups)
        base, extra = divmod(total, n_groups)
        return [base + (1 if i < extra else 0) for i in range(n_groups)]

    def _sample_prior_accepted(
        self, quota: int, active: list[tuple[str, str]]
    ) -> list[ValidationClipRecord]:
        if quota <= 0:
            return []
        cand_by_id = {str(c.window_id): c for c in self._candidates.load_candidates()}
        accepted: dict[str, list] = defaultdict(list)
        for d in self._reviews.load_decisions():
            if d.decision != ReviewDecisionType.ACCEPT:
                continue
            label = str(d.behavior_label or "").strip()
            if not label or _is_no_behavior(label):
                continue
            cand = cand_by_id.get(d.clip_id)
            if cand is None:
                continue
            accepted[label].append((d, cand))

        import random  # noqa: PLC0415

        active_ids = [bid for bid, _ in active]
        per = self._balanced_quota(quota, len(active_ids))
        out: list[ValidationClipRecord] = []
        for bid, want in zip(active_ids, per):
            pool = accepted.get(bid, [])
            random.shuffle(pool)
            for d, cand in pool[:want]:
                out.append(ValidationClipRecord(
                    clip_id=str(d.clip_id),
                    category="prior_accepted",
                    behavior_id=bid,
                    machine_label=bid,
                    reference_label=bid,
                    session_id=str(cand.session_id),
                    start_frame=int(cand.start_frame),
                    end_frame=int(cand.end_frame),
                    probability=float(getattr(cand, "total_score", 0.0) or 0.0),
                ))
        return out

    def _band_of(self, prob: float, edges: list[float]) -> int:
        band = 0
        for i, e in enumerate(sorted(edges)):
            if prob >= e:
                band = i
        return band

    def _sample_positive(
        self,
        quota: int,
        active: list[tuple[str, str]],
        settings: ValidationSettings,
        fringe: bool,
    ) -> list[ValidationClipRecord]:
        if quota <= 0:
            return []
        import random  # noqa: PLC0415

        active_ids = [bid for bid, _ in active]
        per = self._balanced_quota(quota, len(active_ids))
        out: list[ValidationClipRecord] = []
        for bid, want in zip(active_ids, per):
            if want <= 0:
                continue
            onset, _, _ = self._threshold_for(bid)
            # Build candidate center-windows from every bout in every session.
            windows: list[tuple[str, int, int, float]] = []  # (session, s, e, prob)
            for sid, df in self._load_traces(bid).items():
                fps = self._session_fps(sid)
                w = max(2, int(round(settings.clip_seconds * fps)))
                for bs, be, mp in self._detect_bouts(bid, df):
                    near_thr = abs(mp - onset) <= settings.fringe_half_width
                    if fringe and not near_thr:
                        continue
                    if not fringe and near_thr:
                        # Reserve near-threshold bouts for the fringe category.
                        continue
                    mid = (bs + be) // 2
                    ws = max(0, mid - w // 2)
                    we = ws + w - 1
                    windows.append((sid, ws, we, mp))
            if not windows:
                continue
            chosen = self._stratified_pick(windows, want, settings.prob_band_edges)
            for sid, ws, we, mp in chosen:
                category = "fringe" if fringe else "unreviewed_positive"
                cid = f"val_{category}_{_safe_name(bid)}_{sid}_{ws}_{we}"
                out.append(ValidationClipRecord(
                    clip_id=cid,
                    category=category,  # type: ignore[arg-type]
                    behavior_id=bid,
                    machine_label=bid,
                    reference_label=None if fringe else bid,
                    session_id=sid,
                    start_frame=ws,
                    end_frame=we,
                    probability=mp,
                    is_fringe=fringe,
                ))
        return out

    def _stratified_pick(
        self,
        windows: list[tuple[str, int, int, float]],
        want: int,
        edges: list[float],
    ) -> list[tuple[str, int, int, float]]:
        """Pick ``want`` windows spread across probability bands."""
        import random  # noqa: PLC0415

        by_band: dict[int, list] = defaultdict(list)
        for win in windows:
            by_band[self._band_of(win[3], edges)].append(win)
        for lst in by_band.values():
            random.shuffle(lst)
        bands = sorted(by_band.keys())
        picked: list[tuple[str, int, int, float]] = []
        # Round-robin across bands so each probability range is represented.
        while len(picked) < want and any(by_band[b] for b in bands):
            for b in bands:
                if by_band[b]:
                    picked.append(by_band[b].pop())
                    if len(picked) >= want:
                        break
        return picked

    def _sample_negative(
        self,
        quota: int,
        active: list[tuple[str, str]],
        settings: ValidationSettings,
    ) -> list[ValidationClipRecord]:
        if quota <= 0:
            return []
        import random  # noqa: PLC0415

        active_ids = [bid for bid, _ in active]
        per = self._balanced_quota(quota, len(active_ids))
        out: list[ValidationClipRecord] = []
        for bid, want in zip(active_ids, per):
            if want <= 0:
                continue
            windows: list[tuple[str, int, int, float]] = []
            for sid, df in self._load_traces(bid).items():
                fps = self._session_fps(sid)
                w = max(2, int(round(settings.clip_seconds * fps)))
                for bs, be, mp in self._detect_negative_runs(bid, df):
                    mid = (bs + be) // 2
                    ws = max(0, mid - w // 2)
                    we = ws + w - 1
                    windows.append((sid, ws, we, mp))
            random.shuffle(windows)
            for sid, ws, we, mp in windows[:want]:
                cid = f"val_negative_{_safe_name(bid)}_{sid}_{ws}_{we}"
                out.append(ValidationClipRecord(
                    clip_id=cid,
                    category="negative",
                    behavior_id=bid,
                    machine_label=NO_BEHAVIOR_ID,
                    reference_label=NO_BEHAVIOR_ID,
                    session_id=sid,
                    start_frame=ws,
                    end_frame=we,
                    probability=mp,
                ))
        return out

    # ==================================================================
    # Clip extraction
    # ==================================================================
    def _extract_clips(
        self,
        records: list[ValidationClipRecord],
        settings: ValidationSettings,
        progress_callback: Any = None,
    ) -> None:
        """Populate ``clip_path`` for each record, reusing existing files or decoding."""
        manifest = self._imports.load_manifest(self._root())
        presets = self._clips.load_project_presets()
        preset = presets[0] if presets else None

        # Group records needing extraction by session.
        by_session: dict[str, list[ValidationClipRecord]] = defaultdict(list)
        done = 0
        total = len(records)
        for rec in records:
            existing = self._find_existing_clip(rec.clip_id, rec.session_id)
            if existing is not None:
                rec.clip_path = str(existing)
                done += 1
                if progress_callback:
                    progress_callback(done, total)
            else:
                by_session[rec.session_id].append(rec)

        if not by_session or manifest is None or preset is None:
            return
        if not ClipExtractionService.can_decode_video():
            return

        out_root = self._clips_dir()
        for sid, recs in by_session.items():
            video_path = self._imports.video_path_for_session(manifest, sid)
            if not video_path or not video_path.exists():
                logger.warning("Validation: missing video for session %s; skipping %d clips.", sid, len(recs))
                continue
            windows = [
                CandidateWindow(
                    window_id=r.clip_id,
                    session_id=sid,
                    start_frame=r.start_frame,
                    end_frame=r.end_frame,
                    behavior_id=r.behavior_id,
                )
                for r in recs
            ]
            # Center the crop on the subject using pose centroids (same as Clip Review).
            pose_cx = pose_cy = None
            pose_path = self._imports.pose_path_for_session(manifest, sid)
            if pose_path and pose_path.exists():
                try:
                    pose = self._pose.load_and_clean(
                        pose_path, getattr(manifest, "smoothing_settings", None)
                    )
                    pose_cx = pose.centroid_x
                    pose_cy = pose.centroid_y
                except Exception:
                    logger.warning("Validation: could not load pose centroids for %s; using center crop.", sid)
            cfg = ClipExtractionConfig(
                video_path=video_path,
                session_id=sid,
                preset=preset,
                output_dir=out_root / sid,
                pose_centroid_x=pose_cx,
                pose_centroid_y=pose_cy,
                pixels_per_mm=self._imports.pixels_per_mm_for_session(manifest, sid),
            )
            result = self._clips.extract_selected_clips(windows, cfg)
            path_by_id = {c.clip_id: c.processed_clip_path for c in result.clips if c.processed_clip_path}
            for r in recs:
                r.clip_path = path_by_id.get(r.clip_id)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

    def _find_existing_clip(self, clip_id: str, session_id: str) -> Path | None:
        """Reuse an already-extracted, subject-centered clip for this id if one exists.

        Only the Clip Review output (``derived/clips``) is trusted for reuse — those
        clips are always cropped around the subject via pose centroids.  Clips in the
        validation cache are deliberately NOT reused: re-sampled windows are always
        re-extracted (overwriting any stale file) so freshly generated tests are
        guaranteed to be subject-centered.
        """
        stem = ClipExtractionService.clip_filename_for_id(clip_id)
        candidate = self._root() / "derived" / "clips" / session_id / f"{stem}.mp4"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
        return None

    # ==================================================================
    # Behavior grid (5×5 montage of strong positive bouts)
    # ==================================================================
    def behavior_grid_behaviors(self) -> list[tuple[str, str]]:
        """(behavior_id, name) pairs eligible for the Behavior Grid dropdown."""
        return self._active_behaviors()

    def select_grid_bouts(
        self,
        behavior_id: str,
        n_cells: int = 25,
        top_fraction: float = 0.4,
    ) -> list[GridCellSpec]:
        """Pick up to ``n_cells`` strong positive bouts spread across subjects.

        Bouts are detected per session, filtered to the top ``top_fraction`` by
        mean probability (the confident detections), then chosen round-robin so
        distinct *subjects* are preferred — every subject contributes one bout
        before any subject contributes a second.  Once unique subjects are
        exhausted, additional bouts are drawn from subjects that have more than
        one bout, skipping any that frame-overlap an already-chosen bout from the
        same session so no bout is duplicated.  Sessions whose subject is unknown
        (no manifest mapping) are each treated as their own subject.
        """
        import random  # noqa: PLC0415

        bid = str(behavior_id or "").strip()
        if not bid:
            return []
        by_session: dict[str, list[tuple[int, int, float]]] = {}
        all_probs: list[float] = []
        for sid, df in self._load_traces(bid).items():
            bouts = self._detect_bouts(bid, df)
            if bouts:
                by_session[sid] = bouts
                all_probs.extend(mp for _s, _e, mp in bouts)
        if not all_probs:
            return []

        frac = min(1.0, max(0.0, float(top_fraction)))
        cutoff = float(np.quantile(all_probs, 1.0 - frac)) if frac < 1.0 else -np.inf

        # Keep only confident bouts; shuffle within each session.
        pools: dict[str, list[tuple[int, int, float]]] = {}
        for sid, bouts in by_session.items():
            kept = [b for b in bouts if b[2] >= cutoff]
            if kept:
                random.shuffle(kept)
                pools[sid] = kept
        if not pools:
            return []

        # Group sessions by subject so the round-robin prefers unique subjects.
        # Sessions with no known subject fall back to being their own subject.
        subject_of = self._session_subject_map()
        subjects: dict[str, list[str]] = defaultdict(list)
        for sid in pools:
            subjects[subject_of.get(sid, sid)].append(sid)

        chosen: list[GridCellSpec] = []
        used: dict[str, list[tuple[int, int]]] = defaultdict(list)
        subject_keys = list(subjects.keys())
        # Round-robin passes across subjects until full or every pool is drained.
        # Pass 1 gives one bout per subject; later passes reuse subjects that have
        # additional (non-overlapping) bouts.
        while len(chosen) < n_cells and any(pools.values()):
            random.shuffle(subject_keys)
            for subj in subject_keys:
                if len(chosen) >= n_cells:
                    break
                sids = [s for s in subjects[subj] if pools.get(s)]
                random.shuffle(sids)
                for sid in sids:
                    pool = pools[sid]
                    picked = False
                    while pool:
                        s, e, mp = pool.pop()
                        if any(not (e < us or s > ue) for us, ue in used[sid]):
                            continue  # overlaps an already-chosen bout from this session
                        used[sid].append((s, e))
                        chosen.append(GridCellSpec(sid, int(s), int(e), float(mp)))
                        picked = True
                        break
                    if picked:
                        break  # one bout per subject per pass
        return chosen[:n_cells]

    def render_behavior_grid(
        self,
        behavior_id: str,
        pre_seconds: float,
        post_seconds: float,
        grid_px: int,
        show_keypoints: bool,
        out_path: Path,
        progress_callback: Any = None,
        crop_scale: float = 1.0,
    ) -> Path:
        """Render a 5×5 looping montage of strong positive bouts to *out_path*.

        Each cell is a subject-centred crop of one bout (padded by ``pre_seconds``
        / ``post_seconds``) with optional pose-keypoint overlay.  ``crop_scale`` is
        a linear multiplier on each cell's crop half-width (>1 zooms out to show
        more surroundings, <1 tightens onto the subject).  Raises ``RuntimeError``
        when there is nothing to render or video decoding is unavailable.
        """
        from abel.services import behavior_grid_render as render  # noqa: PLC0415

        if not ClipExtractionService.can_decode_video():
            raise RuntimeError("OpenCV video decoding is not available in this environment.")

        specs = self.select_grid_bouts(behavior_id)
        if not specs:
            raise RuntimeError(
                "No positive bouts found for this behavior. Run temporal inference first, "
                "or pick a behavior with detected bouts."
            )

        manifest = self._imports.load_manifest(self._root())
        if manifest is None:
            raise RuntimeError("Project import manifest is missing — re-open the project.")
        presets = self._clips.load_project_presets()
        preset = presets[0] if presets else None
        if preset is None:
            raise RuntimeError("No clip preset configured for this project.")

        rows = cols = 5
        n_cells = rows * cols
        cell_px = max(32, int(grid_px) // cols)
        smoothing = getattr(manifest, "smoothing_settings", None)
        cells_dir = out_path.parent / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        # Pose arrays are reused across bouts from the same session.
        pose_cache: dict[str, tuple] = {}

        def _pose_for(sid: str) -> tuple:
            if sid in pose_cache:
                return pose_cache[sid]
            result: tuple = (None, None, None, None, None)
            pose_path = self._imports.pose_path_for_session(manifest, sid)
            if pose_path and pose_path.exists():
                try:
                    pose = self._pose.load_and_clean(pose_path, smoothing)
                    result = (
                        pose.x.to_numpy(dtype=float),
                        pose.y.to_numpy(dtype=float),
                        pose.likelihood.to_numpy(dtype=float),
                        np.asarray(pose.centroid_x, dtype=float),
                        np.asarray(pose.centroid_y, dtype=float),
                    )
                except Exception:
                    logger.warning("Behavior grid: could not load pose for %s.", sid)
            pose_cache[sid] = result
            return result

        cell_paths: list[Path | None] = []
        total = len(specs)
        for i, spec in enumerate(specs):
            cell_paths_entry: Path | None = None
            video_path = self._imports.video_path_for_session(manifest, spec.session_id)
            if video_path and video_path.exists():
                fps = self._session_fps(spec.session_id)
                pre = int(round(max(0.0, pre_seconds) * fps))
                post = int(round(max(0.0, post_seconds) * fps))
                start = max(0, spec.bout_start - pre)
                end = spec.bout_end + post
                px, py, pc, cx, cy = _pose_for(spec.session_id)
                cell_out = cells_dir / f"cell_{i:02d}.mp4"
                ok = render.render_cell(
                    video_path,
                    px, py, pc, cx, cy,
                    start, end,
                    crop_margin_px=preset.crop_margin_px,
                    crop_area_scale=float(getattr(preset, "crop_area_scale", 1.25) or 1.25),
                    cell_px=cell_px,
                    show_keypoints=show_keypoints,
                    out_path=cell_out,
                    crop_scale=float(crop_scale),
                )
                if ok:
                    cell_paths_entry = cell_out
            else:
                logger.warning("Behavior grid: missing video for session %s.", spec.session_id)
            cell_paths.append(cell_paths_entry)
            if progress_callback:
                progress_callback(i + 1, total)

        # Pad to a full grid so trailing cells render black.
        cell_paths += [None] * (n_cells - len(cell_paths))
        render.stitch_grid(cell_paths, cell_px * cols, out_path, rows=rows, cols=cols)

        # Clean temp cell clips.
        for p in cells_dir.glob("cell_*.mp4"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            cells_dir.rmdir()
        except Exception:
            pass

        logger.info(
            "Behavior grid rendered: %s (%d cells, %dpx).", out_path, len(specs), cell_px * cols
        )
        return out_path

    def behavior_grid_preview_path(self) -> Path:
        """Stable path the Behavior Grid preview video is written to."""
        return self._validation_dir() / "behavior_grid" / "preview.mp4"

    # ==================================================================
    # Run / answer persistence
    # ==================================================================
    def _save_run(self, run: ValidationRun) -> None:
        write_json(self._runs_dir() / f"{run.run_id}.json", run.model_dump(mode="json"))

    def load_run(self, run_id: str) -> ValidationRun | None:
        raw = read_json(self._runs_dir() / f"{run_id}.json", {})
        if not raw:
            return None
        try:
            return ValidationRun.model_validate(raw)
        except Exception:
            return None

    def load_active_run(self) -> ValidationRun | None:
        raw = read_json(self._active_run_path(), {})
        run_id = str(raw.get("run_id", "") or "").strip()
        if not run_id:
            return None
        return self.load_run(run_id)

    def list_runs(self) -> list[ValidationRun]:
        runs: list[ValidationRun] = []
        rdir = self._runs_dir()
        if not rdir.exists():
            return runs
        for p in sorted(rdir.glob("*.json")):
            try:
                runs.append(ValidationRun.model_validate(read_json(p, {})))
            except Exception:
                continue
        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete a saved test and all its reviewer answers.

        Clears the active-run pointer if it referenced the deleted run.  Extracted
        clip videos are left on disk (they may be shared/reused by other tests).
        Returns True if a run file was removed.
        """
        if not run_id:
            return False
        run_path = self._runs_dir() / f"{run_id}.json"
        removed = run_path.exists()
        try:
            run_path.unlink(missing_ok=True)
        except Exception:
            removed = False
        adir = self._answers_dir()
        if adir.exists():
            for p in adir.glob(f"{run_id}__*.json"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        active = read_json(self._active_run_path(), {})
        if str(active.get("run_id", "") or "").strip() == run_id:
            write_json(self._active_run_path(), {"run_id": ""})
        logger.info("Deleted validation run %s (removed=%s).", run_id, removed)
        return removed

    def export_results_xlsx(self, out_path: Path, run_ids: list[str] | None = None) -> int:
        """Write validation results to an Excel workbook, one sheet per test.

        Each test (run) gets a sheet with its per-reviewer / per-behavior metrics,
        intra-rater consistency, and overlap-excluded counts.  A leading ``Index``
        sheet summarises every test.  Returns the number of test sheets written.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        runs = self.list_runs()
        if run_ids is not None:
            wanted = set(run_ids)
            runs = [r for r in runs if r.run_id in wanted]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        if not runs:
            raise RuntimeError("No validation tests to export.")

        index_rows: list[dict[str, Any]] = []
        sheets: list[tuple[str, pd.DataFrame]] = []
        used_names: set[str] = set()

        def _pct(v: Any) -> Any:
            return round(float(v), 4) if isinstance(v, (int, float)) else None

        for run in runs:
            answers = self.load_all_answers(run.run_id)
            metrics = self.compute_metrics(run, answers)
            inter = metrics.get("inter_rater", {})
            overlap = metrics.get("overlap", {})
            stamp = run.created_at.strftime("%Y-%m-%d %H:%M")

            rows: list[dict[str, Any]] = []
            for rid, rdata in metrics["per_reviewer"].items():
                intra = (metrics.get("intra_rater") or {}).get(rid, {})
                base = {
                    "reviewer": rid,
                    "answered": rdata.get("n_answered", 0),
                    "agreement_with_model": _pct(rdata.get("agreement")),
                    "unsure_rate": _pct(rdata.get("unsure_rate")),
                    "overlap_excluded": rdata.get("n_overlap", 0),
                    "intra_self_agreement": _pct(intra.get("agreement")),
                    "intra_kappa": _pct(intra.get("kappa")),
                }
                per_behavior = rdata.get("per_behavior") or {}
                if per_behavior:
                    for b, pb in per_behavior.items():
                        rows.append({
                            **base,
                            "behavior": self._behavior_name(b),
                            "precision": _pct(pb.get("precision")),
                            "recall": _pct(pb.get("recall")),
                            "f1": _pct(pb.get("f1")),
                            "tp": pb.get("tp", 0),
                            "fp": pb.get("fp", 0),
                            "fn": pb.get("fn", 0),
                        })
                else:
                    rows.append({**base, "behavior": "(no answers)"})
            if not rows:
                rows.append({"reviewer": "(no reviewers yet)"})

            sheet_name = self._unique_sheet_name(stamp.replace(":", "."), used_names)
            sheets.append((sheet_name, pd.DataFrame(rows)))
            index_rows.append({
                "test_date": stamp,
                "sheet": sheet_name,
                "run_id": run.run_id,
                "clips": len(run.clips),
                "reviewers": len(answers),
                "inter_rater_kappa": _pct(inter.get("kappa")),
                "inter_rater_agreement": _pct(inter.get("agreement")),
                "overlap_rate": _pct(overlap.get("overall_overlap_rate")),
            })

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(index_rows).to_excel(writer, sheet_name="Index", index=False)
            try:
                pd.DataFrame(self.model_overview()).to_excel(
                    writer, sheet_name="Model Summary", index=False
                )
            except Exception:
                logger.exception("Validation export: model summary sheet failed")
            for sheet_name, df in sheets:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info("Exported %d validation test(s) to %s", len(sheets), out_path)
        return len(sheets)

    @staticmethod
    def _unique_sheet_name(base: str, used: set[str]) -> str:
        """Return an Excel-safe (<=31 char, unique) sheet name."""
        for ch in r"[]:*?/\\":
            base = base.replace(ch, "-")
        base = base.strip() or "Test"
        name = base[:31]
        i = 2
        while name in used:
            suffix = f" ({i})"
            name = base[: 31 - len(suffix)] + suffix
            i += 1
        used.add(name)
        return name

    def _answers_path(self, run_id: str, reviewer_id: str) -> Path:
        safe = _safe_name(reviewer_id) or "reviewer"
        return self._answers_dir() / f"{run_id}__{safe}.json"

    def save_answer(self, run_id: str, answer: ValidationAnswerRecord) -> None:
        path = self._answers_path(run_id, answer.reviewer_id)
        raw = read_json(path, {"reviewer_id": answer.reviewer_id, "answers": {}})
        answers = dict(raw.get("answers", {}))
        answers[answer.clip_id] = answer.model_dump(mode="json")
        write_json(path, {"reviewer_id": answer.reviewer_id, "answers": answers})

    def load_answers(self, run_id: str, reviewer_id: str) -> dict[str, ValidationAnswerRecord]:
        raw = read_json(self._answers_path(run_id, reviewer_id), {})
        out: dict[str, ValidationAnswerRecord] = {}
        for cid, item in (raw.get("answers") or {}).items():
            try:
                out[str(cid)] = ValidationAnswerRecord.model_validate(item)
            except Exception:
                continue
        return out

    def list_reviewers(self, run_id: str) -> list[str]:
        out: list[str] = []
        adir = self._answers_dir()
        if not adir.exists():
            return out
        for p in adir.glob(f"{run_id}__*.json"):
            raw = read_json(p, {})
            rid = str(raw.get("reviewer_id", "") or "").strip()
            if rid:
                out.append(rid)
        return sorted(set(out))

    def load_all_answers(self, run_id: str) -> dict[str, dict[str, ValidationAnswerRecord]]:
        return {rid: self.load_answers(run_id, rid) for rid in self.list_reviewers(run_id)}

    # ==================================================================
    # Metrics
    # ==================================================================
    def compute_metrics(
        self,
        run: ValidationRun,
        answers_by_reviewer: dict[str, dict[str, ValidationAnswerRecord]],
    ) -> dict[str, Any]:
        """Compute user-vs-machine and inter-rater metrics for a run."""
        # Tag clips where the model flagged two or more behaviors at once so they
        # can be excluded from scoring (recomputed here so older runs benefit too).
        self._assign_coactive_labels(run.clips)
        clip_by_id = {c.clip_id: c for c in run.clips}
        behavior_ids = sorted({c.behavior_id for c in run.clips if not _is_no_behavior(c.behavior_id)})

        per_reviewer: dict[str, Any] = {}
        for rid, answers in answers_by_reviewer.items():
            per_reviewer[rid] = self._reviewer_vs_machine(clip_by_id, answers, behavior_ids)

        inter_rater = self._inter_rater(clip_by_id, answers_by_reviewer)
        intra_rater = self._intra_rater(clip_by_id, answers_by_reviewer)
        confusion = self._confusion_analysis(clip_by_id, answers_by_reviewer, behavior_ids)
        overlap = self.overlap_analysis()

        return {
            "run_id": run.run_id,
            "n_clips": len(run.clips),
            "behaviors": behavior_ids,
            "per_reviewer": per_reviewer,
            "inter_rater": inter_rater,
            "intra_rater": intra_rater,
            "confusion": confusion,
            "overlap": overlap,
        }

    def _intra_rater(
        self,
        clip_by_id: dict[str, ValidationClipRecord],
        answers_by_reviewer: dict[str, dict[str, ValidationAnswerRecord]],
    ) -> dict[str, Any]:
        """Per-reviewer test-retest reliability vs their own prior-accepted labels.

        For the ``prior_accepted`` clips (which carry the behavior the reviewer
        previously accepted as ``reference_label``), compare this reviewer's quiz
        answer to that original label.  This measures how consistently a reviewer
        agrees with their own past judgments (intra-rater reliability).
        """
        out: dict[str, Any] = {}
        for rid, answers in answers_by_reviewer.items():
            refs: list[str] = []
            users: list[str] = []
            n_unsure = 0
            per_behavior: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # ref -> [agree, total]
            for cid, ans in answers.items():
                clip = clip_by_id.get(cid)
                if clip is None or clip.category != "prior_accepted" or not clip.reference_label:
                    continue
                if ans.is_unsure:
                    n_unsure += 1
                    continue
                ref = clip.reference_label
                refs.append(ref)
                users.append(ans.label)
                pb = per_behavior[ref]
                pb[1] += 1
                if ans.label == ref:
                    pb[0] += 1
            n = len(refs)
            agreement = (sum(1 for r, u in zip(refs, users) if r == u) / n) if n else None
            kappa = None
            if n >= 2 and len(set(refs) | set(users)) >= 2:
                try:
                    from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

                    kappa = float(cohen_kappa_score(refs, users))
                except Exception:
                    kappa = None
            out[rid] = {
                "n": n,
                "n_unsure": n_unsure,
                "agreement": agreement,
                "kappa": kappa,
                "per_behavior": {
                    b: (a / t if t else None) for b, (a, t) in per_behavior.items()
                },
                "per_behavior_counts": {b: t for b, (a, t) in per_behavior.items()},
            }
        return out

    def _confusion_analysis(
        self,
        clip_by_id: dict[str, ValidationClipRecord],
        answers_by_reviewer: dict[str, dict[str, ValidationAnswerRecord]],
        behavior_ids: list[str],
    ) -> dict[str, Any]:
        """Aggregate where model and reviewers disagree, and why.

        Builds a confusion matrix (rows = what the model asserted, columns = what
        reviewers said) plus a ranked list of the most common confusions, each
        split into borderline (fringe) vs clear-cut disagreements so the user can
        tell threshold problems from genuine model errors.
        """
        labels = [*behavior_ids, NO_BEHAVIOR_ID]
        matrix: dict[str, dict[str, int]] = {m: {u: 0 for u in labels} for m in labels}
        # (machine, user) -> [total, fringe]
        pair_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        n_fringe_disagree = 0
        n_clear_disagree = 0
        n_total = 0

        for answers in answers_by_reviewer.values():
            for cid, ans in answers.items():
                clip = clip_by_id.get(cid)
                if clip is None or ans.is_unsure:
                    continue
                # Skip ambiguous (multi-behavior) clips — they are not scored.
                if len(set(clip.coactive_labels or [])) >= 2:
                    continue
                machine = clip.machine_label if clip.machine_label in matrix else NO_BEHAVIOR_ID
                user = ans.label if ans.label in matrix else NO_BEHAVIOR_ID
                matrix[machine][user] += 1
                n_total += 1
                if machine != user:
                    pair = pair_counts[(machine, user)]
                    pair[0] += 1
                    if clip.is_fringe:
                        pair[1] += 1
                        n_fringe_disagree += 1
                    else:
                        n_clear_disagree += 1

        top_confusions = [
            {"machine": m, "user": u, "count": c[0], "fringe_count": c[1]}
            for (m, u), c in sorted(pair_counts.items(), key=lambda kv: kv[1][0], reverse=True)
        ]
        return {
            "labels": labels,
            "matrix": matrix,
            "top_confusions": top_confusions,
            "n_disagreements": n_fringe_disagree + n_clear_disagree,
            "n_fringe_disagreements": n_fringe_disagree,
            "n_clear_disagreements": n_clear_disagree,
            "n_total": n_total,
        }

    def _reviewer_vs_machine(
        self,
        clip_by_id: dict[str, ValidationClipRecord],
        answers: dict[str, ValidationAnswerRecord],
        behavior_ids: list[str],
    ) -> dict[str, Any]:
        n_answered = 0
        n_unsure = 0
        n_agree = 0
        n_overlap = 0
        n_overlap_matched = 0
        # Per-behavior confusion against the human verdict (human = truth).
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)
        category_correct: dict[str, int] = defaultdict(int)
        category_total: dict[str, int] = defaultdict(int)

        for cid, ans in answers.items():
            clip = clip_by_id.get(cid)
            if clip is None:
                continue
            if ans.is_unsure:
                n_unsure += 1
                continue
            # Ambiguous clips (model flagged 2+ behaviors) do not count against the
            # reviewer — they are excluded from scoring and reported as feedback.
            coactive = set(clip.coactive_labels or [])
            if len(coactive) >= 2:
                n_overlap += 1
                if ans.label in coactive:
                    n_overlap_matched += 1
                continue
            n_answered += 1
            user = ans.label
            machine = clip.machine_label
            if user == machine:
                n_agree += 1
            for b in behavior_ids:
                if machine == b and user == b:
                    tp[b] += 1
                elif machine == b and user != b:
                    fp[b] += 1
                elif machine != b and user == b:
                    fn[b] += 1
            # Per-category accuracy vs the expected (reference or machine) label.
            expected = clip.reference_label or clip.machine_label
            if expected is not None:
                category_total[clip.category] += 1
                if user == expected:
                    category_correct[clip.category] += 1

        per_behavior: dict[str, Any] = {}
        for b in behavior_ids:
            p_den = tp[b] + fp[b]
            r_den = tp[b] + fn[b]
            precision = tp[b] / p_den if p_den else None
            recall = tp[b] / r_den if r_den else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall
                else None
            )
            per_behavior[b] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp[b],
                "fp": fp[b],
                "fn": fn[b],
            }

        category_accuracy = {
            c: (category_correct[c] / category_total[c] if category_total[c] else None)
            for c in category_total
        }
        return {
            "n_answered": n_answered,
            "n_unsure": n_unsure,
            "n_overlap": n_overlap,
            "n_overlap_matched": n_overlap_matched,
            "unsure_rate": (n_unsure / (n_answered + n_unsure)) if (n_answered + n_unsure) else None,
            "agreement": (n_agree / n_answered) if n_answered else None,
            "per_behavior": per_behavior,
            "category_accuracy": category_accuracy,
        }

    def _inter_rater(
        self,
        clip_by_id: dict[str, ValidationClipRecord],
        answers_by_reviewer: dict[str, dict[str, ValidationAnswerRecord]],
    ) -> dict[str, Any]:
        reviewers = sorted(answers_by_reviewer.keys())
        if len(reviewers) < 2:
            return {"n_reviewers": len(reviewers), "shared_clips": 0, "kappa": None, "agreement": None}

        # Clips answered (non-unsure) by every reviewer.
        per_clip_labels: dict[str, list[str]] = {}
        for cid in clip_by_id:
            labels: list[str] = []
            for rid in reviewers:
                ans = answers_by_reviewer[rid].get(cid)
                if ans is None or ans.is_unsure:
                    labels = []
                    break
                labels.append(ans.label)
            if labels:
                per_clip_labels[cid] = labels

        shared = len(per_clip_labels)
        if shared == 0:
            return {"n_reviewers": len(reviewers), "shared_clips": 0, "kappa": None, "agreement": None}

        # Overall percent agreement (all reviewers identical).
        full_agree = sum(1 for labels in per_clip_labels.values() if len(set(labels)) == 1)
        agreement = full_agree / shared

        kappa = self._kappa(reviewers, per_clip_labels)
        return {
            "n_reviewers": len(reviewers),
            "reviewers": reviewers,
            "shared_clips": shared,
            "agreement": agreement,
            "kappa": kappa,
        }

    @staticmethod
    def _kappa(reviewers: list[str], per_clip_labels: dict[str, list[str]]) -> float | None:
        """Cohen's kappa for 2 reviewers, Fleiss' kappa for >2."""
        try:
            if len(reviewers) == 2:
                from sklearn.metrics import cohen_kappa_score  # noqa: PLC0415

                a = [labels[0] for labels in per_clip_labels.values()]
                b = [labels[1] for labels in per_clip_labels.values()]
                return float(cohen_kappa_score(a, b))
            # Fleiss' kappa.
            categories = sorted({lab for labels in per_clip_labels.values() for lab in labels})
            cat_index = {c: i for i, c in enumerate(categories)}
            n_items = len(per_clip_labels)
            n_raters = len(reviewers)
            if n_items == 0 or n_raters < 2 or len(categories) < 2:
                return None
            counts = np.zeros((n_items, len(categories)), dtype=float)
            for row, labels in enumerate(per_clip_labels.values()):
                for lab in labels:
                    counts[row, cat_index[lab]] += 1
            p_j = counts.sum(axis=0) / (n_items * n_raters)
            P_i = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
            P_bar = P_i.mean()
            P_e = float(np.square(p_j).sum())
            if np.isclose(P_e, 1.0):
                return None
            return float((P_bar - P_e) / (1.0 - P_e))
        except Exception:
            return None

    # ==================================================================
    # Suggestions
    # ==================================================================
    def suggestions(self, metrics: dict[str, Any]) -> list[dict[str, str]]:
        """Rule-based model-improvement guidance derived from the metrics."""
        out: list[dict[str, str]] = []
        behavior_ids = metrics.get("behaviors", [])
        per_reviewer = metrics.get("per_reviewer", {})

        # Aggregate per-behavior precision/recall across reviewers.
        agg_prec: dict[str, list[float]] = defaultdict(list)
        agg_rec: dict[str, list[float]] = defaultdict(list)
        unsure_rates: list[float] = []
        for rdata in per_reviewer.values():
            if rdata.get("unsure_rate") is not None:
                unsure_rates.append(rdata["unsure_rate"])
            for b, pb in (rdata.get("per_behavior") or {}).items():
                if pb.get("precision") is not None:
                    agg_prec[b].append(pb["precision"])
                if pb.get("recall") is not None:
                    agg_rec[b].append(pb["recall"])

        def _mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        for b in behavior_ids:
            name = self._behavior_name(b)
            prec = _mean(agg_prec.get(b, []))
            rec = _mean(agg_rec.get(b, []))
            if prec is not None and prec < 0.7:
                out.append({
                    "behavior": name,
                    "severity": "high" if prec < 0.5 else "medium",
                    "message": (
                        f"{name}: model over-predicts (precision {prec:.0%}). Reviewers reject many "
                        "detections. Add hard negatives via Temporal Review FP flagging or Active Learning, "
                        "then retrain."
                    ),
                })
            if rec is not None and rec < 0.7:
                out.append({
                    "behavior": name,
                    "severity": "high" if rec < 0.5 else "medium",
                    "message": (
                        f"{name}: model misses positives (recall {rec:.0%}). Add more positive seed examples "
                        "or labeled clips for this behavior and retrain."
                    ),
                })

        inter = metrics.get("inter_rater", {})
        kappa = inter.get("kappa")
        if kappa is not None and inter.get("n_reviewers", 0) >= 2 and kappa < 0.6:
            out.append({
                "behavior": "All",
                "severity": "high" if kappa < 0.4 else "medium",
                "message": (
                    f"Reviewers disagree with each other (κ = {kappa:.2f}). Behavior definitions may be "
                    "ambiguous — refine the operational definition and inclusion/exclusion criteria in the "
                    "Behaviors tab before trusting model metrics."
                ),
            })

        mean_unsure = _mean(unsure_rates)
        if mean_unsure is not None and mean_unsure > 0.2:
            out.append({
                "behavior": "All",
                "severity": "medium",
                "message": (
                    f"High 'Unsure' rate ({mean_unsure:.0%}). Many clips are hard to judge — clarify behavior "
                    "definitions or lengthen clips for more context."
                ),
            })

        # Confusion-aware guidance: name the behaviors getting mixed up, and say
        # whether the disagreements are borderline (threshold) or clear (model error).
        confusion = metrics.get("confusion", {})
        for conf in (confusion.get("top_confusions") or [])[:3]:
            count = conf.get("count", 0)
            if count < 3:
                continue
            machine = self._label_display(conf.get("machine", ""))
            user = self._label_display(conf.get("user", ""))
            fringe = conf.get("fringe_count", 0)
            mostly_fringe = fringe >= max(1, count / 2)
            if mostly_fringe:
                msg = (
                    f"Model calls these clips '{machine}' but reviewers say '{user}' "
                    f"({count}×, mostly borderline). These sit near the detection threshold — "
                    f"tune {machine}'s onset threshold in Temporal Review rather than retraining."
                )
                sev = "medium"
            else:
                msg = (
                    f"Model confuses '{machine}' with '{user}' ({count}× on clear cases). "
                    f"Add labeled examples that contrast {machine} and {user}, then retrain."
                )
                sev = "high"
            out.append({"behavior": machine, "severity": sev, "message": msg})

        n_dis = confusion.get("n_disagreements", 0)
        n_fr = confusion.get("n_fringe_disagreements", 0)
        if n_dis >= 5 and n_fr >= 0.6 * n_dis:
            out.append({
                "behavior": "All",
                "severity": "medium",
                "message": (
                    f"Most disagreements ({n_fr}/{n_dis}) are on borderline clips. Models are largely "
                    "correct; revisit per-behavior thresholds in Temporal Review for finer control."
                ),
            })

        # Behavior-overlap guidance: behaviors flagged simultaneously suggest weak
        # inhibition. Offer a one-click mutual-inhibition fix per offending pair.
        overlap = metrics.get("overlap", {})
        for pair in (overlap.get("pairs") or [])[:3]:
            fa = pair.get("frac_a") or 0.0
            fb = pair.get("frac_b") or 0.0
            worst = max(fa, fb)
            if worst < 0.15:
                continue
            a, b = pair["a"], pair["b"]
            na, nb = self._behavior_name(a), self._behavior_name(b)
            heavier = na if fa >= fb else nb
            out.append({
                "behavior": f"{na} × {nb}",
                "severity": "high" if worst >= 0.35 else "medium",
                "message": (
                    f"{na} and {nb} are flagged at the same time on {worst:.0%} of {heavier}'s detected "
                    f"frames. Overlapping detections usually mean thresholds are too lax or inhibition is "
                    f"too weak. Add mutual inhibition between {na} and {nb}, then re-run Temporal Refinement."
                ),
                "action": "apply_inhibit",
                "pair_a": a,
                "pair_b": b,
            })

        if not out:
            out.append({
                "behavior": "All",
                "severity": "ok",
                "message": "No problems detected — reviewers and models agree well across behaviors.",
            })
        return out

    def _behavior_name(self, bid: str) -> str:
        for b, name in self._active_behaviors():
            if b == bid:
                return name
        return bid

    def _label_display(self, label: str) -> str:
        """Human-readable name for a label id (``no_behavior`` -> 'No Behavior')."""
        if _is_no_behavior(label):
            return "No Behavior"
        return self._behavior_name(label)

    # ==================================================================
    # Opt-in write-back into training labels
    # ==================================================================
    def commit_answers_to_training(self, run_id: str, reviewer_id: str) -> int:
        """Write a reviewer's non-unsure answers into reviewer_labels for training.

        Returns the number of labels committed.
        """
        run = self.load_run(run_id)
        if run is None:
            return 0
        answers = self.load_answers(run_id, reviewer_id)
        clip_ids = {c.clip_id for c in run.clips}
        committed = 0
        for cid, ans in answers.items():
            if ans.is_unsure or cid not in clip_ids:
                continue
            self._reviews.append_segment_label(ReviewerLabelRecord(
                segment_id=cid,
                review_label=ans.label,
                reviewer_id=reviewer_id,
                notes="validation_quiz",
            ))
            committed += 1
        logger.info("Committed %d validation labels to training (reviewer=%s).", committed, reviewer_id)
        return committed
