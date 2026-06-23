"""Active learning scheduler for iterative candidate refinement.

After each review round, reranks candidates based on human decisions using
uncertainty sampling and diversity to focus subsequent reviews on informative examples.

Pipeline position:
    Review Interface → Human Decisions → **Active Learning** ← here
    → Updated Candidate Ranking → Next Review Round
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.models.schemas import CandidateWindow, ReviewDecision
from abel.services.active_learning_trainer_service import ActiveLearningTrainerService, TrainingConfig
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")


def _label_signature(df: pd.DataFrame) -> str:
    """Build a stable fingerprint for reviewer labels.

    This detects edits/relabels even when row count stays unchanged.
    """
    if df.empty:
        return ""

    norm = pd.DataFrame()
    norm["segment_id"] = df.get("segment_id", pd.Series(dtype=str)).astype(str)
    norm["review_label"] = df.get("review_label", pd.Series(dtype=str)).astype(str)
    norm["reviewer_id"] = df.get("reviewer_id", pd.Series(dtype=str)).astype(str)
    norm["confidence"] = pd.to_numeric(df.get("confidence", 1.0), errors="coerce").fillna(1.0).round(6)
    norm["notes"] = df.get("notes", pd.Series(dtype=str)).astype(str)
    norm = norm.sort_values(
        by=["segment_id", "reviewer_id", "review_label", "confidence", "notes"],
        kind="mergesort",
    ).reset_index(drop=True)

    payload = norm.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


@dataclass
class ActiveLearningConfig:
    """Parameters for active learning."""
    uncertainty_weight: float = 0.4
    diversity_weight: float = 0.3
    feedback_weight: float = 0.3
    batch_size: int = 20


@dataclass
class ActiveLearningResult:
    """Outcome of learning iteration."""
    n_reviewed: int = 0
    n_uncertainty_selected: int = 0
    n_diversity_selected: int = 0
    n_hard_negatives_selected: int = 0
    selected_candidates: list[CandidateWindow] = field(default_factory=list)
    success: bool = False


class ActiveLearningScheduler:
    """Selects informative candidates for next review round."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._trainer = ActiveLearningTrainerService()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def select_next_batch(
        self,
        all_candidates: list[CandidateWindow],
        reviewed_decisions: list[ReviewDecision],
        config: ActiveLearningConfig | None = None,
    ) -> ActiveLearningResult:
        """Select next batch of candidates for review using active learning strategies.

        Combines:
        1. Uncertainty sampling — candidates with scores near decision boundary
        2. Diversity sampling — candidates far from previously reviewed examples
        3. Hard negative sampling — candidates with conflicting review outcomes
        """
        config = config or ActiveLearningConfig()
        result = ActiveLearningResult()

        if not all_candidates:
            return result

        # Compute uncertainties for each candidate
        uncertainties = self._compute_uncertainties(all_candidates)

        # Build reviewed set for diversity
        reviewed_ids = {d.clip_id for d in reviewed_decisions}
        reviewed_candidates = [c for c in all_candidates if c.window_id in reviewed_ids]
        behavior_weights = self._behavior_focus_weights(all_candidates, reviewed_decisions)

        # Selection strategy
        selection_counts = {
            "uncertainty": int(config.batch_size * config.uncertainty_weight),
            "diversity": int(config.batch_size * config.diversity_weight),
            "hard_negative": int(config.batch_size * (1 - config.uncertainty_weight - config.diversity_weight)),
        }

        # 1. Uncertainty sampling
        unreviewed = [c for c in all_candidates if c.window_id not in reviewed_ids]
        uncertain_candidates = self._select_by_uncertainty(
            unreviewed,
            uncertainties,
            selection_counts["uncertainty"],
            behavior_weights,
        )
        result.selected_candidates.extend(uncertain_candidates)
        result.n_uncertainty_selected = len(uncertain_candidates)

        # 2. Diversity sampling
        diversity_candidates = self._select_by_diversity(
            unreviewed,
            reviewed_candidates,
            selection_counts["diversity"],
            behavior_weights,
        )
        result.selected_candidates.extend(diversity_candidates)
        result.n_diversity_selected = len(diversity_candidates)

        # 3. Hard negative sampling (conflicting examples)
        if reviewed_candidates:
            hard_negatives = self._find_hard_negatives(
                all_candidates,
                reviewed_decisions,
                selection_counts["hard_negative"],
                behavior_weights,
            )
            result.selected_candidates.extend(hard_negatives)
            result.n_hard_negatives_selected = len(hard_negatives)

        # Remove duplicates
        seen = set()
        unique = []
        for c in result.selected_candidates:
            if c.window_id not in seen:
                unique.append(c)
                seen.add(c.window_id)
        result.selected_candidates = unique

        # Sort by score (descending)
        result.selected_candidates.sort(key=lambda c: c.total_score, reverse=True)

        result.n_reviewed = len(reviewed_decisions)
        result.success = True
        logger.info(
            "Selected next batch: %d uncertainty + %d diversity + %d hard-negatives",
            result.n_uncertainty_selected,
            result.n_diversity_selected,
            result.n_hard_negatives_selected,
        )
        return result

    def _compute_uncertainties(self, candidates: list[CandidateWindow]) -> dict[str, float]:
        """Compute uncertainty (distance from decision boundary) for each candidate.

        Higher uncertainty = closer to 0.5 (50% confidence).
        Lower uncertainty = more confident (close to 0 or 1).
        """
        uncertainties = {}
        scores = np.array([c.total_score for c in candidates])

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.ones_like(scores) * 0.5

        for candidate, score in zip(candidates, scores_norm):
            # Uncertainty = distance to 0.5
            uncertainty = 1.0 - abs(score - 0.5) * 2
            uncertainties[candidate.window_id] = float(np.clip(uncertainty, 0, 1))

        return uncertainties

    def _select_by_uncertainty(
        self,
        candidates: list[CandidateWindow],
        uncertainties: dict[str, float],
        n_select: int,
        behavior_weights: dict[str, float] | None = None,
    ) -> list[CandidateWindow]:
        """Select candidates with highest uncertainty."""
        if not candidates:
            return []

        behavior_weights = behavior_weights or {}

        # Sort by uncertainty (highest first)
        sorted_cands = sorted(
            candidates,
            key=lambda c: (
                uncertainties.get(c.window_id, 0)
                + 0.25 * behavior_weights.get(str(c.behavior_id or ""), 0.0),
                c.total_score,
            ),
            reverse=True,
        )
        return sorted_cands[:n_select]

    def _select_by_diversity(
        self,
        unreviewed: list[CandidateWindow],
        reviewed: list[CandidateWindow],
        n_select: int,
        behavior_weights: dict[str, float] | None = None,
    ) -> list[CandidateWindow]:
        """Select candidates most diverse from reviewed set."""
        if not unreviewed or not reviewed:
            return unreviewed[:n_select]

        behavior_weights = behavior_weights or {}

        # Compute average score of reviewed
        avg_reviewed_score = float(np.mean([c.total_score for c in reviewed]))

        # Select candidates furthest from average (most different)
        sorted_cands = sorted(
            unreviewed,
            key=lambda c: (
                abs(c.total_score - avg_reviewed_score)
                + 0.20 * behavior_weights.get(str(c.behavior_id or ""), 0.0),
                c.total_score,
            ),
            reverse=True,
        )
        return sorted_cands[:n_select]

    def _find_hard_negatives(
        self,
        all_candidates: list[CandidateWindow],
        reviewed_decisions: list[ReviewDecision],
        n_select: int,
        behavior_weights: dict[str, float] | None = None,
    ) -> list[CandidateWindow]:
        """Find conflicting/ambiguous examples for clarification.

        These are candidates similar to rejected examples but scored high,
        or similar to accepted examples but scored low.
        """
        if not reviewed_decisions:
            return []

        behavior_weights = behavior_weights or {}

        # Build decision map
        decisions_by_id = {d.clip_id: d for d in reviewed_decisions}
        candidate_by_id = {c.window_id: c for c in all_candidates}
        rejected_scores = [
            all_candidates[i].total_score
            for i, c in enumerate(all_candidates)
            if c.window_id in decisions_by_id
            and decisions_by_id[c.window_id].new_status == "rejected"
        ]

        if not rejected_scores:
            return []

        # Find candidates with ambiguous characteristics
        # (score-close to rejected examples, plus relabel-confusion neighborhoods)
        avg_rejected = float(np.mean(rejected_scores))

        relabel_to = {
            str(d.behavior_label or "").strip()
            for d in reviewed_decisions
            if str(d.decision).lower().endswith("relabel") and str(d.behavior_label or "").strip()
        }

        hard_negatives = [
            c
            for c in all_candidates
            if c.window_id not in decisions_by_id
            and (
                abs(c.total_score - avg_rejected) < 0.2
                or str(c.behavior_id or "").strip() in relabel_to
            )
        ]

        hard_negatives.sort(
            key=lambda c: (
                abs(c.total_score - avg_rejected)
                - 0.20 * behavior_weights.get(str(c.behavior_id or ""), 0.0),
                -c.total_score,
            )
        )
        return hard_negatives[:n_select]

    def _behavior_focus_weights(
        self,
        all_candidates: list[CandidateWindow],
        reviewed_decisions: list[ReviewDecision],
    ) -> dict[str, float]:
        """Compute per-behavior selection weights from review coverage and confusion."""
        candidate_by_id = {c.window_id: c for c in all_candidates}
        reviewed_count: dict[str, int] = {}
        confusion_count: dict[str, int] = {}

        for decision in reviewed_decisions:
            cand = candidate_by_id.get(str(decision.clip_id))
            if cand is None:
                continue
            behavior = str(cand.behavior_id or "").strip()
            if not behavior:
                continue
            reviewed_count[behavior] = int(reviewed_count.get(behavior, 0)) + 1

            relabel = str(decision.behavior_label or "").strip()
            if relabel and relabel != behavior:
                confusion_count[behavior] = int(confusion_count.get(behavior, 0)) + 1

        behavior_universe = {str(c.behavior_id or "").strip() for c in all_candidates if str(c.behavior_id or "").strip()}
        if not behavior_universe:
            return {}

        weights: dict[str, float] = {}
        for behavior in behavior_universe:
            n = float(reviewed_count.get(behavior, 0))
            confusion = float(confusion_count.get(behavior, 0))
            scarcity = 1.0 / (1.0 + n)
            confusion_rate = confusion / max(1.0, n)
            weights[behavior] = float(scarcity + confusion_rate)

        max_w = max(weights.values()) if weights else 0.0
        if max_w <= 0:
            return {k: 0.0 for k in weights}
        return {k: float(v / max_w) for k, v in weights.items()}

    def save_learning_history(self, round_num: int, result: ActiveLearningResult) -> None:
        """Log active learning iteration for reproducibility."""
        if not self._project_root:
            return

        al_dir = self._project_root / "derived" / "active_learning"
        al_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "round": round_num,
            "n_reviewed": result.n_reviewed,
            "n_uncertainty": result.n_uncertainty_selected,
            "n_diversity": result.n_diversity_selected,
            "n_hard_negatives": result.n_hard_negatives_selected,
            "candidates_selected": [
                c.model_dump(mode="json") for c in result.selected_candidates
            ],
        }

        write_json(al_dir / f"round_{round_num:03d}.json", history)
        logger.info("Saved active learning round %d", round_num)

    def retrain_if_new_labels(
        self,
        config: TrainingConfig | None = None,
        require_new_labels: bool = True,
        session_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not self._project_root:
            return None

        labels_path = self._project_root / "derived" / "review_labels" / "reviewer_labels.parquet"
        features_path = self._project_root / "derived" / "representations" / "segment_features.parquet"
        if not labels_path.exists() or not features_path.exists():
            return None

        labels_df = read_json(
            self._project_root / "derived" / "active_learning" / "label_cache.json",
            {"n_rows": 0, "label_signature": ""},
        )
        lbl = pd.read_parquet(labels_path)
        if lbl.empty:
            return None

        required = {"segment_id", "review_label"}
        if not required.issubset(set(lbl.columns)):
            logger.warning("Reviewer labels parquet is missing required columns: %s", sorted(required - set(lbl.columns)))
            return None

        if "reviewer_id" not in lbl.columns:
            lbl["reviewer_id"] = "reviewer"
        if "confidence" not in lbl.columns:
            lbl["confidence"] = 1.0

        lbl["segment_id"] = lbl["segment_id"].astype(str)
        lbl["review_label"] = lbl["review_label"].astype(str)
        lbl["reviewer_id"] = lbl["reviewer_id"].astype(str)
        lbl["confidence"] = pd.to_numeric(lbl["confidence"], errors="coerce").fillna(1.0)

        if not lbl.empty:
            lbl = lbl.drop_duplicates(subset=["segment_id", "reviewer_id"], keep="last")

        n_rows = int(len(lbl))
        signature = _label_signature(lbl)
        prev_rows = int(labels_df.get("n_rows", 0))
        prev_signature = str(labels_df.get("label_signature", ""))
        if require_new_labels and n_rows <= prev_rows and signature == prev_signature:
            return None

        seg = pd.read_parquet(features_path)
        grouped = lbl.groupby("segment_id", as_index=False).agg(
            review_labels=("review_label", lambda s: [str(v) for v in s if str(v)]),
            reviewer_id=("reviewer_id", lambda s: ",".join(sorted(set(str(v) for v in s if str(v))))),
            confidence=("confidence", "mean"),
        )

        def _resolve(labels: list[str]) -> str:
            uniq = sorted(set(labels))
            if not uniq:
                return "ambiguous"
            informative = [
                x
                for x in uniq
                if x not in {"ambiguous", "boundary_error", "no_behavior"} and not x.startswith("not_")
            ]
            if len(informative) == 1:
                return informative[0]
            if len(informative) > 1:
                return "ambiguous"
            neg = [x for x in uniq if x.startswith("not_") or x == "no_behavior"]
            if len(neg) == 1:
                return neg[0]
            return "ambiguous"

        grouped["label"] = grouped["review_labels"].apply(_resolve)
        grouped = grouped.rename(columns={"reviewer_id": "label_source", "confidence": "reviewer_confidence"})

        train_df = seg.merge(
            grouped[["segment_id", "label", "label_source", "reviewer_confidence"]],
            on="segment_id",
            how="inner",
        )
        if train_df.empty:
            logger.warning("Retrain skipped: reviewer labels do not map to current segment feature rows.")
            return None

        self._trainer.merge_and_snapshot_training_set(self._project_root, train_df)
        result = self._trainer.train(self._project_root, config, session_ids=session_ids)
        write_json(
            self._project_root / "derived" / "active_learning" / "label_cache.json",
            {"n_rows": n_rows, "label_signature": signature},
        )
        return result
