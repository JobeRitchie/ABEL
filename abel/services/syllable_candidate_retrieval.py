"""Syllable-based candidate retrieval service.

Uses behavior signatures and syllable assignments to find candidate clips
matching user-defined behaviors.

Pipeline position:
    Behavior Signature → **Candidate Retrieval** ← here
    → Clip Extraction → Review Interface
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from abel.models.schemas import BehaviorSignature, CandidateWindow
from abel.storage.file_store import read_json, write_json

logger = logging.getLogger("abel")


@dataclass
class SyllableCandidateRetrievalConfig:
    """Parameters for syllable-based candidate retrieval."""
    behavior_id: str
    behavior_signature: BehaviorSignature
    session_ids: list[str]
    window_size_frames: int = 300  # Context windows
    stride_frames: int = 30  # How often to score
    top_k: int = 100
    enrichment_weight: float = 0.5
    transition_weight: float = 0.3
    context_weight: float = 0.2
    min_score_threshold: float = 0.0
    uncertainty_sampling: bool = False


@dataclass
class SyllableCandidateRetrievalResult:
    """Outcome of retrieval."""
    behavior_id: str
    n_windows_scored: int = 0
    n_candidates_selected: int = 0
    candidates: list[CandidateWindow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    success: bool = False


class SyllableCandidateRetrieval:
    """Retrieves candidate clips using behavior signatures and syllable assignments."""

    def __init__(self) -> None:
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def retrieve_candidates(
        self,
        config: SyllableCandidateRetrievalConfig,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> SyllableCandidateRetrievalResult:
        """Retrieve candidate windows using syllable and behavior signature scoring.

        Steps (0->5):
          0. Load syllable assignments
          1. Slide window across each session, score each window
          2. Compute enrichment scores based on syllable composition
          3. Compute transition scores based on syllable sequences
          4. Rank and select top-k candidates
          5. Done
        """
        result = SyllableCandidateRetrievalResult(behavior_id=config.behavior_id)
        _prog = progress_callback or (lambda _a, _b: None)

        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        sig = config.behavior_signature
        if not sig:
            result.warnings.append("No behavior signature provided.")
            return result

        # ── Step 0: Load syllable assignments ────────────────────────
        _prog(0, 5)
        syllable_data = self._load_syllables(config.session_ids)
        if not syllable_data:
            result.warnings.append("No syllable assignments found. Run syllable discovery first.")
            return result

        # ── Step 1-2: Score windows ────────────────────────────────────
        _prog(1, 5)
        candidates = []
        total_windows = sum(len(syl) for syl in syllable_data.values()) - sum(
            config.window_size_frames for _ in syllable_data
        )
        total_windows = max(1, total_windows)
        windows_scored = 0

        for session_id, syllables in syllable_data.items():
            if cancel_flag and cancel_flag[0]:
                return result

            n_frames = len(syllables)
            n_windows = max(1, n_frames - config.window_size_frames)

            for start_idx in range(0, n_windows, config.stride_frames):
                if cancel_flag and cancel_flag[0]:
                    return result

                end_idx = min(start_idx + config.window_size_frames, n_frames - 1)
                window_syllables = syllables[start_idx:end_idx]

                # Compute scores
                enrichment_score = self._compute_enrichment_score(window_syllables, sig)
                transition_score = self._compute_transition_score(window_syllables, sig)
                context_score = self._compute_bout_compactness_score(window_syllables, sig)

                total_score = (
                    config.enrichment_weight * enrichment_score
                    + config.transition_weight * transition_score
                    + config.context_weight * context_score
                )

                if total_score >= config.min_score_threshold:
                    candidate = CandidateWindow(
                        window_id=f"cand_{session_id}_{start_idx}_{np.random.randint(0, 10000):05d}",
                        session_id=session_id,
                        start_frame=int(start_idx),
                        end_frame=int(end_idx),
                        behavior_id=config.behavior_id,
                        motif_score=enrichment_score,
                        seed_similarity_score=transition_score,
                        total_score=total_score,
                    )
                    candidates.append(candidate)

                windows_scored += 1
                pct = int(100 * windows_scored / total_windows)
                _prog(1 + int(3 * pct / 100), 5)

        result.n_windows_scored = windows_scored

        # ── Step 4: Select top-k with diversity ────────────────────────
        _prog(4, 5)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        if config.uncertainty_sampling:
            # Select mix of high-confidence and uncertain candidates
            selected = self._select_with_uncertainty_diversity(
                candidates, config.top_k
            )
        else:
            # Simple top-k selection
            selected = candidates[: config.top_k]

        result.candidates = selected
        result.n_candidates_selected = len(selected)

        _prog(5, 5)
        result.success = True
        logger.info(
            "Retrieved %d candidates from %d windows for %s",
            len(selected),
            windows_scored,
            config.behavior_id,
        )
        return result

    def _load_syllables(self, session_ids: list[str]) -> dict[str, np.ndarray]:
        """Load syllable assignments for sessions."""
        if not self._project_root:
            return {}

        syllables_dir = self._project_root / "derived" / "syllables"
        result = {}

        for sid in session_ids:
            path = syllables_dir / f"{sid}_syllables.npz"
            if not path.exists():
                logger.warning("No syllables for session %s", sid)
                continue
            try:
                data = np.load(path, allow_pickle=True)
                result[sid] = data["syllables"]
            except Exception as exc:
                logger.warning("Failed to load syllables for %s: %s", sid, exc)
                continue

        return result

    def _compute_enrichment_score(self, window_syllables: np.ndarray, sig: BehaviorSignature) -> float:
        """Score window by enrichment of behavior syllables."""
        if len(window_syllables) == 0:
            return 0.0

        # Count syllables in window
        from collections import Counter
        syl_counts = Counter(window_syllables)
        total = len(window_syllables)

        # Score by enrichment
        score = 0.0
        for syl_id, count in syl_counts.items():
            syl_id_str = str(syl_id)
            if syl_id_str in sig.enriched_syllables:
                enrichment = sig.enriched_syllables[syl_id_str]
                freq = count / total
                score += enrichment * freq

        # Normalize
        return min(float(score), 1.0)

    def _compute_transition_score(self, window_syllables: np.ndarray, sig: BehaviorSignature) -> float:
        """Score window by transition structure."""
        if len(window_syllables) < 2 or not sig.transition_matrix:
            return 0.0

        # Count transition matches
        score = 0.0
        transitions_found = 0

        for i in range(len(window_syllables) - 1):
            syl_i = str(window_syllables[i])
            syl_j = str(window_syllables[i + 1])

            if syl_i in sig.transition_matrix:
                if syl_j in sig.transition_matrix[syl_i]:
                    score += sig.transition_matrix[syl_i][syl_j]
                    transitions_found += 1

        if transitions_found == 0:
            return 0.0

        return min(float(score / transitions_found), 1.0)

    def _compute_bout_compactness_score(self, window_syllables: np.ndarray, sig: BehaviorSignature) -> float:
        """Score a window by how compactly enriched syllables cluster within it.

        Returns the fraction of the window occupied by the longest contiguous run of
        enriched syllables.  1.0 = a single unbroken bout covering the full window;
        0.0 = no enriched syllables present.
        """
        if len(window_syllables) == 0 or not sig.enriched_syllables:
            return 0.0

        is_enriched = np.asarray(
            [str(s) in sig.enriched_syllables for s in window_syllables], dtype=np.float32
        )
        if not is_enriched.any():
            return 0.0

        max_run = 0
        current_run = 0
        for v in is_enriched:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return float(max_run / len(window_syllables))

    def _select_with_uncertainty_diversity(
        self, candidates: list[CandidateWindow], top_k: int
    ) -> list[CandidateWindow]:
        """Select candidates balancing confidence and diversity (for active learning)."""
        if len(candidates) <= top_k:
            return candidates

        # Sort by score
        candidates = sorted(candidates, key=lambda c: c.total_score, reverse=True)

        # Take top 50% by score (high confidence)
        high_conf_k = max(1, int(top_k * 0.5))
        selected = candidates[:high_conf_k]

        # Add uncertain candidates (middle scores) for active learning
        mid_start = high_conf_k
        mid_end = min(len(candidates), int(len(candidates) * 0.75))
        if mid_start < mid_end:
            mid_k = top_k - high_conf_k
            step = max(1, (mid_end - mid_start) // mid_k)
            selected.extend(candidates[mid_start:mid_end:step][:mid_k])

        return selected[:top_k]

    def save_candidates(self, candidates: list[CandidateWindow]) -> None:
        """Persist candidates to storage."""
        if not self._project_root:
            return

        cand_dir = self._project_root / "derived" / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)

        candidate_data = [c.model_dump(mode="json") for c in candidates]
        write_json(cand_dir / "candidates.json", {"candidates": candidate_data})
        logger.info("Saved %d candidates", len(candidates))

    def load_candidates(self) -> list[CandidateWindow]:
        """Load previously saved candidates."""
        if not self._project_root:
            return []

        cand_file = self._project_root / "derived" / "candidates" / "candidates.json"
        if not cand_file.exists():
            return []

        try:
            data = read_json(cand_file, {})
            return [CandidateWindow.model_validate(c) for c in data.get("candidates", [])]
        except Exception as exc:
            logger.warning("Failed to load candidates: %s", exc)
            return []
