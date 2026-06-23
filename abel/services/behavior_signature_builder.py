"""Behavior signature builder — derives behavior models from seed examples.

Combines syllable assignments with user-provided seed examples to construct
a behavior signature that can be used for candidate retrieval.

Pipeline position:
    Seed Examples → Keypoint-MoSeq → Syllable Assignments
    → **Behavior Signature Builder** ← here
    → Candidate Retrieval
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from abel.models.schemas import BehaviorSignature, SeedExample
from abel.storage.file_store import read_json, read_yaml, write_json

logger = logging.getLogger("abel")


@dataclass
class BehaviorSignatureBuilderConfig:
    """Parameters for signature building."""
    behavior_id: str
    syllable_model_id: str
    seed_examples: list[SeedExample]
    min_seed_overlap_sec: float = 0.1  # Minimum time overlap between seed and window


@dataclass
class BehaviorSignatureBuilderResult:
    """Outcome of signature building."""
    behavior_id: str
    signature: BehaviorSignature | None = None
    n_seed_examples: int = 0
    n_syllable_windows_analyzed: int = 0
    warnings: list[str] = field(default_factory=list)
    success: bool = False


class BehaviorSignatureBuilder:
    """Builds behavior signatures from syllable assignments and seed examples."""

    def __init__(self) -> None:
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def build_signature(
        self,
        config: BehaviorSignatureBuilderConfig,
    ) -> BehaviorSignatureBuilderResult:
        """Build a behavior signature from seed examples.

        Steps:
          1. Map seed examples to syllable windows
          2. Compute syllable enrichment/depletion scores
          3. Extract syllable sequences and transition matrix
          4. Compute duration statistics
          5. Extract pose feature constraints (if available)
          6. Persist signature
        """
        result = BehaviorSignatureBuilderResult(behavior_id=config.behavior_id)

        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        if not config.seed_examples:
            result.warnings.append("No seed examples provided.")
            return result

        result.n_seed_examples = len(config.seed_examples)

        # ── Step 1: Collect syllables from seed-overlapping windows ───────
        seed_syllables = self._extract_seed_syllables(config)
        if not seed_syllables:
            result.warnings.append("No seed examples found in syllable assignments.")
            return result

        result.n_syllable_windows_analyzed = len(seed_syllables)

        # ── Step 2: Compute enrichment/depletion scores ──────────────────
        enriched, depleted = self._compute_enrichment(seed_syllables, config.syllable_model_id)

        # ── Step 3: Extract sequences and transitions ───────────────────
        sequences, transitions = self._extract_sequences_and_transitions(seed_syllables)

        # ── Step 4: Compute duration statistics ──────────────────────────
        duration_stats = self._compute_duration_stats(config.seed_examples)

        # ── Step 5: Extract pose constraints (if available) ──────────────
        pose_constraints = self._extract_pose_constraints(config)

        # ── Step 6: Build and persist signature ──────────────────────────
        signature = BehaviorSignature(
            signature_id=f"sig_{config.behavior_id[:10]}_{np.random.randint(0, 10000):05d}",
            behavior_id=config.behavior_id,
            syllable_model_id=config.syllable_model_id,
            enriched_syllables=enriched,
            depleted_syllables=depleted,
            syllable_sequences=sequences,
            transition_matrix=transitions,
            duration_stats=duration_stats,
            pose_constraints=pose_constraints,
            n_seed_examples=len(config.seed_examples),
            created_from=[s.seed_id for s in config.seed_examples],
        )

        self._save_signature(signature)
        result.signature = signature
        result.success = True

        logger.info(
            "Built behavior signature: %s (%d seeds, %d windows, %d enriched syllables)",
            config.behavior_id,
            len(config.seed_examples),
            len(seed_syllables),
            len(enriched),
        )
        return result

    def _extract_seed_syllables(self, config: BehaviorSignatureBuilderConfig) -> list[str]:
        """Extract syllables from windows that overlap with seed examples.

        Returns list of syllable IDs (as strings) extracted from seed-overlapping windows.
        """
        if not self._project_root:
            return []

        seed_syllables = []

        # Group seeds by session
        seeds_by_session = defaultdict(list)
        for seed in config.seed_examples:
            seeds_by_session[seed.session_id].append(seed)

        # For each session, load syllable assignments
        for session_id, seeds in seeds_by_session.items():
            syllables = self._load_syllables(session_id)
            if syllables is None:
                continue

            # For each seed, find overlapping syllable windows
            for seed in seeds:
                start_frame = seed.start_frame
                end_frame = seed.end_frame

                # Extract syllables from the seed window
                seed_range = syllables[start_frame : end_frame + 1]
                if len(seed_range) > 0:
                    seed_syllables.extend([str(s) for s in seed_range])

        return seed_syllables

    def _compute_enrichment(self, seed_syllables: list[str], model_id: str) -> tuple[dict[str, float], dict[str, float]]:
        """Compute syllable enrichment and depletion scores.

        Enriched syllables are those found frequently in seeds.
        Depleted syllables are those expected but found rarely.

        Returns (enriched_dict, depleted_dict) with scores in [0, 1].
        """
        if not seed_syllables:
            return {}, {}

        # Count syllable frequencies in seeds
        seed_counts = Counter(seed_syllables)
        total = sum(seed_counts.values())

        # Compute enrichment as frequency
        enriched = {sid: count / total for sid, count in seed_counts.items() if count > 0}

        # For depletion: syllables in the model vocabulary that are absent from seeds
        depleted: dict[str, float] = {}
        n_syllables = self._get_model_n_syllables()
        if n_syllables > 0:
            base_rate = 1.0 / n_syllables  # uniform expected frequency
            for i in range(n_syllables):
                sid = str(i)
                if sid not in enriched:
                    depleted[sid] = base_rate

        return enriched, depleted

    def _extract_sequences_and_transitions(
        self, seed_syllables: list[str]
    ) -> tuple[list[list[str]], dict[str, dict[str, float]]]:
        """Extract common syllable sequences and build transition matrix.

        Returns (sequences, transition_matrix) where:
          sequences — list of common n-gram patterns
          transition_matrix — dict of {syl_i: {syl_j: transition_prob}}
        """
        if not seed_syllables or len(seed_syllables) < 2:
            return [], {}

        # Find bigrams
        bigrams = []
        for i in range(len(seed_syllables) - 1):
            bigram = [seed_syllables[i], seed_syllables[i + 1]]
            bigrams.append(bigram)

        # Count transitions
        transition_counts = defaultdict(lambda: Counter())
        for i, syl in enumerate(seed_syllables[:-1]):
            next_syl = seed_syllables[i + 1]
            transition_counts[syl][next_syl] += 1

        # Normalize to probabilities
        transition_matrix = {}
        for syl, nexts in transition_counts.items():
            total = sum(nexts.values())
            transition_matrix[syl] = {next_syl: count / total for next_syl, count in nexts.items()}

        # Find most common sequences (bigrams, trigrams)
        sequences = []
        bigram_counts = Counter(tuple(b) for b in bigrams)
        for bigram, count in bigram_counts.most_common(5):
            sequences.append(list(bigram))

        return sequences, transition_matrix

    def _compute_duration_stats(self, seed_examples: list[SeedExample]) -> dict[str, float]:
        """Compute duration statistics from seed examples."""
        fps = self._get_project_fps()
        durations = []

        for seed in seed_examples:
            duration_frames = seed.end_frame - seed.start_frame + 1
            duration_sec = duration_frames / fps
            durations.append(duration_sec)

        if not durations:
            return {}

        durations_arr = np.array(durations)
        return {
            "mean": float(durations_arr.mean()),
            "std": float(durations_arr.std()),
            "min": float(durations_arr.min()),
            "max": float(durations_arr.max()),
            "median": float(np.median(durations_arr)),
        }

    def _extract_pose_constraints(self, config: BehaviorSignatureBuilderConfig) -> dict[str, float]:
        """Extract mean pose feature statistics from pose-feature windows overlapping seeds.

        Loads the per-session .npz files written by PoseFeaturesService and averages
        the kinematic features across all windows that overlap with seed examples.
        Returns an empty dict when no pose feature files are present.
        """
        if not self._project_root or not config.seed_examples:
            return {}

        feature_names = [
            "speed_mean", "speed_std", "speed_max",
            "disp_mean", "disp_std",
            "axis_cos_mean", "axis_sin_mean", "axis_angle_std",
            "likelihood_mean",
        ]

        seeds_by_session: dict[str, list[SeedExample]] = defaultdict(list)
        for seed in config.seed_examples:
            seeds_by_session[seed.session_id].append(seed)

        all_vecs: list[np.ndarray] = []
        for session_id, seeds in seeds_by_session.items():
            npz_path = self._project_root / "derived" / "pose_features" / f"{session_id}.npz"
            if not npz_path.exists():
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                features: np.ndarray = data["features"]       # (n_windows, n_features)
                window_frames: np.ndarray = data["window_frames"]  # (n_windows, 2)
            except Exception as exc:
                logger.warning("Failed to load pose features for %s: %s", session_id, exc)
                continue

            for seed in seeds:
                for wi in range(len(window_frames)):
                    wf_start = int(window_frames[wi, 0])
                    wf_end = int(window_frames[wi, 1])
                    overlap_start = max(wf_start, seed.start_frame)
                    overlap_end = min(wf_end, seed.end_frame)
                    if overlap_start < overlap_end:
                        all_vecs.append(features[wi])

        if not all_vecs:
            return {}

        feat_arr = np.array(all_vecs, dtype=np.float32)
        means = feat_arr.mean(axis=0)
        return {name: float(means[i]) for i, name in enumerate(feature_names) if i < len(means)}

    def _get_project_fps(self) -> float:
        """Read the project's default fps from project.yaml (falls back to 30.0)."""
        if not self._project_root:
            return 30.0
        try:
            data = read_yaml(self._project_root / "project.yaml", {})
            return float(data.get("default_fps", 30.0) or 30.0)
        except Exception:
            return 30.0

    def _get_model_n_syllables(self) -> int:
        """Return the number of syllables in the current model.

        Derives the count from the actual assignment ``.npz`` files so that a
        stale ``model_metadata.json`` (e.g. left over from a previous run with
        a different syllable count) can never inflate the vocabulary size and
        produce phantom depleted-syllable entries.  Falls back to the metadata
        value when no assignment files are present yet.
        """
        if not self._project_root:
            return 0

        # Ground-truth: scan the actual per-session assignment arrays.
        syllables_dir = self._project_root / "derived" / "syllables"
        max_obs = -1
        for npz_path in syllables_dir.glob("*_syllables.npz"):
            try:
                arr = np.load(npz_path, allow_pickle=True)["syllables"]
                if arr.size > 0:
                    max_obs = max(max_obs, int(arr.max()))
            except Exception:
                continue
        if max_obs >= 0:
            return max_obs + 1

        # Fallback: metadata (used before any assignments exist on disk).
        data = read_json(
            self._project_root / "derived" / "syllables" / "model_metadata.json", {}
        )
        return int(data.get("n_syllables", 0) or 0)

    def _load_syllables(self, session_id: str) -> np.ndarray | None:
        """Load syllable assignment array for a session."""
        if not self._project_root:
            return None

        path = self._project_root / "derived" / "syllables" / f"{session_id}_syllables.npz"
        if not path.exists():
            return None

        try:
            data = np.load(path, allow_pickle=True)
            return data["syllables"]
        except Exception as exc:
            logger.warning("Failed to load syllables for %s: %s", session_id, exc)
            return None

    def _save_signature(self, signature: BehaviorSignature) -> None:
        """Persist behavior signature."""
        if not self._project_root:
            return

        signatures_dir = self._project_root / "derived" / "behavior_signatures"
        signatures_dir.mkdir(parents=True, exist_ok=True)

        # Save signature
        path = signatures_dir / f"{signature.behavior_id}_signature.json"
        write_json(path, signature.model_dump(mode="json"))
        logger.info("Saved behavior signature: %s", signature.behavior_id)

    def load_signature(self, behavior_id: str) -> BehaviorSignature | None:
        """Load a persisted behavior signature."""
        if not self._project_root:
            return None

        # Try to find signature file
        signatures_dir = self._project_root / "derived" / "behavior_signatures"
        for sig_file in signatures_dir.glob(f"{behavior_id}_signature.json"):
            try:
                data = read_json(sig_file, {})
                return BehaviorSignature.model_validate(data)
            except Exception as exc:
                logger.warning("Failed to load signature from %s: %s", sig_file, exc)
                continue

        return None

    def load_all_signatures(self) -> dict[str, BehaviorSignature]:
        """Load all persisted behavior signatures keyed by behavior_id."""
        if not self._project_root:
            return {}

        signatures_dir = self._project_root / "derived" / "behavior_signatures"
        if not signatures_dir.exists():
            return {}

        out: dict[str, BehaviorSignature] = {}
        for sig_file in signatures_dir.glob("*_signature.json"):
            try:
                data = read_json(sig_file, {})
                sig = BehaviorSignature.model_validate(data)
                out[sig.behavior_id] = sig
            except Exception as exc:
                logger.warning("Failed to load signature from %s: %s", sig_file, exc)
        return out

    def clear_all_signatures(self) -> int:
        """Delete all saved behavior signature files. Returns count of files removed."""
        if not self._project_root:
            return 0

        signatures_dir = self._project_root / "derived" / "behavior_signatures"
        if not signatures_dir.exists():
            return 0

        removed = 0
        for sig_file in signatures_dir.glob("*_signature.json"):
            try:
                sig_file.unlink()
                removed += 1
            except Exception as exc:
                logger.warning("Could not delete signature file %s: %s", sig_file, exc)
        return removed
