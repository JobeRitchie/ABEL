"""Behavioral Motif Analysis service.

Provides three layers of sequential-behavior analysis, all operating on
the bout sequences produced by the temporal refinement step:

1. **Behavior Transition Matrix**: directed transition probabilities or
   counts between successive bouts within a user-defined gap threshold.

2. **Behavioral Motif Discovery**: recurring behavior sub-sequences
   discovered via N-gram frequency analysis or session-level sequence
   clustering (UMAP + HDBSCAN).

3. **Hidden Markov Model Analysis**: latent state discovery via a
   categorical HMM fitted to the pooled behavior sequences, with AIC/BIC
   guided model selection.

All heavy computation is performed in this service layer; the UI widget
dispatches calls via QThreadPool / TaskWorker and never blocks the event loop.

Optional dependencies (graceful degradation if absent):
  hmmlearn , HMM analysis
  umap-learn / umap, sequence clustering UMAP reduction
  hdbscan  , sequence clustering density-based clustering
  openpyxl , Excel export (falls back to CSV-only)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class MotifSettings:
    """All configurable parameters for behavioral motif analysis.

    Saved to / loaded from ``{project_root}/config/motif_settings.json``.
    """
    # -- Transition matrix --------------------------------------------------
    max_transition_gap_s: float = 5.0
    """Max gap (seconds) between end of bout A and start of bout B for a
    transition A→B to be counted."""
    bout_overlap_tolerance_s: float = 1.0
    """How many seconds before bout A ends that bout B is allowed to start and
    still be counted as a transition A→B.  Temporal-refinement models run
    independently and can produce bouts that overlap by a fraction of a second;
    without a tolerance those transitions are silently dropped.  Set to 0 to
    restore the old strict behavior (B must start after A ends)."""
    normalize_rows: bool = True
    """When True display row-normalized probabilities; when False show raw counts."""
    include_self_transitions: bool = False
    """Whether to count A→A transitions (same behavior following itself)."""

    # -- N-gram motif discovery ---------------------------------------------
    motif_method: str = "both"
    """'ngram' | 'sequence_clustering' | 'both'"""
    ngram_min_n: int = 2
    ngram_max_n: int = 4
    ngram_top_k: int = 15
    min_ngram_count: int = 2
    """Minimum global occurrence count for a motif to appear in results."""

    # -- Sequence clustering ------------------------------------------------
    umap_n_components: int = 10
    umap_n_neighbors: int = 10
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 3
    hdbscan_min_samples: int = 1
    cluster_ngram_n: int = 3
    """N-gram length used to build session feature vectors for clustering."""

    # -- HMM ----------------------------------------------------------------
    hmm_n_states_mode: str = "auto"
    """'auto' (AIC/BIC model selection) | 'manual' (exact n_states)."""
    hmm_n_states: int = 4
    """Used when hmm_n_states_mode == 'manual'."""
    hmm_n_states_min: int = 2
    hmm_n_states_max: int = 8
    hmm_n_iter: int = 200
    hmm_n_restarts: int = 5
    """Number of random restarts when fitting each HMM (best log-likelihood kept)."""
    hmm_criterion: str = "bic"
    """'aic' | 'bic' | 'aicc' | 'icl' | 'cv': criterion used for automatic model
    selection.

    - 'bic'/'aic': classical information criteria.  Both are known to over-select
      states for behavioral sequence data (Pohle et al. 2017, JABES 22:270-293).
    - 'aicc': AIC with the small-sample correction; use when N/n_free < ~40.
    - 'icl':  BIC plus twice the entropy of the posterior state assignments
      (Biernacki, Celeux & Govaert 2000).  Favors states that are actually
      separable, which is what makes an emission heatmap interpretable.
    - 'cv':   leave-one-session-out cross-validated held-out log-likelihood with
      a 1-SE parsimony rule.  Slowest, but the only criterion here that measures
      generalization to a held-out animal rather than in-sample fit."""
    hmm_random_seed: int = 0
    """Base seed for HMM EM initialization.  Restart *r* uses ``seed + r``, so a
    given (data, settings) pair always reproduces the same fit.  Without this the
    reported state count can change between runs of the same analysis."""

    hmm_occupancy_method: str = "viterbi"
    """How per-session fractional occupancy is measured: 'viterbi' | 'posterior'.

    - 'viterbi': the fraction of a session's bouts that the single most likely
      state path assigns to each state.  Every bout counts for exactly one
      state.
    - 'posterior': the mean over bouts of the forward-backward posterior
      P(state | whole sequence), i.e. *expected* fractional occupancy.  A bout
      splits its weight across the states that could plausibly have produced
      it.

    The two differ most for states whose defining behavior is rare.  Viterbi
    decoding is a global optimization, so entering a state for one or two bouts
    must pay the transition cost twice; when that exceeds the emission gain the
    path stays put and the session reports exactly 0.0 occupancy even though the
    behavior occurred.  Posterior occupancy has no such threshold - it returns a
    small non-zero value - which matters when occupancy is the dependent
    variable in a group comparison and a run of hard zeros would otherwise be
    read as "this animal never did it".

    Both are legitimate; they answer different questions.  Use 'viterbi' when
    the claim is about the discrete state sequence (which state the animal was
    *in*), 'posterior' when the claim is about how much evidence there is for
    each state.  The ethogram, state bouts and latency views always use the
    Viterbi path, because a state bout has to be a definite interval."""

    # -- HMM state bouts (ethogram / boutframes export / latency) -----------
    hmm_state_bout_source: str = "viterbi"
    """Which intervals the state-boutframes export writes: 'viterbi' | 'posterior_topn'.

    - 'viterbi': the state bouts the decoded path actually visited.  A subject
      the path never routes into a state contributes no intervals for it, and
      therefore no downstream value (a photometry mean over that state, say).
    - 'posterior_topn': for every subject and every state, the
      ``hmm_state_bout_top_n`` bouts whose posterior probability of that state
      is highest, whether or not they won the Viterbi decode.

    'posterior_topn' exists because an empty interval set is not a zero - it is
    a missing value, and a subject missing from half the states cannot enter a
    within-subject comparison.  It is defensible only when applied to *every*
    subject, which is what this option does: one selection rule, the same
    number of intervals per subject per state, so a group difference cannot be
    an artefact of which subjects got which rule.  Using it as a patch for only
    the subjects with no winners would confound the comparison with the rule.

    Its cost is that it is guaranteed non-empty by construction, so it can no
    longer answer "did this subject enter the state at all".  The intervals
    carry ``posterior_mean`` and the session's ``expected_bouts`` for exactly
    that reason: report them, and drop or flag subjects whose selected bouts
    carry negligible posterior rather than letting them read as state time."""
    hmm_state_bout_top_n: int = 5
    """How many bouts per subject per state 'posterior_topn' selects."""
    hmm_state_bout_max_gap_s: float = 0.0
    """Gap (seconds) above which a silence breaks a run of same-state bouts.
    0 means never break on time: a state bout is a maximal run of consecutive
    bouts sharing a state, however far apart they are."""
    hmm_state_min_dwell_s: float = 0.0
    """Minimum state-bout duration for that bout to count as *entering* the
    state, used by the latency metric.  0 counts every entry, including
    single-bout flickers through a state."""

    # -- Permutation testing ------------------------------------------------
    n_permutations: int = 1000
    permutation_seed: int = 42
    transition_pval_correction: str = "fdr_bh"
    """Multiple-comparison correction for transition-matrix cell-wise p-values.

    Supported values:
    - "fdr_bh": Benjamini-Hochberg FDR correction (default)
    - "none": no correction (raw permutation p-values)
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotifSettings":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def load_motif_settings(project_root: Path) -> MotifSettings:
    """Load settings from ``{project_root}/config/motif_settings.json``."""
    path = project_root / "config" / "motif_settings.json"
    if not path.exists():
        return MotifSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MotifSettings.from_dict(raw)
    except Exception:
        logger.warning("Could not load motif_settings.json; using defaults.")
        return MotifSettings()


def save_motif_settings(project_root: Path, settings: MotifSettings) -> None:
    """Persist settings to ``{project_root}/config/motif_settings.json``."""
    path = project_root / "config" / "motif_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Could not save motif_settings.json.")


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def build_sequences(
    raw_bouts: dict[str, pd.DataFrame],
    fps: float,
    selected_bids: set[str] | None = None,
) -> dict[str, list[tuple[float, float, str]]]:
    """Convert raw bout DataFrames into per-session ordered event lists.

    Parameters
    ----------
    raw_bouts:
        ``{behavior_id: DataFrame}`` with columns ``session_id``,
        ``start_frame``, ``end_frame`` (from BehaviorAnalyticsTab._raw_bouts).
    fps:
        Project frames-per-second (used to convert frames → seconds).
    selected_bids:
        If provided, restrict to these behavior IDs.

    Returns
    -------
    ``{session_id: [(start_s, end_s, behavior_id), ...]}`` sorted by start_s.
    Each session's list contains ALL selected behaviors interleaved in time.
    """
    events: dict[str, list[tuple[float, float, str]]] = {}
    for bid, bdf in raw_bouts.items():
        if selected_bids is not None and bid not in selected_bids:
            continue
        if bdf.empty or not {"session_id", "start_frame", "end_frame"}.issubset(bdf.columns):
            continue
        for _, row in bdf.iterrows():
            sid = str(row["session_id"])
            start_s = float(row["start_frame"]) / fps
            end_s = float(row["end_frame"]) / fps
            events.setdefault(sid, []).append((start_s, end_s, bid))

    for sid in events:
        events[sid].sort(key=lambda x: x[0])

    return events


def filter_overlapping_events(
    events: list[tuple[float, float, str]],
    overlap_tolerance_s: float = 1.0,
) -> list[tuple[float, float, str]]:
    """Remove events that are genuinely concurrent with the preceding event.

    When multiple per-behavior temporal-refinement models run independently
    they can produce bouts that overlap by more than a fraction of a second.
    A following event B whose start is more than *overlap_tolerance_s* seconds
    before the previous event A ends is considered truly concurrent (not a
    sequential transition) and is dropped from the linear event stream.

    Set *overlap_tolerance_s* to 0 to drop ALL overlapping events; set it to
    a large value (or negative) to keep everything.

    This should be applied before N-gram extraction, sequence clustering, and
    HMM encoding so that concurrent bouts do not generate spurious motifs.
    """
    if not events:
        return []
    out: list[tuple[float, float, str]] = [events[0]]
    for ev in events[1:]:
        prev_end = out[-1][1]
        # Allow small overlaps (within tolerance) but drop deep concurrencies
        if ev[0] >= prev_end - overlap_tolerance_s:
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def compute_transition_matrix(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    max_gap_s: float,
    include_self: bool = False,
    overlap_tolerance_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute raw transition count matrices per session.

    A transition A→B is counted when:
      - bout B starts no more than ``overlap_tolerance_s`` seconds *before*
        bout A ends (allowing for the small overlaps produced by independent
        per-behavior temporal-refinement models), AND
      - bout B starts no more than ``max_gap_s`` seconds *after* bout A ends.
      - A != B unless ``include_self`` is True.
      - Each bout B is counted at most once per preceding bout A
        (first occurrence within the window wins).

    Parameters
    ----------
    overlap_tolerance_s:
        Seconds of early-start overlap to tolerate.  E.g. 1.0 means a bout B
        that starts up to 1 second *before* bout A ends still counts as a
        transition A→B.  Set to 0 for the strict (legacy) behavior where B
        must start strictly after A ends.

    Returns
    -------
    ``{session_id: n x n float64 array}``, raw counts, NOT normalized.
    """
    n = len(behavior_ids)
    bid_idx = {bid: i for i, bid in enumerate(behavior_ids)}
    result: dict[str, np.ndarray] = {}

    for sid, events in sequences.items():
        mat = np.zeros((n, n), dtype=np.float64)
        m = len(events)
        for i, (start_a, end_a, bid_a) in enumerate(events):
            idx_a = bid_idx.get(bid_a)
            if idx_a is None:
                continue
            seen: set[int] = set()
            for j in range(i + 1, m):
                start_b, _, bid_b = events[j]
                if start_b > end_a + max_gap_s:
                    break
                # Skip if B started well before A ended (deep overlap means the
                # two bouts are truly concurrent, not a transition).  A small
                # overlap of <= overlap_tolerance_s is allowed because
                # independent per-behavior models often fire a few frames early.
                if start_b < end_a - overlap_tolerance_s:
                    continue
                idx_b = bid_idx.get(bid_b)
                if idx_b is None:
                    continue
                if not include_self and idx_a == idx_b:
                    continue
                if idx_b in seen:
                    continue
                mat[idx_a, idx_b] += 1.0
                seen.add(idx_b)
        result[sid] = mat

    return result


def normalize_transition_matrix(count_mat: np.ndarray) -> np.ndarray:
    """Row-normalize a count matrix to obtain transition probabilities."""
    prob = np.zeros_like(count_mat)
    row_sums = count_mat.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    prob[mask] = count_mat[mask] / row_sums[mask]
    return prob


def group_mean_matrix(
    per_session_mats: dict[str, np.ndarray],
    session_to_group: dict[str, str],
) -> dict[str, np.ndarray]:
    """Average transition matrices within each group.

    Returns ``{group_name: mean_matrix}``.
    """
    accum: dict[str, list[np.ndarray]] = {}
    for sid, mat in per_session_mats.items():
        grp = session_to_group.get(sid)
        if grp:
            accum.setdefault(grp, []).append(mat)
    return {grp: np.mean(mats, axis=0) for grp, mats in accum.items() if mats}


def permutation_test_transition(
    group_a_mats: list[np.ndarray],
    group_b_mats: list[np.ndarray],
    n_permutations: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Cell-wise permutation test comparing two groups' transition matrices.

    Observed statistic: |mean_A - mean_B| per cell.
    Null distribution: shuffle group labels and recompute per cell.

    p-values use the ``(b + 1) / (m + 1)`` estimator: the observed arrangement is
    itself one of the possible label assignments, so it belongs in the null count.
    The naive ``b / m`` can return exactly zero, which is not a valid p-value and
    survives FDR correction as an apparently infinitely significant cell
    (Phipson & Smyth 2010, Stat Appl Genet Mol Biol 9:39).  The smallest value
    this can return is ``1 / (n_permutations + 1)``.

    Returns
    -------
    p-value matrix (same shape as each input matrix).
    """
    all_mats = group_a_mats + group_b_mats
    if len(all_mats) == 0 or all_mats[0].ndim != 2:
        shape = (1, 1)
        if all_mats:
            shape = all_mats[0].shape
        return np.ones(shape)

    n_a = len(group_a_mats)
    stacked = np.stack(all_mats, axis=0)  # shape: (n_total, n, n)
    observed = np.abs(stacked[:n_a].mean(axis=0) - stacked[n_a:].mean(axis=0))

    rng = np.random.default_rng(seed)
    null_counts = np.zeros_like(observed)
    n_total = len(all_mats)

    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_a = stacked[perm[:n_a]].mean(axis=0)
        perm_b = stacked[perm[n_a:]].mean(axis=0)
        diff = np.abs(perm_a - perm_b)
        null_counts += (diff >= observed).astype(float)

    return (null_counts + 1.0) / (n_permutations + 1.0)


# ---------------------------------------------------------------------------
# N-gram motif analysis
# ---------------------------------------------------------------------------

def extract_ngrams_from_sequences(
    sequences: dict[str, list[tuple[float, float, str]]],
    n: int,
    *,
    filter_uniform: bool = True,
) -> dict[str, Counter[tuple[str, ...]]]:
    """Extract all n-grams of length *n* from each session's behavior sequence.

    The caller is responsible for deduplicating consecutive same-behavior
    events *before* passing ``sequences`` here (the UI checkbox controls that).

    ``filter_uniform`` (default True) additionally discards any n-gram where
    every element is identical (e.g. ``(groom, groom, groom)``), which are
    uninformative regardless of deduplication.

    Returns ``{session_id: Counter({(bid1, bid2, ...): count})}``
    """
    result: dict[str, Counter[tuple[str, ...]]] = {}
    for sid, events in sequences.items():
        behavior_seq = tuple(e[2] for e in events)
        c: Counter[tuple[str, ...]] = Counter()
        for i in range(len(behavior_seq) - n + 1):
            gram = behavior_seq[i : i + n]
            if filter_uniform and len(set(gram)) == 1:
                continue
            c[gram] += 1
        result[sid] = c
    return result


def aggregate_ngrams(
    per_session_ngrams: dict[str, Counter[tuple[str, ...]]],
    top_k: int = 15,
    min_count: int = 2,
    behavior_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate n-gram counts across all sessions.

    Returns list of dicts, sorted by total count descending, capped at top_k.
    Each dict: ``{'motif': tuple, 'motif_label': str, 'total': int, 'per_session': dict}``.
    """
    total_counter: Counter[tuple[str, ...]] = Counter()
    for counts in per_session_ngrams.values():
        total_counter.update(counts)

    results: list[dict[str, Any]] = []
    for gram, total in total_counter.most_common():
        if total < min_count:
            break
        per_session = {sid: per_session_ngrams[sid].get(gram, 0)
                       for sid in per_session_ngrams}
        if behavior_names:
            label = " → ".join(behavior_names.get(b, b) for b in gram)
        else:
            label = " → ".join(gram)
        results.append({
            "motif": gram,
            "motif_label": label,
            "total": total,
            "per_session": per_session,
        })
        if len(results) >= top_k:
            break

    return results


def ngram_group_comparison(
    per_session_ngrams: dict[str, Counter[tuple[str, ...]]],
    session_to_group: dict[str, str],
    top_k: int = 15,
    min_count: int = 2,
    n_permutations: int = 1000,
    seed: int = 42,
    behavior_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compare n-gram frequencies across two groups using permutation tests.

    Returns list of result dicts sorted by observed group difference.
    Each dict includes motif, per-group means, and permutation p-value.
    """
    groups = sorted({g for g in session_to_group.values() if g})
    if len(groups) < 2:
        return aggregate_ngrams(per_session_ngrams, top_k, min_count, behavior_names)

    # Collect all motifs with sufficient total count
    total_counter: Counter[tuple[str, ...]] = Counter()
    for counts in per_session_ngrams.values():
        total_counter.update(counts)

    candidate_motifs = [m for m, c in total_counter.most_common() if c >= min_count][:max(top_k * 5, 100)]

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for motif in candidate_motifs:
        group_vals: dict[str, list[float]] = {g: [] for g in groups}
        for sid, counts in per_session_ngrams.items():
            grp = session_to_group.get(sid)
            if grp in group_vals:
                group_vals[grp].append(float(counts.get(motif, 0)))

        # Skip if any group has no sessions
        if any(len(v) == 0 for v in group_vals.values()):
            continue

        means = {g: float(np.mean(v)) for g, v in group_vals.items()}

        # Compute ALL pairwise permutation tests
        pval_pairs: dict[str, float] = {}
        observed_diff = 0.0
        for g1, g2 in combinations(groups, 2):
            v1 = np.array(group_vals[g1])
            v2 = np.array(group_vals[g2])
            all_vals_pair = np.concatenate([v1, v2])
            obs = abs(v1.mean() - v2.mean())
            if obs > observed_diff:
                observed_diff = obs  # track max diff for sorting
            n1 = len(v1)
            n_total_pair = len(all_vals_pair)
            null_count = 0
            for _ in range(n_permutations):
                perm = rng.permutation(n_total_pair)
                null_diff = abs(
                    all_vals_pair[perm[:n1]].mean()
                    - all_vals_pair[perm[n1:]].mean()
                )
                if null_diff >= obs:
                    null_count += 1
            pval_pairs[f"{g1} vs {g2}"] = null_count / n_permutations

        # Overall pval = minimum across all pairs (most conservative display)
        pval = min(pval_pairs.values()) if pval_pairs else 1.0

        if behavior_names:
            label = " → ".join(behavior_names.get(b, b) for b in motif)
        else:
            label = " → ".join(motif)

        per_session = {sid: per_session_ngrams[sid].get(motif, 0)
                       for sid in per_session_ngrams}

        results.append({
            "motif": motif,
            "motif_label": label,
            "total": total_counter[motif],
            "per_session": per_session,
            "group_means": means,
            "observed_diff": observed_diff,
            "pval": pval,
            "pval_pairs": pval_pairs,  # all pairwise p-values
        })

    # Sort by observed_diff descending, then keep top_k
    results.sort(key=lambda x: -x.get("observed_diff", 0))
    return results[:top_k]


# ---------------------------------------------------------------------------
# Sequence clustering
# ---------------------------------------------------------------------------

def session_ngram_vectors(
    sequences: dict[str, list[tuple[float, float, str]]],
    n: int = 3,
    min_vocab_count: int = 2,
) -> tuple[np.ndarray, list[str], list[tuple[str, ...]]]:
    """Build session-level n-gram count vectors.

    Returns
    -------
    vectors : float array of shape (n_sessions, n_vocab)
    session_ids : list of session IDs (matching row order)
    vocab : list of n-gram tuples (matching column order)
    """
    per_session = extract_ngrams_from_sequences(sequences, n)
    # Build vocabulary from motifs occurring in ≥ min_vocab_count sessions
    motif_session_count: Counter[tuple[str, ...]] = Counter()
    for counts in per_session.values():
        motif_session_count.update(set(counts.keys()))
    vocab = [m for m, c in motif_session_count.most_common() if c >= min_vocab_count]
    if not vocab:
        vocab = list({g for counts in per_session.values() for g in counts})

    session_ids = sorted(per_session.keys())
    vectors = np.zeros((len(session_ids), len(vocab)), dtype=np.float64)
    for i, sid in enumerate(session_ids):
        counts = per_session[sid]
        for j, gram in enumerate(vocab):
            vectors[i, j] = float(counts.get(gram, 0))

    # TF-IDF-style normalization: divide by row sum then apply log1p
    row_sums = vectors.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    vectors = np.log1p(vectors / row_sums)

    return vectors, session_ids, vocab


def cluster_sessions(
    sequences: dict[str, list[tuple[float, float, str]]],
    settings: MotifSettings,
) -> dict[str, Any]:
    """Cluster sessions based on their N-gram profile.

    Returns dict with keys:
      'session_ids', 'labels', 'embedding' (2-D UMAP), 'n_clusters',
      'error' (str | None)
    """
    vectors, session_ids, vocab = session_ngram_vectors(
        sequences, n=settings.cluster_ngram_n
    )

    if len(session_ids) < 3:
        return {
            "session_ids": session_ids,
            "labels": np.zeros(len(session_ids), dtype=int),
            "embedding": np.zeros((len(session_ids), 2)),
            "n_clusters": 1,
            "error": "Need at least 3 sessions for clustering.",
        }

    # -- UMAP reduction ---------------------------------------------------
    try:
        import umap  # type: ignore[import-untyped]
        n_comp = min(settings.umap_n_components, vectors.shape[1], len(session_ids) - 1)
        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=min(settings.umap_n_neighbors, len(session_ids) - 1),
            min_dist=settings.umap_min_dist,
            random_state=42,
            verbose=False,
        )
        embedding_high = reducer.fit_transform(vectors)
    except ImportError:
        logger.warning("umap-learn not installed; using raw n-gram vectors.")
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        n_comp = min(2, vectors.shape[1])
        pca = PCA(n_components=n_comp)
        embedding_high = pca.fit_transform(vectors)
    except Exception as exc:
        return {
            "session_ids": session_ids,
            "labels": np.zeros(len(session_ids), dtype=int),
            "embedding": np.zeros((len(session_ids), 2)),
            "n_clusters": 1,
            "error": f"Dimensionality reduction failed: {exc}",
        }

    # 2-D UMAP for visualization
    try:
        import umap  # type: ignore[import-untyped]
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=min(settings.umap_n_neighbors, len(session_ids) - 1),
            min_dist=settings.umap_min_dist,
            random_state=42,
            verbose=False,
        )
        embedding_2d = reducer_2d.fit_transform(vectors)
    except Exception:
        embedding_2d = embedding_high[:, :2] if embedding_high.shape[1] >= 2 else embedding_high

    # -- HDBSCAN clustering -----------------------------------------------
    try:
        import hdbscan as _hdbscan  # type: ignore[import-untyped]
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=max(2, settings.hdbscan_min_cluster_size),
            min_samples=max(1, settings.hdbscan_min_samples),
        )
        labels = clusterer.fit_predict(embedding_high)
    except ImportError:
        try:
            from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]
            from sklearn.preprocessing import StandardScaler
            scaled = StandardScaler().fit_transform(embedding_high)
            clusterer = DBSCAN(
                eps=0.5,
                min_samples=max(1, settings.hdbscan_min_samples),
            )
            labels = clusterer.fit_predict(scaled)
        except Exception as exc:
            return {
                "session_ids": session_ids,
                "labels": np.zeros(len(session_ids), dtype=int),
                "embedding": embedding_2d,
                "n_clusters": 1,
                "error": f"Clustering failed (install hdbscan for best results): {exc}",
            }
    except Exception as exc:
        return {
            "session_ids": session_ids,
            "labels": np.zeros(len(session_ids), dtype=int),
            "embedding": embedding_2d,
            "n_clusters": 1,
            "error": f"Clustering failed: {exc}",
        }

    n_clusters = int(np.max(labels)) + 1 if (labels >= 0).any() else 0
    return {
        "session_ids": session_ids,
        "labels": labels,
        "embedding": embedding_2d,
        "n_clusters": n_clusters,
        "error": None,
    }


# ---------------------------------------------------------------------------
# HMM analysis
# ---------------------------------------------------------------------------

def retained_events(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
) -> dict[str, list[tuple[float, float, str]]]:
    """The events an HMM fit actually sees, in decode order.

    Events whose behavior is not among *behavior_ids* are dropped, and sessions
    left with nothing are omitted entirely, exactly what
    :func:`_encode_sequences` does.  Decoded state sequences are indexed by
    position in this list, so anything that maps states back onto bout times
    must filter through here rather than re-deriving the rule; otherwise a
    single dropped event shifts every later state by one bout.
    """
    keep = set(behavior_ids)
    out: dict[str, list[tuple[float, float, str]]] = {}
    for sid, events in sequences.items():
        kept = [e for e in events if e[2] in keep]
        if kept:
            out[sid] = kept
    return out


def _encode_sequences(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
) -> dict[str, list[int]]:
    """Map behavior IDs to integer codes per session."""
    bid_code = {bid: i for i, bid in enumerate(behavior_ids)}
    return {
        sid: [bid_code[e[2]] for e in events]
        for sid, events in retained_events(sequences, behavior_ids).items()
    }


def _posterior_entropy(model: Any, observations: list[np.ndarray]) -> float:
    """Total entropy of the per-observation posterior state distribution.

    Used for the ICL criterion.  Zero when every observation is assigned to one
    state with certainty; large when states overlap and the decoding is
    ambiguous.  Returns 0.0 if the posteriors cannot be computed, which makes
    ICL degrade gracefully to BIC rather than failing the whole fit.
    """
    try:
        lengths = [len(o) for o in observations]
        X = np.concatenate(observations).reshape(-1, 1)
        gamma = np.asarray(model.predict_proba(X, lengths), dtype=np.float64)
        gamma = np.clip(gamma, 1e-12, 1.0)
        return float(-np.sum(gamma * np.log(gamma)))
    except Exception:
        return 0.0


def hmm_free_params(n_states: int, n_features: int) -> int:
    """Number of free parameters in a categorical HMM.

    ``K(K-1)`` transition + ``K(F-1)`` emission + ``(K-1)`` initial-state
    probabilities, where K = n_states and F = n_features (number of behaviors).
    Rows are simplex-constrained, hence the -1 in each term.
    """
    return n_states * (n_states - 1) + n_states * (n_features - 1) + (n_states - 1)


def _fit_single_hmm(
    observations: list[np.ndarray],
    n_states: int,
    n_iter: int,
    n_features: int,
    seed: int | None = None,
) -> tuple[Any, float, bool, int]:
    """Fit one CategoricalHMM.

    Returns ``(model, log_likelihood, converged, iters_used)``.

    ``converged`` is computed here rather than read from
    ``model.monitor_.converged``: hmmlearn's property reports ``True`` whenever
    EM exhausts ``n_iter``, so a fit that ran out of iterations is
    indistinguishable from one that reached a stationary point.  Model selection
    compares log-likelihoods across state counts and is only valid if each fit
    is at (or near) its own maximum, so the distinction matters.
    """
    try:
        from hmmlearn import hmm as hmmlearn_hmm  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("hmmlearn is required for HMM analysis. Install with: pip install hmmlearn")

    lengths = [len(o) for o in observations]
    concatenated = np.concatenate(observations).reshape(-1, 1)

    model = hmmlearn_hmm.CategoricalHMM(
        n_components=n_states,
        n_iter=n_iter,
        tol=1e-4,
        verbose=False,
        n_features=n_features,
        random_state=seed,
    )
    try:
        model.fit(concatenated, lengths)
        ll = model.score(concatenated, lengths)
        iters = int(getattr(model.monitor_, "iter", n_iter))
        return model, float(ll), bool(iters < n_iter), iters
    except Exception:
        return model, float("-inf"), False, 0


def fit_hmm(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    settings: MotifSettings,
) -> dict[str, Any]:
    """Fit a categorical HMM to the pooled behavior sequences.

    Returns
    -------
    dict with keys:
      'n_states', 'transition_matrix', 'emission_matrix',
      'state_sequences', 'log_likelihood', 'aic', 'bic',
      'model_selection' (list of {n: int, aic, bic, ll} for auto mode),
      'behavior_ids', 'behavior_names_used',
      'error' (str | None)
    """
    try:
        from hmmlearn import hmm as _  # noqa: F401  (just check import)
    except ImportError:
        return {
            "n_states": 0,
            "error": (
                "hmmlearn is required for HMM analysis.\n"
                "Install with:  pip install hmmlearn"
            ),
        }

    encoded = _encode_sequences(sequences, behavior_ids)
    if not encoded:
        return {"n_states": 0, "error": "No behavior sequences found."}

    observations = [np.array(seq, dtype=int) for seq in encoded.values()]
    session_ids = list(encoded.keys())
    n_features = len(behavior_ids)

    total_obs = sum(len(o) for o in observations)

    model_selection: list[dict[str, Any]] = []

    if settings.hmm_n_states_mode == "auto":
        n_range = range(settings.hmm_n_states_min, settings.hmm_n_states_max + 1)
    else:
        n_range = range(settings.hmm_n_states, settings.hmm_n_states + 1)

    best_model = None
    best_criterion_val = float("inf")
    best_n = settings.hmm_n_states_min
    unconverged: list[int] = []
    seed0 = int(getattr(settings, "hmm_random_seed", 0))

    # 'cv' is a calibration-time criterion; at fit time fall back to ICL, which
    # is the cheap in-sample criterion that behaves most like held-out fit.
    criterion = str(settings.hmm_criterion or "bic").lower()
    if criterion not in {"aic", "bic", "aicc", "icl"}:
        criterion = "icl" if criterion == "cv" else "bic"

    for n_states in n_range:
        # Multiple random restarts: keep best log-likelihood.  Seeds are
        # deterministic so the selected state count reproduces across runs.
        best_ll = float("-inf")
        best_run_model = None
        any_converged = False
        for restart in range(settings.hmm_n_restarts):
            try:
                model, ll, conv, _iters = _fit_single_hmm(
                    observations, n_states, settings.hmm_n_iter, n_features,
                    seed=seed0 + restart,
                )
                any_converged = any_converged or conv
                if ll > best_ll:
                    best_ll = ll
                    best_run_model = model
            except Exception as exc:
                logger.debug("HMM restart %d failed for n=%d: %s", restart, n_states, exc)

        if best_run_model is None or best_ll == float("-inf"):
            continue
        if not any_converged:
            unconverged.append(n_states)

        n_free = hmm_free_params(n_states, n_features)
        aic = -2 * best_ll + 2 * n_free
        bic = -2 * best_ll + n_free * np.log(max(total_obs, 1))
        # AICc: small-sample correction; diverges as n_free approaches N.
        denom = total_obs - n_free - 1
        aicc = aic + (2 * n_free * (n_free + 1) / denom) if denom > 0 else float("inf")
        # ICL = BIC + 2 * entropy of the posterior state assignments.  Penalizes
        # models whose states are not cleanly separable.
        icl = bic + 2.0 * _posterior_entropy(best_run_model, observations)

        model_selection.append({
            "n_states": n_states,
            "log_likelihood": best_ll,
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "icl": icl,
            "n_free_params": n_free,
            "obs_per_param": total_obs / n_free if n_free else float("inf"),
            "converged": any_converged,
        })

        criterion_val = {"aic": aic, "bic": bic, "aicc": aicc, "icl": icl}[criterion]
        if criterion_val < best_criterion_val:
            best_criterion_val = criterion_val
            best_model = best_run_model
            best_n = n_states

    if best_model is None:
        return {
            "n_states": 0,
            "error": "HMM fitting failed for all model sizes. Ensure you have sufficient data.",
            "model_selection": model_selection,
        }

    # Decode state sequences for each session
    lengths = [len(o) for o in observations]
    concatenated = np.concatenate(observations).reshape(-1, 1)
    try:
        _, state_seq_all = best_model.viterbi(concatenated, lengths)
    except Exception:
        try:
            state_seq_all = best_model.predict(concatenated, lengths)
        except Exception as exc:
            return {
                "n_states": best_n,
                "error": f"State decoding failed: {exc}",
                "model_selection": model_selection,
            }

    state_sequences: dict[str, list[int]] = {}
    cursor = 0
    for sid, length in zip(session_ids, lengths):
        state_sequences[sid] = state_seq_all[cursor: cursor + length].tolist()
        cursor += length

    # Expected fractional occupancy, computed here because it needs the fitted
    # model rather than just the decoded path.  Always returned; which of the
    # two occupancy measures is actually reported is a display/analysis choice
    # made by the caller (see MotifSettings.hmm_occupancy_method).
    try:
        gamma = np.asarray(
            best_model.predict_proba(concatenated, lengths), dtype=np.float64
        )
        posterior_occupancy = state_occupancy_from_posteriors(
            gamma, lengths, session_ids, best_n
        )
        # The per-bout posteriors themselves, kept because ranking bouts by
        # P(state | sequence) needs them and the mean has thrown them away.
        state_posteriors = {}
        _cur = 0
        for _sid, _len in zip(session_ids, lengths):
            state_posteriors[_sid] = np.round(
                gamma[_cur: _cur + _len], 6
            ).tolist()
            _cur += _len
    except Exception as exc:  # pragma: no cover - hmmlearn internal failure
        logger.warning("Posterior state occupancy unavailable: %s", exc)
        posterior_occupancy = {}
        state_posteriors = {}

    return {
        "n_states": best_n,
        "transition_matrix": best_model.transmat_.tolist(),
        "emission_matrix": best_model.emissionprob_.tolist(),
        "start_prob": best_model.startprob_.tolist(),
        "state_sequences": state_sequences,
        "posterior_occupancy": posterior_occupancy,
        "state_posteriors": state_posteriors,
        "log_likelihood": float(best_model.score(concatenated, lengths)),
        "aic": next((r["aic"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "bic": next((r["bic"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "aicc": next((r["aicc"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "icl": next((r["icl"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "model_selection": model_selection,
        "criterion_used": criterion,
        "n_observations": total_obs,
        "n_sequences": len(observations),
        "unconverged_state_counts": unconverged,
        "behavior_ids": behavior_ids,
        "error": None,
    }


# ---------------------------------------------------------------------------
# HMM auto-calibration
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# Published guidance on choosing the number of HMM states is genuinely
# unsettled, and most of it was written for *continuous* emissions sampled on a
# fixed grid (animal-movement step/turn series).  Two facts drive the design of
# this routine:
#
#   1. AIC and BIC systematically over-select states for behavioral sequence
#      data -- Pohle, Langrock, van Beest & Schmidt (2017), "Selecting the
#      Number of States in Hidden Markov Models: Pitfalls, Practical Challenges
#      and Pragmatic Solutions", JABES 22:270-293.  Their recommendation is to
#      treat the criteria as one input among several, bound the search by what
#      is biologically interpretable, and inspect the fitted states.  Dupont et
#      al. (2025, Methods Ecol Evol 16:e70025) reach the same conclusion and
#      note that cross-validated likelihood does not clearly beat BIC either.
#
#   2. Our emissions are *discrete behavior labels* and our time axis is the
#      *bout index*, not a fixed sampling grid.  That makes this a Markov chain
#      over a bout sequence, so the gap between bouts is not modeled at all.
#      It is a deliberate simplification (surfaced as a caveat in the report),
#      and it means state counts from the movement-HMM literature do not
#      transfer directly.
#
# So calibration does not pick a "correct" K.  It measures what can be measured
# -- whether EM converged, whether restarts agree, how much data there is per
# parameter, and how each criterion ranks the candidates -- then reports where
# they agree and defaults to the most conservative defensible choice.

CALIB_MIN_OBS_PER_PARAM = 10.0
"""Observations required per free parameter before a state count is considered
estimable.  Ten is the conventional lower bound for multinomial-style models;
the report flags anything below it rather than silently fitting it."""


def _hmm_cv_loglik(
    observations: list[np.ndarray],
    n_states: int,
    n_features: int,
    n_iter: int,
    n_restarts: int,
    seed0: int,
    folds: list[list[int]],
) -> tuple[float, float, int]:
    """Cross-validated held-out log-likelihood per observation.

    Sequences are whole sessions, so every fold holds out complete sessions and
    no animal contributes to both training and test within a fold.

    Returns ``(mean_ll_per_obs, sem_across_folds, n_folds_used)``.
    """
    per_fold: list[float] = []
    for fi, test_idx in enumerate(folds):
        held = set(test_idx)
        test = [observations[i] for i in test_idx]
        train = [o for i, o in enumerate(observations) if i not in held]
        if not train or not test:
            continue
        best_ll = float("-inf")
        best_model = None
        for r in range(n_restarts):
            try:
                m, ll, _c, _i = _fit_single_hmm(
                    train, n_states, n_iter, n_features, seed=seed0 + 1000 * fi + r
                )
            except Exception:
                continue
            if ll > best_ll:
                best_ll, best_model = ll, m
        if best_model is None or best_ll == float("-inf"):
            continue
        try:
            X = np.concatenate(test).reshape(-1, 1)
            score = float(best_model.score(X, [len(o) for o in test]))
            per_fold.append(score / max(len(X), 1))
        except Exception:
            continue
    if not per_fold:
        return float("nan"), float("nan"), 0
    arr = np.asarray(per_fold, dtype=np.float64)
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), sem, len(arr)


def _make_folds(n_seq: int, n_folds: int, seed: int) -> list[list[int]]:
    """Split sequence indices into ``n_folds`` disjoint held-out blocks."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_seq)
    return [list(map(int, c)) for c in np.array_split(order, n_folds) if len(c)]


def calibrate_hmm_settings(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    settings: MotifSettings,
    time_budget_s: float = 240.0,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Measure this dataset and propose HMM settings, with the evidence.

    Four stages, each of which measures rather than assumes:

    Stage 1 -- data adequacy.  Counts observations and sequences, then finds the
    largest state count with at least :data:`CALIB_MIN_OBS_PER_PARAM`
    observations per free parameter.  That caps the search range.

    Stage 2 -- EM iterations.  Fits at the largest feasible state count with a
    generous cap and records how many iterations EM actually needed.  This
    matters more than it sounds: a truncated fit can make the log-likelihood
    *decrease* as states are added, which is impossible at the true optimum and
    silently corrupts every information criterion computed from it.

    Stage 3 -- restarts.  Fits many restarts at a mid-range state count and
    measures how often EM reaches the best optimum found, then sets the restart
    count needed for a ~99% chance of reaching it at that hit rate.

    Stage 4 -- state count.  Computes AIC, AICc, BIC and ICL over the feasible
    range, plus cross-validated held-out log-likelihood when it fits in
    ``time_budget_s``.  The recommendation is the most parsimonious state count
    the criteria support: for CV, the smallest K within one standard error of
    the best (the standard 1-SE rule); otherwise ICL, which penalizes states
    that are not cleanly separable and so over-selects less than BIC.

    Returns a dict with ``proposed`` (settings to apply), ``report`` (text
    lines), ``table`` (per-state-count criteria), ``warnings`` and ``error``.
    """
    import time as _time

    def _tick(msg: str, frac: float) -> None:
        if progress_cb:
            try:
                progress_cb(msg, float(frac))
            except Exception:
                pass

    try:
        from hmmlearn import hmm as _  # noqa: F401
    except ImportError:
        return {"error": "hmmlearn is required for calibration.\nInstall with:  pip install hmmlearn"}

    encoded = _encode_sequences(sequences, behavior_ids)
    if not encoded:
        return {"error": "No behavior sequences found. Run temporal refinement first."}

    observations = [np.array(v, dtype=int) for v in encoded.values() if len(v)]
    n_features = len(behavior_ids)
    n_seq = len(observations)
    total_obs = int(sum(len(o) for o in observations))
    seed0 = int(getattr(settings, "hmm_random_seed", 0))

    report: list[str] = []
    warnings_out: list[str] = []

    # ---- Stage 1: data adequacy -------------------------------------------
    _tick("Measuring data adequacy...", 0.02)
    k_max_feasible = 2
    for k in range(2, 21):
        if total_obs / max(hmm_free_params(k, n_features), 1) >= CALIB_MIN_OBS_PER_PARAM:
            k_max_feasible = k
        else:
            break
    k_min = 2
    k_max = max(k_max_feasible, k_min + 1)

    lens = sorted(len(o) for o in observations)
    report += [
        "STAGE 1 - DATA ADEQUACY",
        "  Behaviors (emission symbols): %d" % n_features,
        "  Sessions (sequences):         %d" % n_seq,
        "  Bouts (observations):         %d" % total_obs,
        "  Sequence length: min=%d  median=%d  max=%d" % (lens[0], int(np.median(lens)), lens[-1]),
        "  Largest state count with >=%.0f observations per free parameter: K=%d"
        % (CALIB_MIN_OBS_PER_PARAM, k_max_feasible),
        "  -> searching K = %d...%d" % (k_min, k_max),
        "",
    ]
    if total_obs < 500:
        warnings_out.append(
            "Only %d bouts across %d sessions. State counts above ~4 are weakly identified "
            "at this sample size; treat any hidden state as a hypothesis to check against "
            "the emission heatmap, not a finding." % (total_obs, n_seq)
        )
    if n_seq < 8:
        warnings_out.append(
            "Only %d sessions. Per-group state-occupancy comparisons will have very low "
            "power and the cross-validation SE will be unstable." % n_seq
        )

    # ---- Stage 2: EM iterations -------------------------------------------
    _tick("Calibrating EM iterations...", 0.08)
    hard_cap = 2000
    probe_iters: list[int] = []
    for k in (k_max, max(k_min, k_max - 2)):
        for r in range(3):
            try:
                _m, ll, conv, iters = _fit_single_hmm(
                    observations, k, hard_cap, n_features, seed=seed0 + r
                )
            except Exception:
                continue
            if ll > float("-inf") and conv:
                probe_iters.append(iters)
    if probe_iters:
        needed = int(max(probe_iters))
        proposed_iter = int(min(hard_cap, max(100, np.ceil(needed * 1.5 / 50.0) * 50)))
        conv_note = "EM reached its own optimum in at most %d iterations" % needed
    else:
        needed = hard_cap
        proposed_iter = hard_cap
        conv_note = "EM did NOT converge within %d iterations at any probed state count" % hard_cap
        warnings_out.append(
            "EM did not converge within %d iterations. The likelihood surface for this "
            "dataset is very flat - the state count should be read as indicative only."
            % hard_cap
        )
    report += [
        "STAGE 2 - EM ITERATIONS",
        "  %s." % conv_note,
        "  Current setting: n_iter=%d" % settings.hmm_n_iter,
        "  -> proposing n_iter=%d (measured requirement plus 50%% headroom)" % proposed_iter,
        "",
    ]
    if settings.hmm_n_iter < needed:
        warnings_out.append(
            "Current n_iter=%d truncates EM before it converges (it needs up to %d). "
            "Truncated fits make AIC/BIC comparisons invalid, because the log-likelihood of "
            "a larger model can come out BELOW a smaller one - which cannot happen at the "
            "true maximum." % (settings.hmm_n_iter, needed)
        )

    # ---- Stage 3: restarts -------------------------------------------------
    _tick("Measuring local optima...", 0.20)
    k_probe = int(np.clip((k_min + k_max) // 2, k_min, k_max))
    lls: list[float] = []
    for r in range(25):
        try:
            _m, ll, _c, _i = _fit_single_hmm(
                observations, k_probe, proposed_iter, n_features, seed=seed0 + 500 + r
            )
        except Exception:
            continue
        if ll > float("-inf"):
            lls.append(ll)
    if lls:
        arr = np.asarray(lls)
        top = float(arr.max())
        hits = int(np.sum(arr >= top - 0.5))  # within 0.5 nat of the best found
        hit_rate = hits / len(arr)
        if hit_rate >= 0.999:
            proposed_restarts = 5
        else:
            proposed_restarts = int(np.clip(
                np.ceil(np.log(0.01) / np.log(max(1e-9, 1.0 - hit_rate))), 5, 50
            ))
        report += [
            "STAGE 3 - RANDOM RESTARTS (local optima)",
            "  Probed K=%d with %d restarts." % (k_probe, len(arr)),
            "  Best log-likelihood %.2f; reached by %d/%d restarts (%.0f%%)."
            % (top, hits, len(arr), 100 * hit_rate),
            "  Spread across restarts: %.2f nats." % (arr.max() - arr.min()),
            "  Current setting: n_restarts=%d" % settings.hmm_n_restarts,
            "  -> proposing n_restarts=%d (>=99%% chance of reaching that optimum at the "
            "measured hit rate)" % proposed_restarts,
            "",
        ]
        if settings.hmm_n_restarts < proposed_restarts:
            warnings_out.append(
                "EM lands on the best optimum only %.0f%% of the time here, so %d restarts can "
                "miss it and report a worse fit for some state counts than for others - which "
                "distorts the criterion curve." % (100 * hit_rate, settings.hmm_n_restarts)
            )
    else:
        proposed_restarts = max(10, settings.hmm_n_restarts)
        report += ["STAGE 3 - RANDOM RESTARTS", "  Probe failed; keeping a conservative default.", ""]

    # ---- Stage 4: state count ---------------------------------------------
    _tick("Scoring state counts...", 0.32)
    k_range = list(range(k_min, k_max + 1))
    table: list[dict[str, Any]] = []
    fit_cost_s = 0.0
    for i, k in enumerate(k_range):
        _tick("Scoring K=%d..." % k, 0.32 + 0.33 * i / max(len(k_range), 1))
        best_ll, best_model, any_conv = float("-inf"), None, False
        t0 = _time.time()
        for r in range(proposed_restarts):
            try:
                m, ll, conv, _i2 = _fit_single_hmm(
                    observations, k, proposed_iter, n_features, seed=seed0 + r
                )
            except Exception:
                continue
            any_conv = any_conv or conv
            if ll > best_ll:
                best_ll, best_model = ll, m
        fit_cost_s += (_time.time() - t0) / max(proposed_restarts, 1)
        if best_model is None or best_ll == float("-inf"):
            continue
        pfree = hmm_free_params(k, n_features)
        aic = -2 * best_ll + 2 * pfree
        bic = -2 * best_ll + pfree * np.log(max(total_obs, 1))
        denom = total_obs - pfree - 1
        aicc = aic + (2 * pfree * (pfree + 1) / denom) if denom > 0 else float("inf")
        icl = bic + 2.0 * _posterior_entropy(best_model, observations)
        table.append({
            "n_states": k, "log_likelihood": best_ll, "n_free_params": pfree,
            "obs_per_param": total_obs / pfree, "aic": aic, "aicc": aicc,
            "bic": bic, "icl": icl, "converged": any_conv,
            "cv_loglik": float("nan"), "cv_sem": float("nan"), "cv_folds": 0,
        })

    if not table:
        return {
            "error": "Every HMM fit failed. There may be too few bouts to model.",
            "report": report,
        }

    lls_by_k = [r["log_likelihood"] for r in table]
    if not all(b >= a - 1e-6 for a, b in zip(lls_by_k, lls_by_k[1:])):
        warnings_out.append(
            "Log-likelihood decreases somewhere as states are added, which is impossible at "
            "the true maximum. Some fits are still stuck in local optima even after "
            "calibration - raise n_restarts further before trusting the criterion curve."
        )

    # Cross-validation, if affordable.
    per_fit = fit_cost_s / max(len(k_range), 1)
    cv_restarts = max(3, min(proposed_restarts, 8))
    n_folds = min(n_seq, 16)
    est = per_fit * cv_restarts * n_folds * len(k_range)
    while n_folds > 3 and est > time_budget_s:
        n_folds = max(3, n_folds // 2)
        est = per_fit * cv_restarts * n_folds * len(k_range)
    cv_ok = n_seq >= 4 and est <= time_budget_s * 1.5
    if cv_ok:
        folds = _make_folds(n_seq, n_folds, seed0 + 77)
        for i, row in enumerate(table):
            _tick("Cross-validating K=%d..." % row["n_states"], 0.65 + 0.33 * i / len(table))
            mean_ll, sem, used = _hmm_cv_loglik(
                observations, int(row["n_states"]), n_features, proposed_iter,
                cv_restarts, seed0, folds,
            )
            row["cv_loglik"], row["cv_sem"], row["cv_folds"] = mean_ll, sem, used

    # ---- pick a recommendation --------------------------------------------
    cv_complete = cv_ok and all(np.isfinite(r["cv_loglik"]) for r in table) and len(table) > 1
    if cv_complete:
        best_i = int(np.argmax([r["cv_loglik"] for r in table]))
        thresh = table[best_i]["cv_loglik"] - (table[best_i]["cv_sem"] or 0.0)
        rec_k = int(next(
            (r["n_states"] for r in table if r["cv_loglik"] >= thresh),
            table[best_i]["n_states"],
        ))
        used_criterion = "cv"
    else:
        rec_k = int(min(table, key=lambda r: r["icl"])["n_states"])
        used_criterion = "icl"

    picks = {
        "AIC": int(min(table, key=lambda r: r["aic"])["n_states"]),
        "AICc": int(min(table, key=lambda r: r["aicc"])["n_states"]),
        "BIC": int(min(table, key=lambda r: r["bic"])["n_states"]),
        "ICL": int(min(table, key=lambda r: r["icl"])["n_states"]),
    }
    if cv_complete:
        picks["CV (1-SE)"] = rec_k

    report += [
        "STAGE 4 - NUMBER OF STATES",
        "  AIC and BIC systematically favor more states than are biologically sensible for",
        "  behavioral sequence data (Pohle, Langrock, van Beest & Schmidt 2017, JABES",
        "  22:270-293; Dupont et al. 2025, Methods Ecol Evol 16:e70025). All four criteria are",
        "  shown so you can see the direction and size of that disagreement rather than",
        "  inherit one criterion's answer.",
        "",
    ]
    hdr = ("   K        logL  params  obs/par        AIC       AICc        BIC        ICL")
    if cv_ok:
        hdr += "   CV logL/obs        +/-SE"
    report += ["  " + hdr, "  " + "-" * len(hdr)]
    for r in table:
        line = ("  %4d  %10.2f  %6d  %7.1f  %9.1f  %9.1f  %9.1f  %9.1f"
                % (r["n_states"], r["log_likelihood"], r["n_free_params"],
                   r["obs_per_param"], r["aic"], r["aicc"], r["bic"], r["icl"]))
        if cv_ok:
            cv, sem = r["cv_loglik"], r["cv_sem"]
            line += ("  %+12.4f  %11.4f" % (cv, sem)) if np.isfinite(cv) else \
                "           n/a          n/a"
        if not r["converged"]:
            line += "  (not converged)"
        report.append(line)
    report += ["", "  Each criterion's preferred K:"]
    for name, k in picks.items():
        report.append("    %10s: K=%d" % (name, k))
    agree = len(set(picks.values())) == 1
    report += [
        "",
        ("  All criteria agree on K=%d, which is the strongest evidence this method can "
         "give for a state count." % rec_k) if agree else
        ("  The criteria disagree. That is the normal case, not a failure, and the "
         "disagreement runs in the expected direction: AIC and BIC sit at higher state "
         "counts than ICL and cross-validation."),
    ]
    if used_criterion == "cv":
        report.append(
            "  -> recommending K=%d by %d-fold session-held-out cross-validation with the "
            "1-SE rule: the smallest state count whose held-out likelihood is within one "
            "standard error of the best. This is the only criterion here that measures "
            "generalization to an unseen animal rather than in-sample fit." % (rec_k, n_folds)
        )
    else:
        report.append(
            "  -> recommending K=%d by ICL (BIC plus twice the posterior-assignment entropy; "
            "Biernacki, Celeux & Govaert 2000). ICL penalizes state counts whose states are "
            "not cleanly separable, so the states it keeps are the ones you can actually read "
            "off the emission heatmap." % rec_k
        )
        if not cv_ok:
            report.append(
                "  (Cross-validation was skipped: projected cost %.0fs exceeds the %.0fs "
                "budget, or there are too few sessions.)" % (est, time_budget_s)
            )
    report.append("")

    # How well-determined is K, really?
    if cv_complete:
        cvs = np.asarray([r["cv_loglik"] for r in table], dtype=np.float64)
        sems = np.asarray([r["cv_sem"] for r in table], dtype=np.float64)
        spread = float(cvs.max() - cvs.min())
        typ_sem = float(np.nanmedian(sems))
        within = int(np.sum(cvs >= cvs.max() - sems[int(np.argmax(cvs))]))
        report += [
            "  How well-determined is K?",
            "    Held-out likelihood spans %.4f nats/bout across the candidates, against a "
            "typical fold-to-fold SE of %.4f." % (spread, typ_sem),
            "    %d of %d candidate state counts sit within one SE of the best."
            % (within, len(table)),
        ]
        if spread <= 2.0 * typ_sem:
            warnings_out.append(
                "The held-out likelihood barely separates the candidate state counts "
                "(spread %.4f nats/bout vs a fold SE of %.4f). K is weakly determined by "
                "this data: prefer the smallest state count you can interpret, and do not "
                "present the state count itself as a result." % (spread, typ_sem)
            )
        turns = sum(
            1 for a, b, c in zip(cvs, cvs[1:], cvs[2:])
            if (b - a) * (c - b) < 0
        )
        if turns > 1:
            report.append(
                "    The curve changes direction %d times across the range. At this fold "
                "count that is fold noise, not structure, do not read the exact optimum "
                "as meaningful." % turns
            )
        report.append("")

    # Runtime the proposed settings imply for a subsequent 'Run HMM'.  per_fit was
    # measured at the proposed iteration cap, so this is a like-for-like estimate.
    est_run_s = per_fit * proposed_restarts
    report += [
        "  Cost of the proposed settings",
        "    'Run HMM' will fit %d restarts at K=%d with up to %d EM iterations, roughly "
        "%.0fs. The mode is set to Manual so it fits only the recommended state count "
        "rather than re-sweeping the whole range."
        % (proposed_restarts, rec_k, proposed_iter, max(est_run_s, 1.0)),
        "",
    ]

    # ---- model caveats -----------------------------------------------------
    self_t = sum(int(o[i] == o[i + 1]) for o in observations for i in range(len(o) - 1))
    n_trans = sum(max(len(o) - 1, 0) for o in observations)
    report += [
        "MODEL CAVEATS (read before reporting these states)",
        "  1. The time axis is the bout index, not the clock. This is a Markov chain over the",
        "     ordered bout sequence, so the gap between bouts is not modeled. Two sessions",
        "     with identical bout orders score identically even if one took twice as long.",
        "     Using gap duration would require a hidden semi-Markov model.",
        "  2. %d/%d (%.0f%%) of consecutive bouts repeat the same behavior. The HMM sees these;"
        % (self_t, n_trans, 100 * self_t / max(n_trans, 1)),
        "     the descriptive transition matrix does not when 'include self transitions' is",
        "     off. The two panels are therefore answering slightly different questions.",
        "  3. Emissions are model predictions, not ground truth. Per-behavior detector errors",
        "     propagate into the state structure; a state can encode a confusable pair of",
        "     behaviors rather than a behavioral mode.",
        "  4. Hidden states are identified only up to permutation. State 0 in one run is not",
        "     State 0 in another unless the seed is fixed, which is why calibration pins one.",
        "",
    ]

    proposed = {
        "hmm_n_states_mode": "manual",
        "hmm_n_states": rec_k,
        "hmm_n_states_min": int(k_min),
        "hmm_n_states_max": int(k_max),
        "hmm_n_iter": int(proposed_iter),
        "hmm_n_restarts": int(proposed_restarts),
        "hmm_criterion": "icl",
        "hmm_random_seed": seed0,
    }
    _tick("Done.", 1.0)
    return {
        "error": None,
        "proposed": proposed,
        "current": {
            "hmm_n_states_mode": settings.hmm_n_states_mode,
            "hmm_n_states": settings.hmm_n_states,
            "hmm_n_states_min": settings.hmm_n_states_min,
            "hmm_n_states_max": settings.hmm_n_states_max,
            "hmm_n_iter": settings.hmm_n_iter,
            "hmm_n_restarts": settings.hmm_n_restarts,
            "hmm_criterion": settings.hmm_criterion,
            "hmm_random_seed": seed0,
        },
        "recommended_n_states": rec_k,
        "criterion_picks": picks,
        "selection_basis": used_criterion,
        "cv_used": bool(cv_ok),
        "cv_folds": int(n_folds) if cv_ok else 0,
        "table": table,
        "report": report,
        "warnings": warnings_out,
        "n_observations": total_obs,
        "n_sequences": n_seq,
    }


def state_occupancy(
    state_sequences: dict[str, list[int]],
    n_states: int,
) -> dict[str, list[float]]:
    """Compute fractional occupancy in each hidden state per session.

    Returns ``{session_id: [frac_state0, frac_state1, ...]}``.
    """
    result: dict[str, list[float]] = {}
    for sid, seq in state_sequences.items():
        if not seq:
            result[sid] = [0.0] * n_states
            continue
        counts = np.bincount(seq, minlength=n_states)
        result[sid] = (counts / len(seq)).tolist()
    return result


def state_occupancy_from_posteriors(
    posteriors: np.ndarray,
    lengths: list[int],
    session_ids: list[str],
    n_states: int,
) -> dict[str, list[float]]:
    """Expected fractional occupancy per session from per-bout posteriors.

    *posteriors* is the ``(n_observations, n_states)`` forward-backward
    posterior P(state | whole sequence) over the concatenated observations, in
    the same order and with the same per-session *lengths* that were passed to
    the fit.  Each session's block is averaged down its rows, so the result is
    the same shape as :func:`state_occupancy` and each row still sums to 1.

    This is the soft counterpart of :func:`state_occupancy`.  Viterbi counts a
    bout for exactly one state; here a bout distributes its weight over every
    state that could have produced it.  The practical consequence is that a
    session containing only one or two bouts of a rare behavior gets a small
    non-zero occupancy in that behavior's state instead of an exact 0.0 - the
    Viterbi path will not pay two transition costs to visit a state for a
    single observation, so it reports zero for animals that did in fact perform
    the behavior.
    """
    out: dict[str, list[float]] = {}
    cursor = 0
    for sid, length in zip(session_ids, lengths):
        block = posteriors[cursor: cursor + length]
        cursor += length
        if length <= 0 or block.size == 0:
            out[sid] = [0.0] * n_states
            continue
        means = np.zeros(n_states, dtype=np.float64)
        k = min(n_states, block.shape[1])
        means[:k] = block[:, :k].mean(axis=0)
        out[sid] = means.tolist()
    return out


def resolve_state_occupancy(
    hmm_result: dict[str, Any],
    method: str = "viterbi",
) -> tuple[dict[str, list[float]], str]:
    """The occupancy table to report, and which measure it actually is.

    Falls back to the Viterbi counts when posterior occupancy was requested but
    the fit could not produce it, so a caller never silently reports one measure
    under the other's name.
    """
    n_states = int(hmm_result.get("n_states", 0) or 0)
    viterbi = state_occupancy(hmm_result.get("state_sequences", {}) or {}, n_states)
    if str(method).lower() != "posterior":
        return viterbi, "viterbi"
    posterior = hmm_result.get("posterior_occupancy") or {}
    if not posterior:
        logger.warning(
            "Posterior occupancy was requested but the fit did not produce it; "
            "reporting Viterbi occupancy instead."
        )
        return viterbi, "viterbi"
    return {sid: list(v) for sid, v in posterior.items()}, "posterior"


# ---------------------------------------------------------------------------
# HMM state bouts and ethogram
# ---------------------------------------------------------------------------
#
# The HMM runs over the *bout index*, not the frame grid: observation t is the
# t-th bout of a session, and the decoded state sequence is one state per bout.
# Nothing downstream of the fit knows when those bouts happened.  The functions
# below put the clock back on: they re-pair each decoded state with the bout it
# came from, then collapse consecutive bouts sharing a state into a *state
# bout* - a contiguous stretch of session during which the animal stayed in one
# latent state.  That object is what the ethogram draws and what the TRACY
# boutframes export writes, so both views describe the same intervals.

def decode_state_spans(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    state_sequences: dict[str, list[int]],
) -> dict[str, list[dict[str, Any]]]:
    """Attach each decoded HMM state to the bout that produced it.

    Parameters
    ----------
    sequences:
        The same ``{session_id: [(start_s, end_s, behavior_id), ...]}`` that was
        passed to :func:`fit_hmm` - including the same overlap filtering, or the
        states will not line up with the bouts.
    behavior_ids:
        The same behavior list passed to :func:`fit_hmm`.
    state_sequences:
        ``fit_hmm(...)["state_sequences"]``.

    Returns
    -------
    ``{session_id: [{'bout_index', 'start_s', 'end_s', 'behavior_id',
    'state'}, ...]}`` ordered by start time.  Sessions whose decoded length
    disagrees with their retained-event count are truncated to the shorter of
    the two and logged, rather than silently mis-aligning states with bouts.
    """
    retained = retained_events(sequences, behavior_ids)
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, events in retained.items():
        states = state_sequences.get(sid)
        if not states:
            continue
        n = len(events)
        if len(states) != n:
            logger.warning(
                "HMM state decoding for session %s has %d states for %d bouts; "
                "using the first %d.",
                sid, len(states), n, min(len(states), n),
            )
            n = min(len(states), n)
        out[sid] = [
            {
                "bout_index": i,
                "start_s": float(events[i][0]),
                "end_s": float(events[i][1]),
                "behavior_id": str(events[i][2]),
                "state": int(states[i]),
            }
            for i in range(n)
        ]
    return out


def merge_state_bouts(
    state_spans: dict[str, list[dict[str, Any]]],
    max_gap_s: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collapse consecutive same-state bouts into state bouts.

    A state bout runs from the start of the first bout in a same-state run to
    the end of the last one, so the silent gaps *between* those bouts fall
    inside the state bout.  That is the intended reading - the state is a
    property of the stretch of session, not of the individual bouts - but it
    means a state bout's duration is not the sum of its member bouts'
    durations.

    Parameters
    ----------
    max_gap_s:
        When set, a gap longer than this between the end of one bout and the
        start of the next breaks the run even if the state is unchanged.  Use it
        when a long silence should not be attributed to the state that happened
        to bracket it.  ``None`` (the default) never breaks on time.

    Returns
    -------
    ``{session_id: [{'state', 'start_s', 'end_s', 'duration_s', 'n_bouts',
    'behavior_ids', 'first_bout_index', 'last_bout_index'}, ...]}``
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, spans in state_spans.items():
        if not spans:
            continue
        merged: list[dict[str, Any]] = []
        for span in sorted(spans, key=lambda s: (s["start_s"], s["end_s"])):
            prev = merged[-1] if merged else None
            gap_ok = (
                max_gap_s is None
                or prev is None
                or (float(span["start_s"]) - prev["end_s"]) <= max_gap_s
            )
            if prev is not None and prev["state"] == int(span["state"]) and gap_ok:
                prev["end_s"] = max(prev["end_s"], float(span["end_s"]))
                prev["n_bouts"] += 1
                prev["behavior_ids"].append(span["behavior_id"])
                prev["last_bout_index"] = int(span["bout_index"])
            else:
                merged.append({
                    "state": int(span["state"]),
                    "start_s": float(span["start_s"]),
                    "end_s": float(span["end_s"]),
                    "n_bouts": 1,
                    "behavior_ids": [span["behavior_id"]],
                    "first_bout_index": int(span["bout_index"]),
                    "last_bout_index": int(span["bout_index"]),
                })
        for m in merged:
            m["duration_s"] = float(m["end_s"] - m["start_s"])
        out[sid] = merged
    return out


def default_state_label(state: int) -> str:
    """Column / sheet name for one HMM state.

    ``-1`` is the unclaimed stretch (no scored behavior, no state), which the
    boutframes export carries as its own column so it can serve as a
    within-subject null condition.
    """
    state = int(state)
    if state < 0:
        return "HMM_NoState"
    return f"HMM_State_{state}"


#: State index used for stretches of session no state bout claims.
UNCLAIMED_STATE = -1


def unclaimed_intervals(
    state_bouts: dict[str, list[dict[str, Any]]],
    session_end_s: dict[str, float] | None = None,
    session_start_s: dict[str, float] | None = None,
    min_duration_s: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    """The stretches of each session that no state bout covers.

    A state bout runs from the first to the last bout of a same-state run, so
    what is left over is: the head of the session before the first scored bout,
    the silences where the decoded state changes, and the tail after the last
    bout.  During those stretches the animal is doing none of the behaviors in
    the fit and the HMM makes no claim at all - which is exactly what makes them
    usable as a within-subject null condition for a signal aligned to the state
    intervals.

    Read them as "no scored behavior and no state", not as "baseline": a
    stretch is unclaimed because nothing was detected in it, so anything the
    behavior models miss lands here too.  Set *min_duration_s* to drop the
    sub-second slivers between adjacent bouts, which are too short to average a
    physiological signal over and would otherwise dominate the interval count.

    Intervals carry ``state = UNCLAIMED_STATE`` (-1) and ``source =
    "unclaimed"``, in the same shape as the state bouts they complement.
    """
    ends = session_end_s or {}
    starts = session_start_s or {}
    out: dict[str, list[dict[str, Any]]] = {}

    for sid, bouts in state_bouts.items():
        covered = sorted(
            ((float(b["start_s"]), float(b["end_s"])) for b in bouts),
            key=lambda p: p[0],
        )
        # Union the covered spans first: with a max_gap_s setting in play two
        # state bouts can overlap, and treating them separately would invent a
        # negative-length gap between them.
        merged: list[list[float]] = []
        for st, en in covered:
            if merged and st <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], en)
            else:
                merged.append([st, en])

        gaps: list[tuple[float, float]] = []
        cursor = float(starts.get(sid, 0.0))
        for st, en in merged:
            if st > cursor:
                gaps.append((cursor, st))
            cursor = max(cursor, en)
        end_s = ends.get(sid)
        if end_s is not None and float(end_s) > cursor:
            gaps.append((cursor, float(end_s)))

        rows: list[dict[str, Any]] = []
        for st, en in gaps:
            dur = en - st
            if dur < float(min_duration_s):
                continue
            rows.append({
                "state": UNCLAIMED_STATE,
                "start_s": st,
                "end_s": en,
                "duration_s": dur,
                "n_bouts": 0,
                "behavior_ids": [],
                "source": "unclaimed",
            })
        out[sid] = rows

    return out


def posterior_state_intervals(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    state_posteriors: dict[str, list[list[float]]],
    n_states: int,
    top_n: int = 5,
    max_gap_s: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per subject and per state, the *top_n* bouts most consistent with it.

    Bouts are ranked by the forward-backward posterior P(state = k | whole
    sequence) and the best *top_n* are kept, whether or not the Viterbi decode
    routed them into k.  Consecutive selected bouts merge into one interval, the
    same way :func:`merge_state_bouts` collapses a run of same-state bouts.

    Why this exists
    ---------------
    Viterbi state bouts can be empty for a subject: the decoded path never
    enters a state whose defining behavior that subject performed only once or
    twice, because visiting a state for a single observation has to pay the
    transition cost twice.  An empty interval set is not a zero, it is a missing
    value - there is nothing to average a photometry signal over - and a subject
    missing from half the states drops out of a within-subject comparison.

    What makes it defensible
    ------------------------
    One rule, applied to every subject.  Every session yields the same number of
    intervals for every state, so a group difference cannot be an artefact of
    which subjects got their real state bouts and which got their near-misses.
    Applying this only to the subjects with no winners would mix two selection
    rules in one column and confound the comparison with the rule.

    What it cannot do
    -----------------
    The output is non-empty by construction, so it no longer answers "did this
    subject enter the state at all" - :func:`decode_state_spans` does that.  A
    subject that never performed the behavior still gets *top_n* intervals, and
    theirs will carry a negligible ``posterior_mean``.  Every interval therefore
    reports ``posterior_mean`` (the mean posterior over its member bouts) and
    ``expected_bouts`` (the session's total posterior mass for that state, i.e.
    how many bouts the model thinks were in it).  Report those, and drop or flag
    subjects whose selection carries no real evidence rather than letting it
    read as time in the state.

    Returns
    -------
    ``{session_id: [interval, ...]}`` sorted by start time.  Unlike
    :func:`merge_state_bouts`, intervals from *different* states may overlap in
    time - each state's selection is made independently - so this is a set of
    per-state interval lists flattened into one list, not a partition of the
    session.
    """
    top_n = int(top_n)
    if top_n <= 0:
        return {}
    retained = retained_events(sequences, behavior_ids)
    out: dict[str, list[dict[str, Any]]] = {}

    for sid, events in retained.items():
        post_raw = state_posteriors.get(sid)
        if not post_raw:
            continue
        post = np.asarray(post_raw, dtype=np.float64)
        n = min(len(events), post.shape[0])
        if len(events) != post.shape[0]:
            logger.warning(
                "Posterior selection for session %s has %d posterior rows for "
                "%d bouts; using the first %d.",
                sid, post.shape[0], len(events), n,
            )
        if n <= 0:
            continue
        post = post[:n]
        events = events[:n]

        intervals: list[dict[str, Any]] = []
        for state in range(min(n_states, post.shape[1])):
            col = post[:, state]
            expected = float(col.sum())
            # Stable sort so equal posteriors resolve to the earlier bout and
            # the selection reproduces across runs.
            order = np.argsort(-col, kind="stable")[: min(top_n, n)]
            picked = sorted(int(i) for i in order)

            run: list[int] = []

            def _flush(run: list[int]) -> None:
                if not run:
                    return
                first, last = run[0], run[-1]
                start_s = float(events[first][0])
                end_s = float(events[last][1])
                intervals.append({
                    "state": int(state),
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": max(0.0, end_s - start_s),
                    "n_bouts": len(run),
                    "behavior_ids": [str(events[i][2]) for i in run],
                    "first_bout_index": first,
                    "last_bout_index": last,
                    "posterior_mean": float(np.mean([col[i] for i in run])),
                    "posterior_min": float(np.min([col[i] for i in run])),
                    "expected_bouts": expected,
                    "source": "posterior_topn",
                })

            for idx in picked:
                if not run:
                    run = [idx]
                    continue
                contiguous = idx == run[-1] + 1
                gap_ok = True
                if max_gap_s is not None and max_gap_s > 0:
                    gap_ok = (
                        float(events[idx][0]) - float(events[run[-1]][1])
                    ) <= max_gap_s
                if contiguous and gap_ok:
                    run.append(idx)
                else:
                    _flush(run)
                    run = [idx]
            _flush(run)

        intervals.sort(key=lambda b: (b["start_s"], b["state"]))
        out[sid] = intervals

    return out


def posterior_selection_summary(
    intervals: dict[str, list[dict[str, Any]]],
    n_states: int,
) -> list[dict[str, Any]]:
    """Per session and state, the evidence behind a top-N posterior selection.

    One row per (session, state) with the number of intervals and bouts chosen,
    the mean posterior across the chosen bouts, and the session's total
    posterior mass for that state.  This is the table to read before trusting a
    per-state value: a row with ``expected_bouts`` near zero means the selection
    found nothing that resembles the state, and its intervals should not be
    reported as time in it.
    """
    rows: list[dict[str, Any]] = []
    for sid in sorted(intervals):
        by_state: dict[int, list[dict[str, Any]]] = {}
        for b in intervals[sid]:
            by_state.setdefault(int(b["state"]), []).append(b)
        for state in range(n_states):
            picked = by_state.get(state, [])
            n_bouts = sum(int(b["n_bouts"]) for b in picked)
            weights = [int(b["n_bouts"]) for b in picked]
            mean_post = (
                float(np.average([b["posterior_mean"] for b in picked], weights=weights))
                if picked else 0.0
            )
            rows.append({
                "session_id": sid,
                "state": state,
                "n_intervals": len(picked),
                "n_bouts": n_bouts,
                "total_s": round(sum(float(b["duration_s"]) for b in picked), 4),
                "mean_posterior": round(mean_post, 6),
                "expected_bouts": round(
                    float(picked[0]["expected_bouts"]) if picked else 0.0, 4
                ),
            })
    return rows


def state_bouts_to_frames(
    state_bouts: dict[str, list[dict[str, Any]]],
    fps: float,
    n_states: int,
    frame_offsets: dict[str, int] | None = None,
    state_label: Any = None,
    include_unclaimed: bool = False,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Convert state bouts to ``{session: {state_label: [(start, end), ...]}}``.

    Frame numbers are recovered with the *same* fps that built the sequences, so
    the round trip through seconds is exact.  *frame_offsets* adds a per-session
    analysis prechop back on, returning bouts in raw video frame numbering - the
    numbering ABEL's behavior boutframes and ABELposition exports use, and
    therefore the numbering TRACY expects.

    Every state gets a key in every session, empty list included, so the
    exported workbook has the same columns on every sheet.
    """
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive to convert state bouts to frames.")
    label = state_label or default_state_label
    offsets = frame_offsets or {}
    out: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for sid, bouts in state_bouts.items():
        off = int(offsets.get(sid, 0))
        seeded = list(range(n_states))
        if include_unclaimed:
            # Seeded for every session, not only the ones that have gaps, so the
            # workbook keeps the same columns on every sheet.
            seeded.append(UNCLAIMED_STATE)
        by_state: dict[str, list[tuple[int, int]]] = {
            str(label(st)): [] for st in seeded
        }
        for b in bouts:
            key = str(label(int(b["state"])))
            start = max(0, int(round(float(b["start_s"]) * fps)) + off)
            end = max(start, int(round(float(b["end_s"]) * fps)) + off)
            by_state.setdefault(key, []).append((start, end))
        out[sid] = {k: sorted(v) for k, v in by_state.items()}
    return out


def state_frame_matrix(
    state_bouts: dict[str, list[dict[str, Any]]],
    fps: float,
    session_end_s: dict[str, float] | None = None,
    frame_offsets: dict[str, int] | None = None,
    raw_video_frames: bool = False,
    no_state_value: int | None = None,
) -> dict[str, np.ndarray]:
    """Expand state bouts into one state number per frame, per session.

    Returns ``{session_id: array}`` of float, one element per frame, holding
    the state index for frames a state bout covers and ``NaN`` for frames it
    does not - the stretches the HMM makes no claim about, which the ethogram
    draws as the neutral backdrop.  Float, not int, because "no state" has to
    survive into the export as a blank rather than as a state number.

    Frames run start..end *inclusive*, the same convention as the bout frames
    the sequences were built from (``end_frame`` is the bout's last frame), and
    the same rounding as :func:`state_bouts_to_frames`, so a frame here and a
    frame in the boutframes workbook mean the same frame.

    *raw_video_frames* shifts each session by its analysis prechop, so index
    ``i`` is frame ``i`` of that session's video and the pre-assay frames are
    NaN; the default keeps the analysis clock (index 0 = assay start), which is
    the only numbering under which two sessions' columns line up in time.

    Sessions are padded to their recorded length from *session_end_s* when that
    reaches past the last state bout, so a column stops where the recording
    stops rather than where the animal last did something.

    *no_state_value* replaces the NaNs inside the session with a code of its
    own - pass ``n_states`` to give the unmodeled frames the next number after
    the last state.  It fills only frames the session actually covers: the
    pre-assay head in *raw_video_frames* mode stays NaN, because the HMM was
    never shown those frames at all and coding them as observed-but-stateless
    would be a different claim.
    """
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive to convert state bouts to frames.")
    ends = session_end_s or {}
    offsets = frame_offsets or {}
    out: dict[str, np.ndarray] = {}
    for sid, bouts in state_bouts.items():
        off = int(offsets.get(sid, 0)) if raw_video_frames else 0
        n_frames = int(round(float(ends.get(sid, 0.0)) * fps)) + off
        spans: list[tuple[int, int, int]] = []
        for b in bouts:
            start = max(0, int(round(float(b["start_s"]) * fps)) + off)
            end = max(start, int(round(float(b["end_s"]) * fps)) + off)
            spans.append((start, end, int(b["state"])))
            n_frames = max(n_frames, end + 1)
        arr = np.full(max(n_frames, 0), np.nan, dtype=float)
        # Start order, so that where two state bouts touch on a frame the
        # later one wins - the same frame the ethogram paints last.
        for start, end, st in sorted(spans):
            arr[start:end + 1] = float(st)
        if no_state_value is not None:
            head = arr[off:]
            head[np.isnan(head)] = float(no_state_value)
        out[sid] = arr
    return out


def state_bout_rows(
    state_bouts: dict[str, list[dict[str, Any]]],
    fps: float,
    frame_offsets: dict[str, int] | None = None,
    session_labels: dict[str, str] | None = None,
    session_groups: dict[str, str] | None = None,
    behavior_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten state bouts into one row per state bout, for CSV export."""
    offsets = frame_offsets or {}
    labels = session_labels or {}
    groups = session_groups or {}
    bnames = behavior_names or {}
    rows: list[dict[str, Any]] = []
    for sid in sorted(state_bouts):
        off = int(offsets.get(sid, 0))
        for order, b in enumerate(state_bouts[sid]):
            member = [bnames.get(x, x) for x in b.get("behavior_ids", [])]
            label = labels.get(sid, sid)
            rows.append({
                "session_id": sid,
                "session_label": label,
                "group": groups.get(label, ""),
                "state": int(b["state"]),
                "state_bout_index": order,
                "start_s": round(float(b["start_s"]), 4),
                "end_s": round(float(b["end_s"]), 4),
                "duration_s": round(float(b["duration_s"]), 4),
                "start_frame": int(round(float(b["start_s"]) * fps)) + off,
                "end_frame": int(round(float(b["end_s"]) * fps)) + off,
                "n_bouts": int(b["n_bouts"]),
                "behaviors": " > ".join(member),
            })
    return rows


def state_bout_summary(
    state_bouts: dict[str, list[dict[str, Any]]],
    n_states: int,
) -> dict[str, dict[str, Any]]:
    """Per-session, per-state bout count, total time and mean dwell duration.

    ``time_frac`` is each state's share of the *covered* time (the sum of state
    bout durations), not of the whole session: the HMM says nothing about
    stretches with no bouts at all, so charging them to a state would invent
    occupancy the model never claimed.
    """
    out: dict[str, dict[str, Any]] = {}
    for sid, bouts in state_bouts.items():
        counts = np.zeros(n_states, dtype=np.int64)
        totals = np.zeros(n_states, dtype=np.float64)
        for b in bouts:
            st = int(b["state"])
            if 0 <= st < n_states:
                counts[st] += 1
                totals[st] += float(b["duration_s"])
        covered = float(totals.sum())
        out[sid] = {
            "n_state_bouts": counts.tolist(),
            "total_s": totals.tolist(),
            "mean_dwell_s": [
                (float(totals[i]) / int(counts[i])) if counts[i] else 0.0
                for i in range(n_states)
            ],
            "time_frac": (totals / covered).tolist() if covered > 0 else [0.0] * n_states,
            "covered_s": covered,
        }
    return out


def state_bout_transition_counts(
    state_bouts: dict[str, list[dict[str, Any]]],
    n_states: int,
) -> dict[str, np.ndarray]:
    """Per-session state-bout -> state-bout transition counts.

    Distinct from ``fit_hmm``'s ``transition_matrix``, which is the model's
    per-observation transition probability and is dominated by its diagonal
    (states persist across successive bouts).  This counts *switches* between
    state bouts, so its diagonal is zero by construction and the off-diagonal
    reads directly as "how the animal moved between states".
    """
    out: dict[str, np.ndarray] = {}
    for sid, bouts in state_bouts.items():
        mat = np.zeros((n_states, n_states), dtype=np.int64)
        for a, b in zip(bouts, bouts[1:]):
            i, j = int(a["state"]), int(b["state"])
            if 0 <= i < n_states and 0 <= j < n_states:
                mat[i, j] += 1
        out[sid] = mat
    return out


def state_entry_latency(
    state_bouts: dict[str, list[dict[str, Any]]],
    n_states: int,
    min_dwell_s: float = 0.0,
    session_start_s: dict[str, float] | None = None,
    censor_at_s: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Time from assay start until the animal first settles into each state.

    A state bout counts as an entry only when it lasts at least *min_dwell_s*.
    Without that floor the latency is dominated by one-bout flickers: the HMM
    can pass through a state for a single bout on its way somewhere else, and
    the first such flicker would be reported as the moment the animal "entered"
    the state.  The threshold is a parameter and not a fixed default because
    what counts as settled depends on the assay's bout rate.

    Parameters
    ----------
    session_start_s:
        Assay start for each session, in the same clock as the state bouts.
        Defaults to 0.0, which is correct when bouts are prechop-rebased so
        frame 0 is test onset.
    censor_at_s:
        Session end, used to report how long a non-entering animal was actually
        observed.  Latency itself stays NaN for those animals - substituting the
        session length would turn "never entered" into a measured time and bias
        every group mean toward whoever was recorded longest.

    Returns
    -------
    ``{session_id: {'latency_s': [...], 'entered': [...], 'n_entries': [...],
    'first_entry_duration_s': [...], 'observed_s': float}}`` with one entry per
    state; ``latency_s`` is NaN for states never entered.
    """
    starts = session_start_s or {}
    censor = censor_at_s or {}
    out: dict[str, dict[str, Any]] = {}
    for sid, bouts in state_bouts.items():
        t0 = float(starts.get(sid, 0.0))
        latency = [float("nan")] * n_states
        first_dur = [float("nan")] * n_states
        n_entries = [0] * n_states
        for b in sorted(bouts, key=lambda x: float(x["start_s"])):
            st = int(b["state"])
            if not (0 <= st < n_states):
                continue
            if float(b["duration_s"]) < float(min_dwell_s):
                continue
            n_entries[st] += 1
            if n_entries[st] == 1:
                latency[st] = max(0.0, float(b["start_s"]) - t0)
                first_dur[st] = float(b["duration_s"])
        observed = float(censor.get(sid, 0.0)) - t0
        if observed <= 0:
            observed = max(
                (float(b["end_s"]) for b in bouts), default=t0
            ) - t0
        out[sid] = {
            "latency_s": latency,
            "entered": [n > 0 for n in n_entries],
            "n_entries": n_entries,
            "first_entry_duration_s": first_dur,
            "observed_s": max(0.0, observed),
        }
    return out


# ---------------------------------------------------------------------------
# HMM result persistence
# ---------------------------------------------------------------------------
#
# Fitting is the slow step (restarts x state counts x EM iterations), and the
# result is a pure function of the bout sequences plus the fit settings.  Both
# are known at save time, so the fit is cached to disk with a fingerprint of its
# inputs: reopening a project restores the last fit instead of re-running it,
# and a fingerprint mismatch marks the restored fit stale rather than silently
# showing a figure that no longer matches the data on screen.

HMM_RESULT_FILENAME = "hmm_result.json"

#: Settings that change the numbers a saved result carries.  A change to any of
#: these invalidates a cached result; display-only settings (error bars, bar
#: spacing) do not.  ``hmm_occupancy_method`` is here even though it does not
#: change the fit: it changes the occupancy table and the permutation p-values
#: computed from it, so a result saved under one measure must not be redisplayed
#: as though it were the other.
HMM_FIT_SETTING_KEYS = (
    "hmm_n_states_mode",
    "hmm_n_states",
    "hmm_n_states_min",
    "hmm_n_states_max",
    "hmm_n_iter",
    "hmm_n_restarts",
    "hmm_criterion",
    "hmm_random_seed",
    "hmm_occupancy_method",
    "bout_overlap_tolerance_s",
    "hmm_state_bout_source",
    "hmm_state_bout_top_n",
    "hmm_state_bout_max_gap_s",
    "hmm_state_min_dwell_s",
    "n_permutations",
    "permutation_seed",
)


def hmm_result_path(project_root: Path) -> Path:
    """Where a project's cached HMM fit lives."""
    return Path(project_root) / "derived" / "motif_hmm" / HMM_RESULT_FILENAME


def hmm_input_fingerprint(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
    settings: MotifSettings,
    session_groups: dict[str, str] | None = None,
) -> str:
    """Hash of everything a fit depends on.

    Covers the behaviors in the fit, the sessions and their bout structure, the
    fit settings, and the group assignment (which the permutation tests use).
    Bout *times* are included at 3 decimals, so re-running temporal refinement
    and getting different bouts invalidates the cache even when the counts
    happen to match.
    """
    parts: list[str] = [
        "b:" + "|".join(sorted(str(b) for b in behavior_ids)),
    ]
    for sid in sorted(sequences):
        evs = sequences[sid]
        digest = hashlib.sha1(
            "".join(f"{s:.3f},{e:.3f},{b};" for s, e, b in evs).encode("utf-8")
        ).hexdigest()[:16]
        parts.append(f"s:{sid}:{len(evs)}:{digest}")
    for key in HMM_FIT_SETTING_KEYS:
        parts.append(f"c:{key}={getattr(settings, key, None)!r}")
    for label, group in sorted((session_groups or {}).items()):
        parts.append(f"g:{label}={group}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def save_hmm_result(project_root: Path, result: dict[str, Any]) -> Path | None:
    """Persist a completed fit.  Returns the path, or None if it could not be written.

    A failure here must never break the analysis the user just ran, so the
    error is logged and swallowed, the result stays live in memory either way.
    """
    path = hmm_result_path(project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "result": _json_safe(result),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        return path
    except Exception:
        logger.warning("Could not save the HMM result to %s.", path, exc_info=True)
        return None


def load_hmm_result(project_root: Path) -> dict[str, Any]:
    """Restore the cached fit, or ``{}`` when there is none or it is unreadable."""
    path = hmm_result_path(project_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read the cached HMM result at %s.", path)
        return {}
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != 1:
        return {}
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("n_states"):
        return {}
    result["restored_from_cache"] = True
    result["saved_at"] = payload.get("saved_at", "")
    return result


def clear_hmm_result(project_root: Path) -> None:
    """Delete the cached fit (used when the project's behaviors change)."""
    try:
        hmm_result_path(project_root).unlink(missing_ok=True)
    except Exception:
        logger.warning("Could not delete the cached HMM result.")


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays and tuple keys into JSON-representable data.

    ``fit_hmm`` and the state-bout layer already return lists, but the UI adds
    numpy values (occupancy means, p-values) before the result is saved, and
    ``json`` refuses those outright.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    # NaN and inf are left alone deliberately.  Python's json writes them as the
    # non-standard ``NaN``/``Infinity`` tokens and reads them straight back, so
    # the round trip is exact; mapping them to null instead would hand ``None``
    # to the ``:.2f`` formatting in the stats panel and crash every restored
    # fit whose AIC/BIC was undefined.  This file is only ever read back by us.
    return value
