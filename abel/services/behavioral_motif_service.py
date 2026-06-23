"""Behavioral Motif Analysis service.

Provides three layers of sequential-behavior analysis, all operating on
the bout sequences produced by the temporal refinement step:

1. **Behavior Transition Matrix** — directed transition probabilities or
   counts between successive bouts within a user-defined gap threshold.

2. **Behavioral Motif Discovery** — recurring behavior sub-sequences
   discovered via N-gram frequency analysis or session-level sequence
   clustering (UMAP + HDBSCAN).

3. **Hidden Markov Model Analysis** — latent state discovery via a
   categorical HMM fitted to the pooled behavior sequences, with AIC/BIC
   guided model selection.

All heavy computation is performed in this service layer; the UI widget
dispatches calls via QThreadPool / TaskWorker and never blocks the event loop.

Optional dependencies (graceful degradation if absent):
  hmmlearn  — HMM analysis
  umap-learn / umap — sequence clustering UMAP reduction
  hdbscan   — sequence clustering density-based clustering
  openpyxl  — Excel export (falls back to CSV-only)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
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
    restore the old strict behaviour (B must start after A ends)."""
    normalize_rows: bool = True
    """When True display row-normalised probabilities; when False show raw counts."""
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
    """'aic' | 'bic' — information criterion used for automatic model selection."""

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
        transition A→B.  Set to 0 for the strict (legacy) behaviour where B
        must start strictly after A ends.

    Returns
    -------
    ``{session_id: n x n float64 array}`` — raw counts, NOT normalised.
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
    """Row-normalise a count matrix to obtain transition probabilities."""
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

    return null_counts / max(n_permutations, 1)


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

def _encode_sequences(
    sequences: dict[str, list[tuple[float, float, str]]],
    behavior_ids: list[str],
) -> dict[str, list[int]]:
    """Map behavior IDs to integer codes per session."""
    bid_code = {bid: i for i, bid in enumerate(behavior_ids)}
    encoded: dict[str, list[int]] = {}
    for sid, events in sequences.items():
        coded = [bid_code[e[2]] for e in events if e[2] in bid_code]
        if coded:
            encoded[sid] = coded
    return encoded


def _fit_single_hmm(
    observations: list[np.ndarray],
    n_states: int,
    n_iter: int,
    n_features: int,
) -> tuple[Any, float]:
    """Fit one CategoricalHMM; return (model, log_likelihood)."""
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
    )
    try:
        model.fit(concatenated, lengths)
        ll = model.score(concatenated, lengths)
        return model, float(ll)
    except Exception:
        return model, float("-inf")


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
    n_params_per_state = n_features - 1 + (settings.hmm_n_states_max - 1) + (n_features - 1)

    model_selection: list[dict[str, Any]] = []

    if settings.hmm_n_states_mode == "auto":
        n_range = range(settings.hmm_n_states_min, settings.hmm_n_states_max + 1)
    else:
        n_range = range(settings.hmm_n_states, settings.hmm_n_states + 1)

    best_model = None
    best_criterion_val = float("inf")
    best_n = settings.hmm_n_states_min

    for n_states in n_range:
        # Multiple random restarts — keep best log-likelihood
        best_ll = float("-inf")
        best_run_model = None
        for restart in range(settings.hmm_n_restarts):
            try:
                model, ll = _fit_single_hmm(observations, n_states, settings.hmm_n_iter, n_features)
                if ll > best_ll:
                    best_ll = ll
                    best_run_model = model
            except Exception as exc:
                logger.debug("HMM restart %d failed for n=%d: %s", restart, n_states, exc)

        if best_run_model is None or best_ll == float("-inf"):
            continue

        # AIC / BIC
        n_free = n_states * (n_states - 1) + n_states * (n_features - 1) + (n_states - 1)
        aic = -2 * best_ll + 2 * n_free
        bic = -2 * best_ll + n_free * np.log(max(total_obs, 1))

        model_selection.append({
            "n_states": n_states,
            "log_likelihood": best_ll,
            "aic": aic,
            "bic": bic,
            "n_free_params": n_free,
        })

        criterion_val = aic if settings.hmm_criterion == "aic" else bic
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

    return {
        "n_states": best_n,
        "transition_matrix": best_model.transmat_.tolist(),
        "emission_matrix": best_model.emissionprob_.tolist(),
        "start_prob": best_model.startprob_.tolist(),
        "state_sequences": state_sequences,
        "log_likelihood": float(best_model.score(concatenated, lengths)),
        "aic": next((r["aic"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "bic": next((r["bic"] for r in model_selection if r["n_states"] == best_n), float("nan")),
        "model_selection": model_selection,
        "behavior_ids": behavior_ids,
        "error": None,
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
