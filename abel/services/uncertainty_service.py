"""Uncertainty scoring for active-learning query selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UncertaintyWeights:
    entropy: float = 0.4
    ensemble_variance: float = 0.4
    density_outlier: float = 0.2
    margin: float = 0.0


class UncertaintyScoringService:
    """Compute margin, entropy, disagreement, and density uncertainty metrics."""

    # Maximum reference-set size for the density kNN index.  Building the index
    # on a random subsample instead of all n points avoids O(n²) scaling for
    # large segment tables while preserving the relative outlier ranking.
    _DENSITY_SUBSAMPLE: int = 5_000

    # Columns per pass of the non-finite repair.  Bounds that step's temporary
    # arrays to roughly (n_rows x this) instead of the full feature matrix.
    _REPAIR_BLOCK_COLS: int = 64

    def __init__(self) -> None:
        # Cache keyed by a lightweight feature-matrix fingerprint so that
        # repeated calls with the same features (e.g. multi-behavior passes
        # within one session) skip the expensive kNN computation.
        self._density_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_fingerprint(features: np.ndarray) -> str:
        """Fast, collision-resistant fingerprint for a 2-D float array."""
        stride = max(1, features.size // 8192)
        sample = features.ravel()[::stride]
        digest = hashlib.md5(sample.tobytes()).hexdigest()  # noqa: S324, not crypto
        return f"{features.shape}:{features.dtype}:{digest}"

    # ------------------------------------------------------------------
    # Public static metrics
    # ------------------------------------------------------------------

    @staticmethod
    def classification_margin(prob: np.ndarray) -> np.ndarray:
        if prob.ndim == 1:
            prob = np.column_stack([1.0 - prob, prob])
        top2 = np.sort(prob, axis=1)[:, -2:]
        return np.abs(top2[:, 1] - top2[:, 0])

    @staticmethod
    def entropy(prob: np.ndarray) -> np.ndarray:
        p = np.clip(prob, 1e-9, 1.0)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def ensemble_variance(ensemble_probs: list[np.ndarray]) -> np.ndarray:
        if not ensemble_probs:
            raise ValueError("ensemble_probs cannot be empty")
        stack = np.stack(ensemble_probs, axis=0)
        return np.mean(np.var(stack, axis=0), axis=1)

    @staticmethod
    def density_outlier_score(
        features: np.ndarray,
        k: int = 10,
        subsample: int = _DENSITY_SUBSAMPLE,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Return mean k-NN distance as a density-outlier score for each row.

        Performance notes
        -----------------
        * ``subsample`` caps the kNN *reference* index at that many points
          (random sample, seed 42).  All ``n`` points are still *queried*
          against the smaller index, dropping complexity from O(n²) to
          O(n · k · log(subsample)) for tree-based lookups.
        * ``n_jobs=-1`` lets sklearn use all available CPU cores.
        * When ``n <= subsample`` the full dataset is used as the reference
          (identical to the original behavior).
        """
        from sklearn.neighbors import NearestNeighbors

        # float32 halves the footprint of the biggest array in the pipeline, and
        # matches the dtype segment features are already stored in -- the caller
        # upcasting to float64 bought nothing but bytes.
        features = np.asarray(features)
        owns_copy = features.dtype != np.float32
        if owns_copy:
            features = features.astype(np.float32)

        # NearestNeighbors rejects NaN/Inf (e.g. from z-scoring constant columns).
        # Replace non-finite values with column means so density scoring still runs.
        #
        # Done in column blocks, in place.  The whole-array form of this repair
        # allocated four full-size temporaries at once (the isfinite masks, the
        # np.where result, and nanmean's internal copy); on a 628k x 1612 matrix
        # that is ~30 GiB of peak demand and it ran the process out of memory.
        # Blocking caps the temporaries at _REPAIR_BLOCK_COLS columns apiece.
        for start in range(0, features.shape[1], UncertaintyScoringService._REPAIR_BLOCK_COLS):
            stop = start + UncertaintyScoringService._REPAIR_BLOCK_COLS
            block = features[:, start:stop]
            bad = ~np.isfinite(block)
            if not bad.any():
                continue
            if not owns_copy:
                # Copy once, and only if there is something to repair: this
                # matrix is the caller's model input and must not come back
                # carrying our substituted values.
                features = features.copy()
                owns_copy = True
                block = features[:, start:stop]
            finite = ~bad
            # float64 accumulation: a float32 sum over hundreds of thousands of
            # rows loses real precision in the mean.
            counts = finite.sum(axis=0)
            sums = np.where(finite, block, np.float32(0.0)).sum(axis=0, dtype=np.float64)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            block[bad] = np.take(means, np.nonzero(bad)[1])

        n = len(features)

        # Build the reference index on a random subsample when the dataset is
        # large enough that a full O(n²) search would be prohibitively slow.
        if subsample > 0 and n > subsample:
            rng = np.random.default_rng(42)
            ref_idx = rng.choice(n, size=subsample, replace=False)
            ref = features[ref_idx]
        else:
            ref = features

        k_actual = min(k, len(ref))
        # Brute force, explicitly.  The default "auto" picks a KD-tree for
        # euclidean metrics, and KDTree.query upcasts the *query* array to
        # float64 -- which would silently rebuild the multi-GiB copy the
        # float32 work above exists to avoid.  Brute force is exact (identical
        # neighbors, not an approximation) and is the faster choice anyway at
        # this dimensionality, where trees degenerate to a full scan; sklearn
        # chunks its distance computation, so its working set stays bounded.
        nn = NearestNeighbors(n_neighbors=k_actual, algorithm="brute", n_jobs=n_jobs)
        nn.fit(ref)
        dists, _ = nn.kneighbors(features)
        return np.mean(dists, axis=1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_segments(
        self,
        segment_df: pd.DataFrame,
        class_probs: np.ndarray,
        ensemble_probs: list[np.ndarray],
        feature_cols: list[str],
        weights: UncertaintyWeights | None = None,
        feature_matrix: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Score segment uncertainty.

        ``feature_matrix`` is the model-aligned feature matrix, when the caller
        has already built one.  Pass it whenever the model's feature names do not
        all exist as columns on ``segment_df``, e.g. a model that spells
        symmetric distances under the opposite keypoint ordering.  Indexing
        ``segment_df[feature_cols]`` in that case would raise; materialising the
        missing columns onto ``segment_df`` to avoid that is worse still, because
        the caller writes the frame back to segment_features.parquet, which would
        bake the placeholder values permanently into the feature store.
        """
        w = weights or UncertaintyWeights()

        entropy = self.entropy(class_probs)
        margin = self.classification_margin(class_probs)
        variance = self.ensemble_variance(ensemble_probs)

        # Density is the most expensive metric (kNN over all segments).
        # Skip it entirely when its weight is zero, and use a per-instance
        # cache to avoid redundant computation across multi-behavior passes.
        if w.density_outlier > 0.0:
            feat_matrix = (
                feature_matrix if feature_matrix is not None
                else segment_df[feature_cols].to_numpy(dtype=float)
            )
            fp = self._feature_fingerprint(feat_matrix)
            if fp not in self._density_cache:
                self._density_cache[fp] = self.density_outlier_score(feat_matrix)
            density = self._density_cache[fp]
        else:
            density = np.zeros(len(segment_df), dtype=float)

        # Convert margin to uncertainty by inverting confidence margin.
        margin_unc = 1.0 - np.clip(margin, 0.0, 1.0)

        raw = (
            w.entropy * entropy
            + w.ensemble_variance * variance
            + w.density_outlier * density
            + w.margin * margin_unc
        )
        scaled = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        out = segment_df.copy()
        out["uncertainty_entropy"] = entropy
        out["uncertainty_margin"] = margin_unc
        out["prediction_variance"] = variance
        out["density_outlier_score"] = density
        out["uncertainty_score"] = scaled
        return out
