"""Motif discovery service — unsupervised clustering of pose-feature windows.

Supports two algorithm modes:
- *K-Means* (scikit-learn, Tier-2 required)
- *K-Means + UMAP* — UMAP dimensionality reduction then K-Means clustering
- *HDBSCAN + UMAP* — requires umap-learn + HDBSCAN (sklearn >= 1.3 or hdbscan package)

Graceful degradation:
    scikit-learn missing  → returns early with a clear warning
    umap-learn missing    → falls back to raw (scaled) features; logs warning
    hdbscan missing       → falls back to K-Means with a warning

Pipeline position:
    Pose Features → **Motif Discovery** ← here
    → Candidate Generation → Clip Extraction
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from abel.models.schemas import (
    MotifAssignment,
    MotifDiscoveryPreset,
    MotifModel,
    SeedExample,
)
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml

logger = logging.getLogger("abel")

# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

DEFAULT_PRESETS: list[MotifDiscoveryPreset] = [
    MotifDiscoveryPreset(
        preset_id="kmeans_10",
        name="Quick K-Means (10 clusters)",
        algorithm="kmeans",
        n_clusters=10,
        use_umap=False,
    ),
    MotifDiscoveryPreset(
        preset_id="kmeans_20",
        name="Detailed K-Means (20 clusters)",
        algorithm="kmeans",
        n_clusters=20,
        use_umap=False,
    ),
    MotifDiscoveryPreset(
        preset_id="umap_kmeans_15",
        name="UMAP + K-Means (15 clusters)",
        algorithm="kmeans",
        n_clusters=15,
        use_umap=True,
        umap_n_components=10,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    ),
    MotifDiscoveryPreset(
        preset_id="umap_hdbscan",
        name="UMAP + HDBSCAN (auto clusters)",
        algorithm="hdbscan",
        use_umap=True,
        umap_n_components=10,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        hdbscan_min_cluster_size=50,
        hdbscan_min_samples=5,
    ),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MotifDiscoveryResult:
    """Outcome of one motif discovery run."""
    session_ids: list[str]
    n_windows_total: int = 0       # windows available in the full feature matrices
    n_windows_clustered: int = 0   # windows actually used for clustering (seed-filtered if applicable)
    n_windows_assigned: int = 0
    n_motifs: int = 0
    noise_count: int = 0
    seed_filtered: bool = False
    behavior_id: str | None = None
    model: MotifModel | None = None
    assignments: list[MotifAssignment] = field(default_factory=list)
    cluster_summary: list[dict] = field(default_factory=list)
    success: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MotifDiscoveryService:
    """Loads pose-feature matrices, clusters windows into motifs, saves results.

    No video is decoded.  Reads .npz files produced by PoseFeaturesService.
    """

    def __init__(self) -> None:
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    # ------------------------------------------------------------------
    # Preset management
    # ------------------------------------------------------------------

    @property
    def default_presets(self) -> list[MotifDiscoveryPreset]:
        return list(DEFAULT_PRESETS)

    def load_project_presets(self) -> list[MotifDiscoveryPreset]:
        if not self._project_root:
            return list(DEFAULT_PRESETS)
        path = self._project_root / "config" / "motif_settings.yaml"
        raw = read_yaml(path, {})
        custom: list[MotifDiscoveryPreset] = []
        for item in raw.get("presets", []):
            try:
                custom.append(MotifDiscoveryPreset.model_validate(item))
            except Exception:
                pass
        custom_ids = {p.preset_id for p in custom}
        merged = list(custom)
        for p in DEFAULT_PRESETS:
            if p.preset_id not in custom_ids:
                merged.append(p)
        return merged

    def save_project_preset(self, preset: MotifDiscoveryPreset) -> None:
        if not self._project_root:
            return
        path = self._project_root / "config" / "motif_settings.yaml"
        raw = read_yaml(path, {})
        presets = [p for p in raw.get("presets", []) if p.get("preset_id") != preset.preset_id]
        presets.append(preset.model_dump(mode="json"))
        write_yaml(path, {**raw, "presets": presets})

    # ------------------------------------------------------------------
    # Public discovery entry point
    # ------------------------------------------------------------------

    def run_discovery(
        self,
        session_ids: list[str],
        preset: MotifDiscoveryPreset,
        seeds: list[SeedExample] | None = None,
        behavior_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> MotifDiscoveryResult:
        """Run the full motif-discovery pipeline for the given sessions.

        When *seeds* are provided the cluster model is trained on seed-overlapping
        windows only (to focus on the behaviour's kinematic signature), but motif
        labels are then assigned to every window in the full recording so that
        candidate generation can search across the entire dataset.

        Steps (each increments progress 0→6):
          0. Load & stack feature matrices (full session)
          1. Filter to seed windows for clustering (skipped when seeds=None)
          2. Standardise (z-score) — fit on seeds, transform all
          3. Optional UMAP — fit on seeds, transform all
          4. Cluster (fit on seeds)
          5. Assign labels to all windows; build assignments
          6. Done
        """
        result = MotifDiscoveryResult(
            session_ids=list(session_ids),
            behavior_id=behavior_id,
        )
        _prog = progress_callback or (lambda a, b: None)

        # ── Step 0: Load ALL features ───────────────────────────────
        _prog(0, 6)
        X_all, provenance_all, load_warnings = self._load_session_features(session_ids)
        result.warnings.extend(load_warnings)
        if X_all is None or X_all.shape[0] == 0:
            result.warnings.append(
                "No feature windows found for the selected sessions.  "
                "Run Pose Feature extraction first."
            )
            return result
        result.n_windows_total = X_all.shape[0]
        logger.info("Motif discovery: loaded %d windows from %d session(s).", X_all.shape[0], len(session_ids))

        if cancel_flag and cancel_flag[0]:
            return result

        # ── Step 1: Filter to seed windows for clustering ────────────
        _prog(1, 6)
        # X_cluster/provenance_cluster = windows used to FIT the model (seeds only)
        # X_all/provenance_all         = windows that will be ASSIGNED labels (full session)
        X_cluster, provenance_cluster = X_all, provenance_all
        if seeds:
            X_cluster, provenance_cluster, filter_warnings = self._filter_to_seeds(
                X_all, provenance_all, seeds, behavior_id
            )
            result.warnings.extend(filter_warnings)
            result.seed_filtered = True
            if X_cluster.shape[0] == 0:
                result.warnings.append(
                    "Seed filter removed all windows — aborting.  "
                    "Check that seed frame ranges overlap with the extracted feature windows."
                )
                return result
            logger.info(
                "Seed filter: %d / %d windows kept for clustering.",
                X_cluster.shape[0], result.n_windows_total,
            )
        result.n_windows_clustered = X_cluster.shape[0]

        if cancel_flag and cancel_flag[0]:
            return result

        # ── Step 2: Standardise — fit on cluster windows, transform ALL ──
        _prog(2, 6)
        try:
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
            scaler = StandardScaler().fit(X_cluster.astype(np.float64))
            X_cluster_scaled: np.ndarray = scaler.transform(X_cluster.astype(np.float64))
            X_all_scaled: np.ndarray = scaler.transform(X_all.astype(np.float64))
        except ImportError:
            result.warnings.append(
                "scikit-learn is not installed.  Install it via the Dependencies tab "
                "to enable motif discovery."
            )
            return result
        except Exception as exc:
            result.warnings.append(f"Feature scaling failed: {exc}")
            return result

        if cancel_flag and cancel_flag[0]:
            return result

        # ── Step 3: Optional UMAP — fit on cluster windows, transform ALL ──
        _prog(3, 6)
        X_cluster_embed = X_cluster_scaled
        X_all_embed = X_all_scaled
        use_umap = preset.use_umap or preset.algorithm == "hdbscan"
        if use_umap:
            X_cluster_embed, X_all_embed, umap_warn = self._apply_umap(
                X_cluster_scaled,
                X_all_scaled,
                n_components=preset.umap_n_components,
                n_neighbors=preset.umap_n_neighbors,
                min_dist=preset.umap_min_dist,
                random_state=preset.random_state,
            )
            if umap_warn:
                result.warnings.extend(umap_warn)

        if cancel_flag and cancel_flag[0]:
            return result

        # ── Step 4: Cluster (fit on seed windows) ───────────────────
        _prog(4, 6)
        try:
            if preset.algorithm == "hdbscan":
                labels_cluster, confidences_cluster = self._run_hdbscan(
                    X_cluster_embed,
                    min_cluster_size=preset.hdbscan_min_cluster_size,
                    min_samples=preset.hdbscan_min_samples,
                )
                # HDBSCAN has no native predict(); assign all windows by nearest centroid
                labels_all, confidences_all = self._assign_by_nearest_centroid(
                    X_all_embed, X_cluster_embed, labels_cluster
                )
            else:
                km, labels_cluster, confidences_cluster = self._run_kmeans(
                    X_cluster_embed,
                    n_clusters=preset.n_clusters,
                    random_state=preset.random_state,
                )
                labels_all = km.predict(X_all_embed)
                confidences_all = self._kmeans_confidences(km, X_all_embed, labels_all)
        except Exception as exc:
            result.warnings.append(f"Clustering failed: {exc}")
            return result

        if cancel_flag and cancel_flag[0]:
            return result

        # ── Step 5: Build assignments for ALL windows ────────────────
        _prog(5, 6)
        assignments: list[MotifAssignment] = []
        for i, (sess_id, sf, ef) in enumerate(provenance_all):
            label = int(labels_all[i])
            motif_id = "noise" if label == -1 else f"motif_{label:02d}"
            assignments.append(
                MotifAssignment(
                    assignment_id=uuid.uuid4().hex[:12],
                    session_id=sess_id,
                    start_frame=int(sf),
                    end_frame=int(ef),
                    motif_id=motif_id,
                    confidence=float(np.clip(confidences_all[i], 0.0, 1.0)),
                    behavior_id=behavior_id,
                )
            )

        noise_count = sum(1 for a in assignments if a.motif_id == "noise")
        unique_motifs = sorted({a.motif_id for a in assignments if a.motif_id != "noise"})

        result.assignments = assignments
        result.n_windows_assigned = len(assignments) - noise_count  # across the full session
        result.n_motifs = len(unique_motifs)
        result.noise_count = noise_count

        # Cluster summary
        count_by_motif: Counter[str] = Counter(a.motif_id for a in assignments)
        sessions_by_motif: dict[str, set[str]] = defaultdict(set)
        for a in assignments:
            sessions_by_motif[a.motif_id].add(a.session_id)

        result.cluster_summary = [
            {
                "motif_id": mid,
                "count": count_by_motif[mid],
                "sessions": sorted(sessions_by_motif[mid]),
            }
            for mid in sorted(count_by_motif.keys())
        ]

        # MotifModel
        algo_label = preset.algorithm.upper()
        cluster_hint = (
            f"{preset.n_clusters} clusters" if preset.algorithm == "kmeans"
            else "auto clusters"
        )
        umap_hint = " + UMAP" if use_umap else ""
        seed_hint = (
            f" (seeded: {behavior_id})" if behavior_id
            else (" (seeded)" if result.seed_filtered else "")
        )
        result.model = MotifModel(
            model_id=uuid.uuid4().hex,
            name=(
                f"{algo_label}{umap_hint} {cluster_hint}{seed_hint} — "
                f"{len(session_ids)} session(s), "
                f"{result.n_windows_clustered} windows clustered"
            ),
            algorithm=preset.algorithm,
            parameters={
                "preset": preset.model_dump(mode="json"),
                "behavior_id": behavior_id,
                "seed_filtered": result.seed_filtered,
                "n_windows_total": result.n_windows_total,
                "n_windows_clustered": result.n_windows_clustered,
                "n_windows_assigned": result.n_windows_assigned,
                "n_motifs": result.n_motifs,
                "noise_count": result.noise_count,
                "session_ids": session_ids,
                "cluster_summary": result.cluster_summary,
            },
        )

        _prog(6, 6)
        result.success = True
        logger.info(
            "Motif discovery complete: %d motifs, %d assigned windows, %d noise.",
            result.n_motifs, result.n_windows_assigned, result.noise_count,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, result: MotifDiscoveryResult) -> None:
        """Persist MotifModel and MotifAssignment list to derived/motifs/."""
        if not self._project_root or not result.model:
            return
        out_dir = self._project_root / "derived" / "motifs"
        out_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            out_dir / "motif_model.json",
            result.model.model_dump(mode="json"),
        )
        write_json(
            out_dir / "assignments.json",
            {"assignments": [a.model_dump(mode="json") for a in result.assignments]},
        )
        logger.info("Motif results saved to %s", out_dir)

    def load_model(self) -> MotifModel | None:
        if not self._project_root:
            return None
        path = self._project_root / "derived" / "motifs" / "motif_model.json"
        if not path.exists():
            return None
        try:
            return MotifModel.model_validate(read_json(path, {}))
        except Exception as exc:
            logger.warning("Failed to load motif model: %s", exc)
            return None

    def load_assignments(self) -> list[MotifAssignment]:
        if not self._project_root:
            return []
        path = self._project_root / "derived" / "motifs" / "assignments.json"
        raw = read_json(path, {"assignments": []})
        results = []
        for item in raw.get("assignments", []):
            try:
                results.append(MotifAssignment.model_validate(item))
            except Exception:
                pass
        return results

    def has_results(self) -> bool:
        if not self._project_root:
            return False
        return (self._project_root / "derived" / "motifs" / "motif_model.json").exists()

    def clear_results(self) -> int:
        """Delete persisted motif model + assignments for the current project."""
        if not self._project_root:
            return 0
        out_dir = self._project_root / "derived" / "motifs"
        removed = 0
        for name in ("motif_model.json", "assignments.json"):
            path = out_dir / name
            if path.exists():
                path.unlink(missing_ok=True)
                removed += 1
        logger.info("Cleared motif results (%d file(s)) from %s", removed, out_dir)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_to_seeds(
        self,
        X: np.ndarray,
        provenance: list[tuple[str, int, int]],
        seeds: list[SeedExample],
        behavior_id: str | None = None,
    ) -> tuple[np.ndarray, list[tuple[str, int, int]], list[str]]:
        """Return only the feature windows that overlap with positive seed examples.

        Args:
            behavior_id: if set, only seeds for that behavior are used.

        A feature window [sf, ef) overlaps a seed [seed_sf, seed_ef) when
        sf < seed_ef and ef > seed_sf  (i.e. they share at least one frame).
        """
        warnings_: list[str] = []
        provenance_sessions = {sess_id for sess_id, _, _ in provenance}

        # Build per-session positive seed intervals
        seed_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for seed in seeds:
            if seed.label_type != "positive":
                continue
            if behavior_id and seed.behavior_id != behavior_id:
                continue
            seed_intervals[seed.session_id].append((seed.start_frame, seed.end_frame))

        if not seed_intervals:
            msg = "No positive seed examples found"
            if behavior_id:
                msg += f" for behavior '{behavior_id}'"
            msg += " — using all feature windows."
            warnings_.append(msg)
            return X, provenance, warnings_

        total_seeds = sum(len(v) for v in seed_intervals.values())

        # Keep only seeds for sessions that actually have loaded feature windows.
        seed_intervals = {
            sid: intervals for sid, intervals in seed_intervals.items() if sid in provenance_sessions
        }
        if not seed_intervals:
            warnings_.append(
                "Positive seeds were found, but none belong to the selected sessions' feature files."
            )
            return np.empty((0, X.shape[1]), dtype=X.dtype), [], warnings_

        keep_indices: list[int] = []
        for i, (sess_id, sf, ef) in enumerate(provenance):
            for seed_sf, seed_ef in seed_intervals.get(sess_id, []):
                if sf < seed_ef and ef > seed_sf:  # overlap condition
                    keep_indices.append(i)
                    break

        if not keep_indices:
            win_min = min(sf for _, sf, _ in provenance)
            win_max = max(ef for _, _, ef in provenance)
            seed_min = min(sf for intervals in seed_intervals.values() for sf, _ in intervals)
            seed_max = max(ef for intervals in seed_intervals.values() for _, ef in intervals)
            warnings_.append(
                "No feature windows overlap with seed examples.  "
                "Check that seed frame ranges fall within the sessions' feature data.  "
                f"Seed range: {seed_min}-{seed_max}; feature-window range: {win_min}-{win_max}."
            )
            return np.empty((0, X.shape[1]), dtype=X.dtype), [], warnings_

        idx_arr = np.array(keep_indices)
        filtered_prov = [provenance[i] for i in keep_indices]
        n_kept = len(keep_indices)
        n_total = len(provenance)
        logger.info(
            "Seed filter: %d / %d windows match %d seed example(s).",
            n_kept, n_total, total_seeds,
        )
        return X[idx_arr], filtered_prov, warnings_

    def _load_session_features(
        self,
        session_ids: list[str],
    ) -> tuple[np.ndarray | None, list[tuple[str, int, int]], list[str]]:
        """Load and stack feature matrices for all sessions.

        Returns (X, provenance, warnings) where:
          X          — float32 array (n_windows_total × n_features)
          provenance — list of (session_id, start_frame, end_frame) per row in X
          warnings   — list of warning strings for missing / failed sessions
        """
        if not self._project_root:
            return None, [], ["No project loaded."]

        features_dir = self._project_root / "derived" / "pose_features"
        all_features: list[np.ndarray] = []
        provenance: list[tuple[str, int, int]] = []
        warnings: list[str] = []

        for sid in session_ids:
            npz_path = features_dir / f"{sid}.npz"
            if not npz_path.exists():
                warnings.append(f"[SKIP] {sid}: feature file not found — run Pose Features first.")
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                feats: np.ndarray = data["features"].astype(np.float32)
                wf: np.ndarray = data["window_frames"]
                if feats.ndim != 2 or feats.shape[0] == 0:
                    warnings.append(f"[SKIP] {sid}: feature array is empty or malformed.")
                    continue
                # Replace NaN / inf with 0 (rare edge case from short sessions)
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                all_features.append(feats)
                for row_idx in range(len(feats)):
                    sf = int(wf[row_idx, 0]) if row_idx < len(wf) else 0
                    ef = int(wf[row_idx, 1]) if row_idx < len(wf) else 0
                    provenance.append((sid, sf, ef))
            except Exception as exc:
                warnings.append(f"[SKIP] {sid}: failed to load features — {exc}")

        if not all_features:
            return None, [], warnings

        X = np.vstack(all_features)
        return X, provenance, warnings

    def _apply_umap(
        self,
        X_fit: np.ndarray,
        X_transform: np.ndarray,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Fit UMAP on X_fit, then transform both X_fit and X_transform.

        Returns (X_fit_reduced, X_transform_reduced, warnings).
        When UMAP is unavailable, both inputs are returned unchanged.
        """
        warnings: list[str] = []
        n_fit = X_fit.shape[0]
        # Cap n_components to avoid requesting more dims than samples
        n_components = min(n_components, n_fit - 1, X_fit.shape[1])
        n_components = max(2, n_components)
        n_neighbors = min(n_neighbors, n_fit - 1)
        n_neighbors = max(2, n_neighbors)
        try:
            import umap  # noqa: PLC0415
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                verbose=False,
            )
            X_fit_reduced = np.asarray(reducer.fit_transform(X_fit), dtype=np.float64)
            X_transform_reduced = np.asarray(reducer.transform(X_transform), dtype=np.float64)
            logger.info(
                "UMAP: %d → %d dims; assigned %d total windows.",
                X_fit.shape[1], n_components, X_transform_reduced.shape[0],
            )
            return X_fit_reduced, X_transform_reduced, warnings
        except ImportError:
            warnings.append(
                "umap-learn is not installed — falling back to raw features.  "
                "Install it via the Dependencies tab for better clustering."
            )
            return X_fit, X_transform, warnings
        except Exception as exc:
            warnings.append(f"UMAP failed ({exc}) — falling back to raw features.")
            return X_fit, X_transform, warnings

    def _run_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        random_state: int,
    ) -> tuple:
        """K-Means clustering.  Returns (fitted_km, labels, confidences)."""
        from sklearn.cluster import KMeans  # noqa: PLC0415

        n_clusters = min(n_clusters, X.shape[0])
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels: np.ndarray = km.fit_predict(X)
        confidences = self._kmeans_confidences(km, X, labels)
        return km, labels, confidences

    def _kmeans_confidences(
        self,
        km,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-window confidence as inverse normalised intra-cluster distance."""
        confidences = np.ones(len(labels), dtype=np.float32)
        dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
        n_clusters = km.cluster_centers_.shape[0]
        for k in range(n_clusters):
            mask = labels == k
            if not mask.any():
                continue
            max_d = dists[mask].max()
            confidences[mask] = 1.0 - (dists[mask] / max_d) if max_d > 0 else 1.0
        return confidences

    def _assign_by_nearest_centroid(
        self,
        X_all: np.ndarray,
        X_cluster: np.ndarray,
        labels_cluster: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign each row of X_all to the nearest cluster centroid derived from X_cluster.

        Used for HDBSCAN, which has no native predict() for new data.
        -1 (noise) centroids are excluded; if all cluster members are noise the
        point is assigned noise.

        Returns (labels_all, confidences_all).
        """
        unique_labels = sorted(set(labels_cluster.tolist()))
        non_noise = [l for l in unique_labels if l != -1]

        if not non_noise:
            # All seed windows were noise — assign everything as noise
            return (
                np.full(len(X_all), -1, dtype=np.int64),
                np.zeros(len(X_all), dtype=np.float32),
            )

        # Compute centroid per non-noise cluster
        centroids = np.array(
            [X_cluster[labels_cluster == k].mean(axis=0) for k in non_noise],
            dtype=np.float64,
        )
        centroid_labels = np.array(non_noise, dtype=np.int64)

        # Nearest-centroid assignment
        diffs = X_all[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, K, D)
        dists = np.linalg.norm(diffs, axis=2)  # (N, K)
        nearest = np.argmin(dists, axis=1)
        labels_all = centroid_labels[nearest]

        # Confidence = 1 - normalised distance to assigned centroid
        min_dists = dists[np.arange(len(X_all)), nearest]
        confidences_all = np.ones(len(X_all), dtype=np.float32)
        for ki, k in enumerate(non_noise):
            mask = nearest == ki
            if not mask.any():
                continue
            max_d = min_dists[mask].max()
            confidences_all[mask] = 1.0 - (min_dists[mask] / max_d) if max_d > 0 else 1.0

        return labels_all, confidences_all

    def _run_hdbscan(
        self,
        X: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """HDBSCAN clustering.  Returns (labels, confidences).

        Tries sklearn's built-in HDBSCAN first (sklearn >= 1.3), then falls back
        to the hdbscan package, then falls back to K-Means with a warning.
        """
        min_cluster_size = max(2, min(min_cluster_size, X.shape[0] // 2))
        min_samples = max(1, min_samples)

        # Try sklearn's built-in HDBSCAN (sklearn >= 1.3)
        try:
            import sklearn.cluster as _skc  # noqa: PLC0415
            _HDBSCAN = getattr(_skc, "HDBSCAN", None)
            if _HDBSCAN is None:
                raise ImportError("HDBSCAN not in this sklearn version")
            hdb = _HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels: np.ndarray = hdb.fit_predict(X)
            probabilities: np.ndarray = getattr(hdb, "probabilities_", np.ones(len(labels)))
            return labels, probabilities
        except ImportError:
            pass

        # Try hdbscan package
        try:
            import hdbscan  # type: ignore[import-not-found]  # noqa: PLC0415
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            labels = hdb.fit_predict(X)
            probabilities = getattr(hdb, "probabilities_", np.ones(len(labels)))
            return labels, probabilities
        except ImportError:
            pass

        # Fallback: K-Means (HDBSCAN unavailable)
        logger.warning(
            "HDBSCAN not available (need sklearn >= 1.3 or hdbscan package). "
            "Falling back to K-Means with 10 clusters."
        )
        return self._run_kmeans(X, n_clusters=10, random_state=42)
