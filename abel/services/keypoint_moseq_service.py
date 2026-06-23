"""Syllable discovery service.

Builds frame-level syllable assignments from imported DLC pose trajectories.
The implementation uses temporal pose embeddings and unsupervised clustering
to produce deterministic, reproducible syllable labels for each session.

Pipeline position:
    Data Import → Behavior Definitions → Seed Examples
    → Pose Features → **Syllable Discovery** ← here
    → Syllable Assignment → Behavior Signature → Candidate Retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from abel.models.schemas import SyllableModel
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml

logger = logging.getLogger("abel")


@dataclass
class KeypointMoSeqConfig:
    """Parameters for syllable discovery."""
    session_ids: list[str]
    model_name: str = "default_moseq"
    backend: str = "keypoint_moseq"  # temporal_kmeans | keypoint_moseq
    n_lags: int = 2
    n_syllables: int = 50
    learning_rate: float = 0.0001
    batch_size: int = 32
    max_iterations: int = 1000
    seed: int = 42
    save_path: Path | None = None
    overwrite: bool = True


@dataclass
class KeypointMoSeqResult:
    """Outcome of syllable discovery."""
    session_ids: list[str] = field(default_factory=list)
    model_id: str = ""
    n_syllables: int = 0
    model_path: Path | None = None
    syllable_assignments: dict[str, Path] = field(default_factory=dict)  # session_id -> assignment file
    warnings: list[str] = field(default_factory=list)
    success: bool = False


class KeypointMoSeqService:
    """Manages syllable discovery and assignment persistence."""

    def __init__(self) -> None:
        self._project_root: Path | None = None
        self._imports = ImportService()
        self._pose = PoseProcessingService()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def load_existing_result(self) -> KeypointMoSeqResult | None:
        """Load previously discovered syllable metadata and assignment paths."""
        if not self._project_root:
            return None

        syllables_dir = self._project_root / "derived" / "syllables"
        model_path = syllables_dir / "model_metadata.json"
        assign_path = syllables_dir / "assignments.json"
        if not model_path.exists() or not assign_path.exists():
            return None

        model_meta = read_json(model_path, {})
        assign_meta = read_json(assign_path, {})

        result = KeypointMoSeqResult(
            session_ids=list(model_meta.get("session_ids", [])),
            model_id=str(model_meta.get("model_id", "")),
            n_syllables=int(model_meta.get("n_syllables", 0) or 0),
            success=True,
        )

        assignments: dict[str, Path] = {}
        for sid, rel_path in assign_meta.items():
            path = self._project_root / str(rel_path)
            if path.exists():
                assignments[str(sid)] = path
        result.syllable_assignments = assignments
        if not result.syllable_assignments:
            result.success = False
            result.warnings.append("No existing syllable assignment files were found on disk.")
        return result

    def clear_results(self) -> int:
        """Delete persisted syllable model/assignment artifacts for the current project."""
        if not self._project_root:
            return 0

        syllables_dir = self._project_root / "derived" / "syllables"
        if not syllables_dir.exists():
            return 0

        removed = 0
        for path in syllables_dir.glob("*.npz"):
            path.unlink(missing_ok=True)
            removed += 1
        for name in ("model_metadata.json", "assignments.json"):
            path = syllables_dir / name
            if path.exists():
                path.unlink(missing_ok=True)
                removed += 1

        logger.info("Cleared syllable results (%d file(s)) from %s", removed, syllables_dir)
        return removed

    def run_discovery(
        self,
        config: KeypointMoSeqConfig,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> KeypointMoSeqResult:
        """Run syllable discovery on imported pose sessions.

        Steps (0->8):
          0. Validate dependencies
          1. Load and clean pose trajectories
          2. Build temporal frame embeddings
          3. Fit clustering model
          4. Predict frame-level syllable assignments per session
          5. Save model
          6. Save assignments
          7. Update metadata
          8. Done
        """
        result = KeypointMoSeqResult(session_ids=list(config.session_ids))
        _prog = progress_callback or (lambda _a, _b, _c: None)

        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        backend = (config.backend or "temporal_kmeans").strip().lower()
        if backend not in {"temporal_kmeans", "keypoint_moseq"}:
            result.warnings.append(
                f"Unknown backend '{config.backend}'. Choose temporal_kmeans or keypoint_moseq."
            )
            return result

        try:
            if backend == "keypoint_moseq":
                assignments, discovered_n, model_meta = self._run_keypoint_moseq_backend(
                    config=config,
                    progress_callback=_prog,
                    cancel_flag=cancel_flag,
                )
            else:
                assignments, discovered_n, model_meta = self._run_temporal_kmeans_backend(
                    config=config,
                    progress_callback=_prog,
                    cancel_flag=cancel_flag,
                )
        except Exception as exc:
            result.warnings.append(f"Syllable discovery failed: {exc}")
            logger.exception("Syllable backend failure")
            return result

        if cancel_flag and cancel_flag[0]:
            return result

        result.model_id = config.model_name
        result.n_syllables = int(discovered_n)
        result.syllable_assignments = self._save_assignment_arrays(assignments)

        _prog(5, 8, "Saving syllable model metadata...")
        self._save_model_metadata(
            config=config,
            result=result,
            algorithm=backend,
            model_parameters=model_meta,
        )

        _prog(6, 8, "Saving assignment metadata...")
        self._save_assignments_metadata(result)

        _prog(7, 8, "Finalising...")
        _prog(8, 8, "Syllable discovery complete")

        result.success = True
        logger.info(
            "Syllable discovery complete (%s): %d syllables, %d sessions",
            backend,
            result.n_syllables,
            len(result.syllable_assignments),
        )
        return result

    def _fit_model_with_progress(
        self,
        model: dict,
        data: dict,
        num_iters: int,
        start_iter: int = 0,
        ar_only: bool = False,
        phase_label: str = "Fitting",
        progress_callback: Callable[[int, int, str], None] | None = None,
        step: int = 4,
        total_steps: int = 8,
        cancel_flag: list[bool] | None = None,
    ) -> dict:
        """Run Gibbs sampling with a per-iteration progress callback.

        Replaces ``kpm.fit_model`` for the fitting phases so we can emit a
        live iteration counter to the UI after every sampling step.
        Iteration messages are prefixed with ``↺ `` so the UI handler knows
        to replace the previous iteration line rather than append a new one.
        """
        from keypoint_moseq.fitting import _wrapped_resample, StopResampling, _set_parallel_flag
        from jax_moseq.models import keypoint_slds
        from jax_moseq.utils import device_put_as_scalar

        parallel = _set_parallel_flag(None)
        model = device_put_as_scalar(model)
        resample_func = keypoint_slds.resample_model
        total = num_iters - start_iter  # number of Gibbs steps in this phase

        for iteration in range(start_iter, num_iters + 1):
            if cancel_flag and cancel_flag[0]:
                break
            try:
                model = _wrapped_resample(
                    resample_func, data, model,
                    ar_only=ar_only,
                    verbose=False,
                    jitter=0.001,
                    parallel_message_passing=parallel,
                )
            except StopResampling:
                logger.warning("Keypoint-MoSeq: early stopping at iteration %d (NaNs or user interrupt)", iteration)
                break

            done = iteration - start_iter + 1
            msg = f"↺ {phase_label}: iteration {done}/{total}"
            logger.debug("Keypoint-MoSeq %s iter %d/%d", phase_label, done, total)
            if progress_callback:
                progress_callback(step, total_steps, msg)

        return model

    def _run_temporal_kmeans_backend(
        self,
        config: KeypointMoSeqConfig,
        progress_callback: Callable[[int, int, str], None],
        cancel_flag: list[bool] | None,
    ) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
        progress_callback(0, 8, "Checking dependencies...")
        try:
            import sklearn  # noqa: F401, PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Required dependency missing: {exc}. Install scikit-learn via the Dependencies tab."
            ) from exc

        progress_callback(1, 8, "Loading pose tracks...")
        tracks = self._load_pose_tracks(config.session_ids)
        if not tracks:
            raise RuntimeError(
                "No usable pose trajectories found for selected sessions. "
                "Ensure sessions are imported and pose files are accessible."
            )

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        progress_callback(2, 8, "Building temporal embeddings...")
        X, per_session_embeddings = self._build_temporal_embeddings(
            tracks=tracks,
            n_lags=max(1, int(config.n_lags)),
        )
        if X is None or X.shape[0] == 0:
            raise RuntimeError("Could not build temporal embeddings from pose tracks.")

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        n_clusters = int(np.clip(config.n_syllables, 2, max(2, X.shape[0] // 10 or 2)))
        progress_callback(3, 8, f"Fitting K-Means ({n_clusters} clusters)...")
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(config.seed),
            batch_size=max(256, min(8192, X.shape[0])),
            n_init=10,
            max_iter=max(100, int(config.max_iterations)),
            reassignment_ratio=0.01,
        )
        clusterer.fit(X)

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        progress_callback(4, 8, "Predicting syllable assignments...")
        assignments = self._predict_syllables(
            per_session_embeddings=per_session_embeddings,
            clusterer=clusterer,
        )

        model_meta = {
            "n_lags": int(config.n_lags),
            "n_syllables": int(n_clusters),
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "max_iterations": int(config.max_iterations),
            "seed": int(config.seed),
            "cluster_centers_shape": list(clusterer.cluster_centers_.shape),
        }
        return assignments, int(n_clusters), model_meta

    def _run_keypoint_moseq_backend(
        self,
        config: KeypointMoSeqConfig,
        progress_callback: Callable[[int, int, str], None],
        cancel_flag: list[bool] | None,
    ) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
        # ── Step 0: Apply NumPy 2.0 compatibility shims and import kpm ──────
        progress_callback(0, 8, "Importing Keypoint-MoSeq library...")
        logger.info("Keypoint-MoSeq: importing library and applying NumPy shims")
        import numpy as _np  # noqa: PLC0415
        if not hasattr(_np, "bool8"):
            _np.bool8 = _np.bool_  # type: ignore[attr-defined]
        if not hasattr(_np, "issctype"):
            _np.issctype = lambda rep: issubclass(rep, _np.generic) if isinstance(rep, type) else False  # type: ignore[attr-defined]
        if not hasattr(_np, "row_stack"):
            _np.row_stack = _np.vstack  # type: ignore[attr-defined]
        if not hasattr(_np, "in1d"):
            _np.in1d = _np.isin  # type: ignore[attr-defined]
        try:
            import jax  # noqa: PLC0415
            jax.config.update("jax_enable_x64", True)
            import keypoint_moseq as kpm  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"keypoint_moseq is not installed: {exc}. Install Tier-2 syllables dependencies."
            ) from exc

        # ── Step 1: Load pose arrays ──────────────────────────────────────────
        progress_callback(1, 8, f"Loading pose data for {len(config.session_ids)} session(s)...")
        logger.info("Keypoint-MoSeq: loading pose arrays for %d session(s)", len(config.session_ids))
        coordinates, confidences, body_parts = self._load_pose_arrays_for_moseq(config.session_ids)
        if not coordinates:
            raise RuntimeError(
                "No usable pose trajectories found for selected sessions. "
                "Ensure sessions are imported and pose files are accessible."
            )

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # Infer anterior/posterior body-part names for heading alignment
        ant_idxs, post_idxs = self._infer_body_axis_indices(body_parts)
        anterior_bodyparts = [body_parts[ant_idxs[0]]]
        posterior_bodyparts = [body_parts[post_idxs[0]]]
        latent_dim = int(np.clip((max(2, len(body_parts)) - 1) * 2, 2, 10))

        # Config dict matching the structure kpm functions expect.
        # init_model / _check_init_args require the four nested hyperparam
        # dicts; fit_pca and format_data consume the flat keys via **kwargs.
        nlags = max(1, int(config.n_lags))
        kpm_cfg: dict[str, Any] = {
            # anatomy
            "bodyparts": body_parts,
            "use_bodyparts": body_parts,
            "anterior_bodyparts": anterior_bodyparts,
            "posterior_bodyparts": posterior_bodyparts,
            # integer index arrays required by fit_pca / init_model
            "anterior_idxs": ant_idxs,
            "posterior_idxs": post_idxs,
            # format_data params (flat)
            "conf_pseudocount": 1e-3,
            "added_noise_level": 0.1,
            "max_seg_length": 5000,
            "min_fragment_length": 4,
            # misc init_model kwargs
            "seed": int(config.seed),
            "whiten": True,
            "conf_threshold": 0.5,
            "fix_heading": False,
            # error estimator for noise prior (slope/intercept from default calibration)
            "error_estimator": {"slope": -0.5, "intercept": 0.25},
            # ── nested hyperparam dicts required by init_model ────────────────
            "trans_hypparams": {
                "num_states": int(config.n_syllables),
                "gamma": 1e3,
                "alpha": 5.7,
                "kappa": 1e4,
            },
            "ar_hypparams": {
                "latent_dim": latent_dim,
                "nlags": nlags,
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
            },
            "obs_hypparams": {
                "sigmasq_0": 0.1,
                "sigmasq_C": 0.1,
                "nu_sigma": 1e5,
                "nu_s": 5,
            },
            "cen_hypparams": {
                "sigmasq_loc": 0.5,
            },
        }

        # ── Step 2: Format data ───────────────────────────────────────────────
        progress_callback(2, 8, f"Formatting data ({len(coordinates)} sessions, {len(body_parts)} body parts)...")
        logger.info("Keypoint-MoSeq: formatting data (%d sessions, %d body parts)", len(coordinates), len(body_parts))
        keys = sorted(coordinates.keys())
        data, metadata = kpm.format_data(coordinates, confidences, keys=keys, **kpm_cfg)
        n_frames = int(data["mask"].sum())
        logger.info("Keypoint-MoSeq: data formatted — %d valid frames across %d segment(s)", n_frames, data["Y"].shape[0])

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # ── Step 3: Fit PCA ───────────────────────────────────────────────────
        # fit_pca(Y, mask, ...) wants the packed JAX arrays from format_data,
        # plus anterior_idxs/posterior_idxs as integer index arrays.
        # conf is the augmented confidence array from format_data.
        progress_callback(3, 8, f"Fitting PCA ({n_frames:,} frames, latent_dim={latent_dim}) — may take a moment on first run...")
        logger.info(
            "Keypoint-MoSeq: fitting PCA (latent_dim=%d, %d frames). "
            "First run triggers JAX JIT compilation which may take several minutes.",
            latent_dim, n_frames,
        )
        pca = kpm.fit_pca(data["Y"], data["mask"], conf=data["conf"], **kpm_cfg)
        logger.info("Keypoint-MoSeq: PCA complete")

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # Save directory for kpm checkpoints and results
        assert self._project_root is not None
        save_dir = self._project_root / "derived" / "syllables" / "kpm_fit"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 4: Initialise model ──────────────────────────────────────────
        logger.info("Keypoint-MoSeq: initialising model (num_states=%d, latent_dim=%d, nlags=%d)",
                    config.n_syllables, latent_dim, nlags)
        model = kpm.init_model(data, pca=pca, **kpm_cfg)
        logger.info("Keypoint-MoSeq: model initialised")

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # ── Step 5: Fit AR-HMM (phase 1) ──────────────────────────────────────
        num_ar_iters = max(25, int(config.max_iterations) // 2)
        progress_callback(4, 8, f"AR-HMM phase 1: {num_ar_iters} Gibbs iterations — JIT-compiling on first run (may take 5–15 min before progress appears)...")
        logger.info(
            "Keypoint-MoSeq: AR-HMM phase 1 — %d Gibbs iterations. "
            "This is the slowest step; JIT compilation happens here on first run "
            "and may take 5-15 min before iterations begin.",
            num_ar_iters,
        )
        model_name = str(config.model_name)
        model = self._fit_model_with_progress(
            model, data, num_iters=num_ar_iters, start_iter=0,
            ar_only=True, phase_label="AR-HMM phase 1",
            progress_callback=progress_callback, step=4, total_steps=8,
            cancel_flag=cancel_flag,
        )

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # ── Step 6: Fit full keypoint-SLDS (phase 2) ──────────────────────────
        total_iters = num_ar_iters + max(25, int(config.max_iterations) // 2)
        num_slds_iters = total_iters - num_ar_iters
        progress_callback(5, 8, f"Full keypoint-SLDS phase 2: {num_slds_iters} iterations — faster than phase 1...")
        logger.info(
            "Keypoint-MoSeq: full keypoint-SLDS phase 2 — %d iterations. "
            "Subsequent runs are faster as JAX reuses the compiled kernels.",
            num_slds_iters,
        )
        model = self._fit_model_with_progress(
            model, data, num_iters=total_iters, start_iter=num_ar_iters,
            ar_only=False, phase_label="SLDS phase 2",
            progress_callback=progress_callback, step=5, total_steps=8,
            cancel_flag=cancel_flag,
        )

        if cancel_flag and cancel_flag[0]:
            return {}, 0, {}

        # ── Step 7: Extract results ───────────────────────────────────────────
        progress_callback(6, 8, "✓ Fitting complete — extracting syllable assignments...")
        logger.info("Keypoint-MoSeq: extracting syllable assignments")
        # extract_results writes {save_dir}/{model_name}/results.h5; ensure the dir exists
        # (it won't exist if fit_model was called with save_every_n_iters=None)
        (save_dir / model_name).mkdir(parents=True, exist_ok=True)
        results = kpm.extract_results(model, metadata, str(save_dir), model_name)

        # ── Step 8: Parse syllable assignments ───────────────────────────────
        progress_callback(7, 8, "Parsing results...")
        assignments: dict[str, np.ndarray] = {}
        max_state = -1
        for sid in keys:
            if sid not in results:
                continue
            syll = np.asarray(results[sid].get("syllable", []), dtype=np.int32)
            if syll.size == 0:
                continue
            max_state = max(max_state, int(syll.max()))
            assignments[sid] = syll

        discovered_n = max(1, max_state + 1)
        model_meta = {
            "n_lags": int(config.n_lags),
            "n_syllables": int(discovered_n),
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "max_iterations": int(config.max_iterations),
            "seed": int(config.seed),
            "recordings_used": len(assignments),
            "body_parts": list(body_parts),
            "model_name": model_name,
            "kpm_save_dir": str(save_dir),
        }
        return assignments, int(discovered_n), model_meta

    def _load_pose_tracks(self, session_ids: list[str]) -> dict[str, np.ndarray]:
        """Load cleaned per-frame pose tracks for selected sessions.

        Returns mapping {session_id: frame_features} where frame_features is
        shape (n_frames, n_features_per_frame).

        Frame features are cached in ``derived/pose_cache/`` so that subsequent
        calls (e.g. QC export, UMAP) can succeed even when the original source
        drive is no longer mounted.
        """
        if not self._project_root:
            return {}

        cache_dir = self._project_root / "derived" / "pose_cache"

        manifest = self._imports.load_manifest(self._project_root)

        tracks: dict[str, np.ndarray] = {}
        for sid in session_ids:
            # ── 1. Try cache first ──────────────────────────────────────────
            cache_path = cache_dir / f"{sid}_frame_features.npz"
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    tracks[sid] = data["frame_features"]
                    logger.debug("Loaded cached frame features: %s", sid)
                    continue
                except Exception as exc:
                    logger.warning("Failed to load cached features for %s: %s — will re-load", sid, exc)

            # ── 2. Load from source pose file ──────────────────────────────
            if not manifest:
                logger.warning("No import manifest found; cannot load pose for %s", sid)
                continue
            pose_path = self._imports.pose_path_for_session(manifest, sid)
            if not pose_path:
                logger.warning("No pose file in manifest for session %s", sid)
                continue
            try:
                pose = self._pose.load_and_clean(pose_path, manifest.smoothing_settings if manifest else None)
                feats = self._frame_features_from_pose(pose)
                tracks[sid] = feats
                # ── 3. Save to cache for future offline use ─────────────────
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, frame_features=feats)
                    logger.debug("Cached frame features: %s", sid)
                except Exception as exc:
                    logger.warning("Could not cache frame features for %s: %s", sid, exc)
            except Exception as exc:
                logger.warning("Failed to load/clean pose for %s: %s", sid, exc)
                continue

        return tracks

    def _load_pose_arrays_for_moseq(
        self,
        session_ids: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
        """Load pose arrays in keypoint_moseq format.

        Returns (coordinates, confidences, body_parts) where coordinates map to
        arrays of shape (n_frames, n_parts, 2) and confidences to (n_frames, n_parts).
        """
        if not self._project_root:
            return {}, {}, []

        manifest = self._imports.load_manifest(self._project_root)
        if not manifest:
            return {}, {}, []

        coordinates: dict[str, np.ndarray] = {}
        confidences: dict[str, np.ndarray] = {}
        canonical_parts: list[str] | None = None

        for sid in session_ids:
            pose_path = self._imports.pose_path_for_session(manifest, sid)
            if not pose_path:
                logger.warning("No pose file in manifest for session %s", sid)
                continue
            try:
                pose = self._pose.load_and_clean(pose_path, manifest.smoothing_settings)
            except Exception as exc:
                logger.warning("Failed to load pose for session %s: %s", sid, exc)
                continue

            parts = list(pose.body_parts)
            if canonical_parts is None:
                canonical_parts = parts
            if parts != canonical_parts:
                # Require consistent body-part ordering across sessions.
                continue

            x = pose.x.to_numpy(dtype=np.float64)
            y = pose.y.to_numpy(dtype=np.float64)
            lk = pose.likelihood.to_numpy(dtype=np.float64)
            coords = np.stack([x, y], axis=-1)
            coordinates[sid] = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            confidences[sid] = np.nan_to_num(lk, nan=0.0, posinf=0.0, neginf=0.0)

        return coordinates, confidences, (canonical_parts or [])

    @staticmethod
    def _infer_body_axis_indices(body_parts: list[str]) -> tuple[list[int], list[int]]:
        """Infer anterior/posterior body-part indices for heading initialization."""
        normalized = [bp.lower() for bp in body_parts]

        def _find(candidates: list[str], fallback: int) -> int:
            for cand in candidates:
                for i, bp in enumerate(normalized):
                    if cand in bp:
                        return i
            return fallback

        ant = _find(["nose", "snout", "head", "ear"], fallback=0)
        post = _find(["tailbase", "tail_base", "tail", "hip", "spine"], fallback=max(0, len(body_parts) - 1))

        if ant == post and len(body_parts) > 1:
            post = len(body_parts) - 1 if ant != len(body_parts) - 1 else 0

        return [ant], [post]

    @staticmethod
    def _frame_features_from_pose(pose) -> np.ndarray:
        """Create per-frame pose embeddings from cleaned DLC coordinates."""
        x = pose.x.to_numpy(dtype=np.float32)
        y = pose.y.to_numpy(dtype=np.float32)
        lk = pose.likelihood.to_numpy(dtype=np.float32)

        centroid_x = pose.centroid_x.astype(np.float32)
        centroid_y = pose.centroid_y.astype(np.float32)

        rel_x = x - centroid_x[:, None]
        rel_y = y - centroid_y[:, None]
        radial = np.sqrt((rel_x ** 2) + (rel_y ** 2))
        body_scale = np.median(radial, axis=1)
        body_scale = np.clip(body_scale, 1e-3, None)

        rel_x /= body_scale[:, None]
        rel_y /= body_scale[:, None]

        speed = np.sqrt(
            np.diff(centroid_x, prepend=centroid_x[0]) ** 2
            + np.diff(centroid_y, prepend=centroid_y[0]) ** 2
        )[:, None]

        pose_vec = np.concatenate([rel_x, rel_y], axis=1)
        quality = lk.mean(axis=1, keepdims=True)
        frame_features = np.concatenate([pose_vec, speed, quality], axis=1)
        return np.nan_to_num(frame_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _lag_embed(features: np.ndarray, n_lags: int) -> np.ndarray:
        """Concatenate the previous n_lags frames to inject temporal context."""
        n_frames, feat_dim = features.shape
        n_lags = max(1, int(n_lags))
        out = np.zeros((n_frames, feat_dim * n_lags), dtype=np.float32)
        for lag in range(n_lags):
            src = np.maximum(0, np.arange(n_frames) - lag)
            out[:, lag * feat_dim: (lag + 1) * feat_dim] = features[src]
        return out

    def _build_temporal_embeddings(
        self,
        tracks: dict[str, np.ndarray],
        n_lags: int,
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        per_session: dict[str, np.ndarray] = {}
        stacked: list[np.ndarray] = []

        for sid, feats in tracks.items():
            if feats.shape[0] < 2:
                continue
            emb = self._lag_embed(feats, n_lags=n_lags)
            per_session[sid] = emb
            stacked.append(emb)

        if not stacked:
            return None, {}
        return np.vstack(stacked), per_session

    @staticmethod
    def _predict_syllables(
        per_session_embeddings: dict[str, np.ndarray],
        clusterer: MiniBatchKMeans,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for sid, emb in per_session_embeddings.items():
            labels = clusterer.predict(emb)
            out[sid] = labels.astype(np.int32)
        return out

    def _save_assignment_arrays(self, assignments: dict[str, np.ndarray]) -> dict[str, Path]:
        if not self._project_root:
            return {}

        assignments_dir = self._project_root / "derived" / "syllables"
        assignments_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for sid, syllables in assignments.items():
            out_path = assignments_dir / f"{sid}_syllables.npz"
            np.savez_compressed(out_path, syllables=syllables)
            paths[sid] = out_path
            logger.info("Saved syllable assignments: %s (%d frames)", sid, len(syllables))
        return paths

    def _save_model_metadata(
        self,
        config: KeypointMoSeqConfig,
        result: KeypointMoSeqResult,
        algorithm: str,
        model_parameters: dict[str, Any],
    ) -> None:
        """Persist model metadata."""
        if not self._project_root:
            return

        syllables_dir = self._project_root / "derived" / "syllables"
        syllables_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_id": config.model_name,
            "algorithm": algorithm,
            "n_syllables": int(result.n_syllables),
            "n_lags": config.n_lags,
            "session_ids": sorted(result.syllable_assignments.keys()),
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "parameters": model_parameters,
        }

        write_json(syllables_dir / "model_metadata.json", metadata)
        logger.info("Saved model metadata: %s", config.model_name)

    def _save_assignments_metadata(self, result: KeypointMoSeqResult) -> None:
        """Persist assignment paths and metadata."""
        if not self._project_root:
            return

        syllables_dir = self._project_root / "derived" / "syllables"
        assignments_meta = {
            str(sid): str(path.relative_to(self._project_root))
            for sid, path in result.syllable_assignments.items()
        }
        write_json(syllables_dir / "assignments.json", assignments_meta)

    def load_syllables(self, session_id: str) -> np.ndarray | None:
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

    def build_umap_embeddings(
        self,
        result: "KeypointMoSeqResult",
        progress_callback: Callable[[str], None] | None = None,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        sample_cap: int = 60_000,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 2-D UMAP embedding of the pose-feature space, coloured by syllable.

        Re-loads and re-processes pose tracks (same pipeline as discovery), draws a
        random subsample of up to *sample_cap* frames to keep UMAP tractable, then
        returns ``(xy, labels)`` arrays of shape ``(n_sampled, 2)`` and ``(n_sampled,)``.

        The caller is responsible for saving / rendering the result.

        Raises RuntimeError on missing dependencies or data.
        """
        _log = progress_callback or (lambda msg: None)

        _log("Checking for umap-learn...")
        try:
            import umap  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "umap-learn is not installed. Install it with:  pip install umap-learn"
            ) from exc

        _log("Re-loading pose tracks for UMAP...")
        session_ids = list(result.syllable_assignments.keys())
        tracks = self._load_pose_tracks(session_ids)
        if not tracks:
            raise RuntimeError(
                "No usable pose tracks found for UMAP. "
                "The original pose files could not be read and no cached frame features exist. "
                "Ensure the source drive is mounted (or run syllable discovery once with the "
                "drive connected so features are cached in derived/pose_cache/)."
            )

        _log(f"Building temporal embeddings for {len(tracks)} session(s)...")
        model_meta_path = (
            self._project_root / "derived" / "syllables" / "model_metadata.json"
            if self._project_root else None
        )
        n_lags = 2
        if model_meta_path and model_meta_path.exists():
            meta = read_json(model_meta_path, {})
            n_lags = int(meta.get("n_lags", 2))

        _X, per_session = self._build_temporal_embeddings(tracks, n_lags=n_lags)
        if _X is None or _X.shape[0] == 0:
            raise RuntimeError("Could not build temporal embeddings for UMAP.")

        # Gather labels aligned to embeddings
        all_embs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for sid in session_ids:
            emb = per_session.get(sid)
            if emb is None:
                continue
            syll = self.load_syllables(sid)
            if syll is None:
                continue
            # Align lengths (temporal embedding trims first n_lags frames)
            n = min(len(emb), len(syll))
            all_embs.append(emb[:n])
            all_labels.append(syll[:n].astype(np.int32))

        if not all_embs:
            raise RuntimeError("No paired embeddings + syllable labels found.")

        X = np.vstack(all_embs)
        labels = np.concatenate(all_labels)

        # Subsample to keep UMAP fast
        if X.shape[0] > sample_cap:
            _log(f"Subsampling {sample_cap:,} / {X.shape[0]:,} frames for UMAP...")
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=sample_cap, replace=False)
            X = X[idx]
            labels = labels[idx]
        else:
            _log(f"Using all {X.shape[0]:,} frames for UMAP...")

        _log(
            f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) — "
            "this may take several minutes on CPU..."
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="euclidean",
            random_state=seed,
            verbose=False,
            low_memory=True,
        )
        xy = np.asarray(reducer.fit_transform(X), dtype=np.float32)
        _log("UMAP complete.")
        return xy, labels

    def build_umap_embeddings_full(
        self,
        result: "KeypointMoSeqResult",
        progress_callback: Callable[[str], None] | None = None,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        sample_cap: int = 60_000,
        subsample_strategy: str = "stratified",
        seed: int = 42,
        cancel_flag: list[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extended UMAP that returns per-frame session and frame-index metadata.

        Returns
        -------
        xy            : (N, 2) float32 – UMAP coordinates
        labels        : (N,)   int32   – syllable IDs
        session_ids   : (N,)   object  – session ID string per frame
        frame_indices : (N,)   int64   – original frame index per row

        Subsampling strategies
        ----------------------
        'uniform'    – plain random draw from all frames
        'stratified' – equal representation from every syllable (then trimmed to cap)
        'all'        – no subsampling (may be slow on very large datasets)
        """
        _log = progress_callback or (lambda msg: None)

        _log("Checking for umap-learn...")
        try:
            import umap  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "umap-learn is not installed. Install it with:  pip install umap-learn"
            ) from exc

        _log("Re-loading pose tracks for UMAP...")
        session_ids_list = list(result.syllable_assignments.keys())
        tracks = self._load_pose_tracks(session_ids_list)
        if not tracks:
            raise RuntimeError(
                "No usable pose tracks found for UMAP. "
                "The original pose files could not be read and no cached frame features exist. "
                "Ensure the source drive is mounted (or run syllable discovery once with the "
                "drive connected so features are cached in derived/pose_cache/)."
            )

        _log(f"Building temporal embeddings for {len(tracks)} session(s)...")
        model_meta_path = (
            self._project_root / "derived" / "syllables" / "model_metadata.json"
            if self._project_root else None
        )
        n_lags = 2
        if model_meta_path and model_meta_path.exists():
            meta = read_json(model_meta_path, {})
            n_lags = int(meta.get("n_lags", 2))

        _X, per_session = self._build_temporal_embeddings(tracks, n_lags=n_lags)
        if _X is None or _X.shape[0] == 0:
            raise RuntimeError("Could not build temporal embeddings for UMAP.")

        # Gather aligned (embedding, label, session_id, frame_index) tuples
        all_embs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_sids: list[np.ndarray] = []
        all_fidxs: list[np.ndarray] = []

        for sid in session_ids_list:
            if cancel_flag and cancel_flag[0]:
                raise RuntimeError("Cancelled.")
            emb = per_session.get(sid)
            if emb is None:
                continue
            syll = self.load_syllables(sid)
            if syll is None:
                continue
            n = min(len(emb), len(syll))
            all_embs.append(emb[:n])
            all_labels.append(syll[:n].astype(np.int32))
            all_sids.append(np.full(n, sid, dtype=object))
            all_fidxs.append(np.arange(n, dtype=np.int64))

        if not all_embs:
            raise RuntimeError("No paired embeddings + syllable labels found.")

        X = np.vstack(all_embs)
        labels = np.concatenate(all_labels)
        sids_arr = np.concatenate(all_sids)
        fidxs_arr = np.concatenate(all_fidxs)

        # ── Subsampling ────────────────────────────────────────────────────
        strategy = (subsample_strategy or "stratified").lower()
        use_cap = sample_cap if sample_cap > 0 else len(X)

        if strategy == "all" or len(X) <= use_cap:
            _log(f"Using all {len(X):,} frames for UMAP...")
        elif strategy == "stratified":
            _log(f"Stratified subsampling → {use_cap:,} / {len(X):,} frames...")
            unique_labels = np.unique(labels)
            n_per_label = max(1, use_cap // len(unique_labels))
            rng = np.random.default_rng(seed)
            chosen: list[np.ndarray] = []
            for lbl in unique_labels:
                mask = labels == lbl
                idxs = np.where(mask)[0]
                k = min(len(idxs), n_per_label)
                chosen.append(rng.choice(idxs, k, replace=False))
            idx = np.concatenate(chosen)
            # Trim to cap if stratified overshot
            if len(idx) > use_cap:
                idx = rng.choice(idx, use_cap, replace=False)
            X = X[idx]
            labels = labels[idx]
            sids_arr = sids_arr[idx]
            fidxs_arr = fidxs_arr[idx]
        else:
            # uniform
            _log(f"Uniform subsampling → {use_cap:,} / {len(X):,} frames...")
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), use_cap, replace=False)
            X = X[idx]
            labels = labels[idx]
            sids_arr = sids_arr[idx]
            fidxs_arr = fidxs_arr[idx]

        _log(
            f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
            f"metric={metric}) — this may take several minutes on CPU..."
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=seed,
            verbose=False,
            low_memory=True,
        )
        xy = np.asarray(reducer.fit_transform(X), dtype=np.float32)
        _log("UMAP complete.")
        return xy, labels, sids_arr, fidxs_arr
