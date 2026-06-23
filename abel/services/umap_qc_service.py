"""Orchestrates UMAP QC: embedding preparation → metrics → figures → export.

Architecture
------------
UMAPConfig         — parameters for building the UMAP embedding
QCExportConfig     — how outputs are written (from umap_export)
SyllableQCMetrics  — per-syllable diagnostics dataclass
UMAPData           — all arrays + metadata produced by the embedding step
UMAPQCService      — main service; call run_full_qc() to produce everything

Outputs are written to:
    <project_root>/results/model_qc/<model_name>/
        figures/
        tables/
        reports/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from abel.services.umap_export import (
    QCExportConfig,
    export_figure,
    get_qc_output_dir,
    save_qc_csv,
    save_qc_report,
    save_umap_table,
)
from abel.services.umap_plotting import (
    DARK_THEME,
    LIGHT_THEME,
    ThemeSettings,
    build_qc_dashboard,
    compute_centroids,
    get_theme,
    plot_compactness_chart,
    plot_density_umap,
    plot_standard_umap,
    plot_syllable_highlights,
    plot_transition_graph,
)

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UMAPConfig:
    """Parameters controlling UMAP embedding construction."""

    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42

    subsample_strategy: str = "stratified"
    """How to downsample large point clouds.
    Options: 'uniform' | 'stratified' | 'all'
    """

    max_frames: int = 60_000
    """Maximum frames to embed.  0 = no limit."""

    n_lags: int = 2
    """Temporal lag depth used when building frame embeddings (overridden from
    saved model metadata when available)."""

    cmap: str = "turbo"
    """Matplotlib colourmap for syllable colours."""


@dataclass
class SyllableQCMetrics:
    """Per-syllable QC diagnostics."""

    syllable_id: int = 0
    occupancy_frames: int = 0
    occupancy_fraction: float = 0.0
    n_bouts: int = 0
    mean_bout_length: float = 0.0
    median_bout_length: float = 0.0
    within_cluster_spread: float = 0.0
    nearest_centroid_distance: float = 0.0
    compactness_score: float = 0.0
    transition_entropy: float = 0.0
    self_transition_probability: float = 0.0
    centroid_umap1: float = 0.0
    centroid_umap2: float = 0.0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class UMAPData:
    """Full result of the embedding step."""

    xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    session_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    frame_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    n_syllables: int = 0
    umap_settings: dict = field(default_factory=dict)
    model_name: str = ""


@dataclass
class QCResult:
    """Outcome of a full QC run."""

    output_dir: Path | None = None
    umap_data: UMAPData | None = None
    metrics: list[SyllableQCMetrics] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exported_paths: list[Path] = field(default_factory=list)
    oversplit_assessment: dict = field(default_factory=dict)
    success: bool = False


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class UMAPQCService:
    """Manages the full UMAP QC pipeline for a trained syllable model."""

    def __init__(self) -> None:
        self._project_root: Path | None = None

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_full_qc(
        self,
        model_name: str,
        umap_config: UMAPConfig | None = None,
        export_config: QCExportConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> QCResult:
        """Run the complete QC pipeline and write all outputs.

        Steps:
          1. Build UMAP embedding (re-loads pose tracks)
          2. Compute per-syllable QC metrics
          3. Generate and export all figures
          4. Write CSV / JSON reports
        """
        result = QCResult()
        _log = progress_callback or (lambda msg: None)

        if not self._project_root:
            result.warnings.append("No project loaded.")
            return result

        cfg = umap_config or UMAPConfig()
        exp_cfg = export_config or QCExportConfig()
        theme = get_theme(exp_cfg.dark_theme)

        out_dir = get_qc_output_dir(self._project_root, model_name)
        result.output_dir = out_dir

        # ── Step 1: Build embeddings ────────────────────────────────────
        _log("Building UMAP embeddings (loading pose tracks)...")
        try:
            umap_data = self._build_embeddings(model_name, cfg, _log, cancel_flag)
        except Exception as exc:
            result.warnings.append(f"UMAP embedding failed: {exc}")
            logger.exception("UMAP QC: embedding failed")
            return result

        if cancel_flag and cancel_flag[0]:
            return result

        result.umap_data = umap_data

        # ── Step 2: Compute QC metrics ──────────────────────────────────
        _log("Computing syllable QC metrics...")
        try:
            syllable_seqs = self._load_syllable_sequences()
            trans_matrix, metrics_list = self._compute_metrics(umap_data, syllable_seqs)
        except Exception as exc:
            result.warnings.append(f"Metric computation failed: {exc}")
            logger.exception("UMAP QC: metrics failed")
            trans_matrix = None
            metrics_list = []

        result.metrics = metrics_list
        metrics_dicts = [m.to_dict() for m in metrics_list]

        # ── Step 3: Generate and export figures ─────────────────────────
        exported: list[Path] = []

        if cancel_flag and cancel_flag[0]:
            return result

        # 3a. Standard UMAP (dark + light)
        _log("Generating standard UMAP figure...")
        _safe_export(
            lambda: plot_standard_umap(
                umap_data.xy, umap_data.labels, DARK_THEME,
                include_labels=exp_cfg.include_labels, cmap=cfg.cmap, dpi=exp_cfg.dpi
            ),
            out_dir, "umap_syllables_dark",
            exp_cfg.export_formats, exp_cfg.dpi, exported,
        )
        _safe_export(
            lambda: plot_standard_umap(
                umap_data.xy, umap_data.labels, LIGHT_THEME,
                include_labels=exp_cfg.include_labels, cmap=cfg.cmap, dpi=exp_cfg.dpi
            ),
            out_dir, "umap_syllables_light",
            exp_cfg.export_formats, exp_cfg.dpi, exported,
        )

        # 3b. Density UMAP
        if exp_cfg.plot_density:
            _log("Generating density UMAP...")
            _safe_export(
                lambda: plot_density_umap(
                    umap_data.xy, umap_data.labels, theme,
                    mode="hexbin", include_labels=exp_cfg.include_labels, dpi=exp_cfg.dpi
                ),
                out_dir, f"umap_density_{'dark' if exp_cfg.dark_theme else 'light'}",
                exp_cfg.export_formats, exp_cfg.dpi, exported,
            )

        if cancel_flag and cancel_flag[0]:
            result.exported_paths = exported
            return result

        # 3c. Per-syllable highlight panels
        if exp_cfg.plot_per_syllable:
            _log("Generating per-syllable highlight panels...")
            _safe_export(
                lambda: plot_syllable_highlights(
                    umap_data.xy, umap_data.labels,
                    metrics=metrics_dicts, theme=theme,
                    n_cols=exp_cfg.n_cols_highlight, dpi=exp_cfg.dpi,
                ),
                out_dir, "umap_syllable_highlights",
                exp_cfg.export_formats, exp_cfg.dpi, exported,
            )

        # 3d. Compactness / QC chart
        if exp_cfg.plot_compactness and metrics_dicts:
            _log("Generating compactness chart...")
            _safe_export(
                lambda: plot_compactness_chart(metrics_dicts, theme, dpi=exp_cfg.dpi),
                out_dir, "syllable_compactness",
                exp_cfg.export_formats, exp_cfg.dpi, exported,
            )

        # 3e. Transition graph
        if exp_cfg.plot_transition_graph and trans_matrix is not None:
            _log("Generating transition graph...")
            _safe_export(
                lambda: plot_transition_graph(
                    trans_matrix,
                    theme=theme,
                    threshold=exp_cfg.transition_threshold,
                    dpi=exp_cfg.dpi,
                ),
                out_dir, f"transition_graph_threshold_{exp_cfg.transition_threshold:.2f}".replace(".", "p"),
                exp_cfg.export_formats, exp_cfg.dpi, exported,
            )

        if cancel_flag and cancel_flag[0]:
            result.exported_paths = exported
            return result

        # 3f. Dashboard
        if exp_cfg.plot_dashboard:
            _log("Generating QC dashboard...")
            _safe_export(
                lambda: build_qc_dashboard(
                    umap_data.xy, umap_data.labels,
                    metrics=metrics_dicts,
                    trans_matrix=trans_matrix,
                    theme=theme,
                    dpi=exp_cfg.dpi,
                    cmap=cfg.cmap,
                ),
                out_dir, "qc_dashboard",
                exp_cfg.export_formats, exp_cfg.dpi, exported,
            )

        # ── Step 4: Write tables and report ─────────────────────────────
        _log("Writing QC tables and report...")
        try:
            save_umap_table(
                out_dir,
                umap_data.xy,
                umap_data.labels,
                session_ids=umap_data.session_ids,
                frame_indices=umap_data.frame_indices,
            )
            save_qc_csv(metrics_dicts, out_dir)

            warnings = self._generate_warnings(metrics_list, umap_data.n_syllables)
            model_summary = _build_model_summary(umap_data, metrics_list)
            save_qc_report(
                per_syllable_metrics=metrics_dicts,
                model_summary=model_summary,
                warnings=warnings,
                output_dir=out_dir,
                umap_settings=umap_data.umap_settings,
            )
        except Exception as exc:
            result.warnings.append(f"Report writing failed: {exc}")
            logger.warning("UMAP QC: report step failed: %s", exc)

        result.exported_paths = exported
        result.success = True

        # Over-split assessment (runs on already-computed metrics)
        result.oversplit_assessment = self._assess_oversplit(
            metrics_list, umap_data.n_syllables
        )

        _log(f"QC complete — {len(exported)} figure(s) exported to {out_dir}")
        logger.info("UMAP QC complete: %d figure(s) → %s", len(exported), out_dir)
        return result

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _build_embeddings(
        self,
        model_name: str,
        cfg: UMAPConfig,
        log: Callable[[str], None],
        cancel_flag: list[bool] | None,
    ) -> UMAPData:
        """Re-use KeypointMoSeqService to build UMAP coordinates."""
        from abel.services.keypoint_moseq_service import KeypointMoSeqService  # noqa: PLC0415

        svc = KeypointMoSeqService()
        svc.set_project(self._project_root)  # type: ignore[arg-type]

        existing = svc.load_existing_result()
        if not existing or not existing.success:
            raise RuntimeError("Run syllable discovery before generating QC outputs.")

        # Read n_lags from saved metadata if available
        meta_path = self._project_root / "derived" / "syllables" / "model_metadata.json"  # type: ignore[operator]
        n_lags = cfg.n_lags
        if meta_path.exists():
            from abel.storage.file_store import read_json  # noqa: PLC0415
            meta = read_json(meta_path, {})
            n_lags = int(meta.get("n_lags", cfg.n_lags))

        def _prog(msg: str) -> None:
            log(msg)

        log("Loading and embedding pose tracks via KeypointMoSeqService...")
        xy, labels, session_ids, frame_indices = svc.build_umap_embeddings_full(
            existing,
            n_neighbors=cfg.n_neighbors,
            min_dist=cfg.min_dist,
            metric=cfg.metric,
            sample_cap=cfg.max_frames if cfg.max_frames > 0 else 200_000,
            subsample_strategy=cfg.subsample_strategy,
            seed=cfg.random_state,
            progress_callback=_prog,
            cancel_flag=cancel_flag,
        )

        n_syllables = int(labels.max()) + 1 if len(labels) > 0 else 0

        return UMAPData(
            xy=xy,
            labels=labels,
            session_ids=session_ids,
            frame_indices=frame_indices,
            n_syllables=n_syllables,
            umap_settings={
                "n_neighbors": cfg.n_neighbors,
                "min_dist": cfg.min_dist,
                "metric": cfg.metric,
                "random_state": cfg.random_state,
                "subsample_strategy": cfg.subsample_strategy,
                "max_frames": cfg.max_frames,
                "n_lags": n_lags,
            },
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Syllable sequence loading
    # ------------------------------------------------------------------

    def _load_syllable_sequences(self) -> list[np.ndarray]:
        """Load all per-session syllable assignment arrays (ordered by frame)."""
        if not self._project_root:
            return []

        syllables_dir = self._project_root / "derived" / "syllables"
        seqs: list[np.ndarray] = []
        for npz_path in sorted(syllables_dir.glob("*_syllables.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                seqs.append(data["syllables"].astype(np.int32))
            except Exception as exc:
                logger.warning("Could not load syllable file %s: %s", npz_path, exc)
        return seqs

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        umap_data: UMAPData,
        syllable_seqs: list[np.ndarray],
    ) -> tuple[np.ndarray | None, list[SyllableQCMetrics]]:
        """Compute per-syllable diagnostics. Returns (transition_matrix, metrics)."""
        xy = umap_data.xy
        labels = umap_data.labels
        n_syllables = max(umap_data.n_syllables, int(labels.max()) + 1 if len(labels) else 0)

        if n_syllables == 0 or len(xy) == 0:
            return None, []

        # Centroids
        centroids = compute_centroids(xy, labels)

        # Nearest-centroid distances
        centroid_arr = np.array([
            centroids.get(i, (np.nan, np.nan)) for i in range(n_syllables)
        ])
        neighbor_dists = _compute_nearest_centroid_distances(centroid_arr)

        # Per-syllable spread
        spreads = {}
        for syl_id in range(n_syllables):
            mask = labels == syl_id
            pts = xy[mask]
            spreads[syl_id] = float(np.std(pts)) if len(pts) > 1 else 0.0

        max_spread = max(spreads.values()) if spreads else 1.0

        # Transition matrix from full assignment sequences
        trans_matrix: np.ndarray | None = None
        trans_entropy: dict[int, float] = {}
        self_trans: dict[int, float] = {}
        if syllable_seqs:
            trans_matrix = _compute_transition_matrix(syllable_seqs, n_syllables)
            for syl_id in range(n_syllables):
                row_sum = trans_matrix[syl_id].sum()
                if row_sum > 0:
                    probs = trans_matrix[syl_id] / row_sum
                    nonzero = probs[probs > 0]
                    trans_entropy[syl_id] = float(-np.sum(nonzero * np.log2(nonzero + 1e-12)))
                    self_trans[syl_id] = float(trans_matrix[syl_id, syl_id] / row_sum)
                else:
                    trans_entropy[syl_id] = 0.0
                    self_trans[syl_id] = 0.0

        # Bout statistics from full sequences
        bout_stats = _compute_bout_stats(syllable_seqs, n_syllables)

        # Total frames from all seqs
        total_frames = sum(len(s) for s in syllable_seqs) if syllable_seqs else len(labels)
        count_in_seqs: dict[int, int] = {}
        if syllable_seqs:
            all_seq = np.concatenate(syllable_seqs)
            for syl_id in range(n_syllables):
                count_in_seqs[syl_id] = int(np.sum(all_seq == syl_id))
        else:
            for syl_id in range(n_syllables):
                count_in_seqs[syl_id] = int(np.sum(labels == syl_id))

        metrics: list[SyllableQCMetrics] = []
        for syl_id in range(n_syllables):
            cx, cy = centroids.get(syl_id, (0.0, 0.0))
            spread = spreads.get(syl_id, 0.0)
            compactness = max(0.0, 1.0 - (spread / max(max_spread, 1e-9)))
            occ_frames = count_in_seqs.get(syl_id, 0)
            occ_frac = occ_frames / max(total_frames, 1)
            bstats = bout_stats.get(syl_id, {})
            metrics.append(
                SyllableQCMetrics(
                    syllable_id=syl_id,
                    occupancy_frames=occ_frames,
                    occupancy_fraction=occ_frac,
                    n_bouts=bstats.get("n_bouts", 0),
                    mean_bout_length=bstats.get("mean_bout_length", 0.0),
                    median_bout_length=bstats.get("median_bout_length", 0.0),
                    within_cluster_spread=spread,
                    nearest_centroid_distance=float(neighbor_dists[syl_id]),
                    compactness_score=compactness,
                    transition_entropy=trans_entropy.get(syl_id, 0.0),
                    self_transition_probability=self_trans.get(syl_id, 0.0),
                    centroid_umap1=cx,
                    centroid_umap2=cy,
                )
            )

        return trans_matrix, metrics

    # ------------------------------------------------------------------
    # Warnings / heuristics
    # ------------------------------------------------------------------

    def _generate_warnings(
        self,
        metrics: list[SyllableQCMetrics],
        n_syllables: int,
    ) -> list[str]:
        """Generate heuristic QC warnings for the report."""
        if not metrics:
            return []
        warnings: list[str] = []

        low_occ = [m for m in metrics if m.occupancy_fraction < 0.005]
        if len(low_occ) / max(n_syllables, 1) > 0.3:
            warnings.append(
                f"High number of low-occupancy syllables ({len(low_occ)}/{n_syllables}): "
                "possible over-segmentation."
            )

        spreads = [m.within_cluster_spread for m in metrics]
        mean_spread = float(np.mean(spreads)) if spreads else 0.0
        diffuse = [m for m in metrics if m.within_cluster_spread > 1.5 * mean_spread]
        if len(diffuse) / max(n_syllables, 1) > 0.25:
            warnings.append(
                f"Many diffuse syllables ({len(diffuse)}/{n_syllables}) with large "
                "within-cluster spread: possible poor state separation."
            )

        high_entropy = [m for m in metrics if m.transition_entropy > 3.5]
        if len(high_entropy) / max(n_syllables, 1) > 0.4:
            warnings.append(
                "Transition graph is highly noisy (many high-entropy transitions): "
                "possible unstable model."
            )

        compact = [m for m in metrics if m.compactness_score > 0.6]
        stable = [m for m in metrics if m.self_transition_probability > 0.5]
        if len(compact) / max(n_syllables, 1) > 0.7 and len(stable) / max(n_syllables, 1) > 0.5:
            warnings.append(
                "Strong compact clusters with stable self-transitions: "
                "good candidate model."
            )

        return warnings

    def _assess_oversplit(
        self,
        metrics: list[SyllableQCMetrics],
        n_syllables: int,
    ) -> dict:
        """Assess whether the model shows signs of over-splitting.

        Returns a dict with keys:
          - severity  : 'none' | 'low' | 'medium' | 'high'
          - score     : float 0–1 (higher = stronger over-split signal)
          - explanation : plain-language summary for the user
          - indicators  : list of specific signal descriptions
        """
        if not metrics or n_syllables < 2:
            return {
                "severity": "none",
                "score": 0.0,
                "explanation": "Insufficient data for over-split assessment.",
                "indicators": [],
            }

        indicators: list[str] = []
        sub_scores: list[float] = []

        # ── 1. Low-occupancy syllables (< 0.5% of total frames) ────────────────
        low_occ = [m for m in metrics if m.occupancy_fraction < 0.005]
        low_occ_frac = len(low_occ) / n_syllables
        if low_occ_frac > 0.40:
            indicators.append(
                f"{len(low_occ)}/{n_syllables} syllables have < 0.5 % occupancy — "
                "almost all data lives in a few dominant clusters while the rest are "
                "near-empty fragments."
            )
            sub_scores.append(min(1.0, low_occ_frac * 1.4))
        elif low_occ_frac > 0.15:
            indicators.append(
                f"{len(low_occ)}/{n_syllables} syllables have < 0.5 % occupancy."
            )
            sub_scores.append(low_occ_frac)

        # ── 2. Very short mean bout lengths (< 5 frames) ─────────────────────
        active = [m for m in metrics if m.n_bouts > 0]
        short_bouts = [m for m in active if 0 < m.mean_bout_length < 5]
        if active:
            short_frac = len(short_bouts) / len(active)
            if short_frac > 0.25:
                indicators.append(
                    f"{len(short_bouts)}/{len(active)} active syllables have mean bout "
                    "length < 5 frames — very short bouts suggest the model is splitting "
                    "within single movements rather than at behavior boundaries."
                )
                sub_scores.append(min(1.0, short_frac * 1.2))

        # ── 3. Overlapping UMAP clusters ────────────────────────────────────
        # centroid distance < within-cluster spread ⇒ overlapping
        overlap_count = sum(
            1 for m in metrics
            if np.isfinite(m.nearest_centroid_distance)
            and m.within_cluster_spread > 0
            and m.nearest_centroid_distance < m.within_cluster_spread
        )
        overlap_frac = overlap_count / n_syllables
        if overlap_frac > 0.35:
            indicators.append(
                f"{overlap_count}/{n_syllables} syllables overlap their nearest neighbour "
                "in UMAP space (centroid distance < cluster spread) — these likely represent "
                "fragments of the same underlying behaviour."
            )
            sub_scores.append(min(1.0, overlap_frac))
        elif overlap_frac > 0.15:
            indicators.append(
                f"{overlap_count}/{n_syllables} syllables show partial UMAP overlap with "
                "their nearest neighbour."
            )
            sub_scores.append(overlap_frac * 0.6)

        # ── 4. Spread-to-separation ratio ──────────────────────────────────
        ncd = [
            m.nearest_centroid_distance for m in metrics
            if np.isfinite(m.nearest_centroid_distance)
        ]
        spreads = [m.within_cluster_spread for m in metrics]
        if ncd and spreads:
            median_ncd = float(np.median(ncd))
            mean_spread = float(np.mean(spreads))
            if median_ncd > 0:
                ratio = mean_spread / median_ncd
                if ratio > 0.75:
                    indicators.append(
                        f"Mean within-cluster spread ({mean_spread:.2f}) is "
                        f"{ratio:.1%} of the median inter-centroid distance "
                        f"({median_ncd:.2f}) — clusters are not well separated."
                    )
                    sub_scores.append(min(1.0, ratio))

        # ── Aggregate ─────────────────────────────────────────────────────
        overall_score = float(np.mean(sub_scores)) if sub_scores else 0.0

        if overall_score < 0.15:
            severity = "none"
        elif overall_score < 0.35:
            severity = "low"
        elif overall_score < 0.60:
            severity = "medium"
        else:
            severity = "high"

        if severity == "none":
            explanation = (
                "No strong signs of over-splitting detected. The model’s syllable "
                "count appears appropriate for the behavioural complexity in your "
                "recordings."
            )
        elif severity == "low":
            explanation = (
                "Mild signs of over-splitting. A small number of syllables may be "
                "redundant fragments of the same movement. Review the syllable clips "
                "for near-identical pairs, or try slightly reducing the syllable count "
                "on the next run."
            )
        elif severity == "medium":
            explanation = (
                "Moderate over-splitting detected. Several syllables appear to represent "
                "fragments of the same underlying behaviour — they show short bout lengths, "
                "low occupancy, or overlapping UMAP clusters. Consider reducing the number "
                "of syllables by 15–30 % and re-running the model."
            )
        else:
            explanation = (
                "Strong signs of over-splitting. The model has created many syllables that "
                "are unlikely to correspond to distinct behaviours: most have very low "
                "occupancy, short bouts, and overlapping UMAP regions. Substantially "
                "reducing the syllable count (by 30–50 %) is strongly recommended before "
                "using this model for behaviour classification."
            )

        return {
            "severity": severity,
            "score": round(overall_score, 3),
            "explanation": explanation,
            "indicators": indicators,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_transition_matrix(syllable_seqs: list[np.ndarray], n: int) -> np.ndarray:
    """Build an (n × n) transition count matrix from per-session sequences."""
    mat = np.zeros((n, n), dtype=np.float64)
    for seq in syllable_seqs:
        if len(seq) < 2:
            continue
        for t in range(len(seq) - 1):
            a, b = int(seq[t]), int(seq[t + 1])
            if 0 <= a < n and 0 <= b < n:
                mat[a, b] += 1
    return mat


def _compute_bout_stats(
    syllable_seqs: list[np.ndarray],
    n_syllables: int,
) -> dict[int, dict]:
    """Return per-syllable bout count and duration statistics."""
    bouts: dict[int, list[int]] = {i: [] for i in range(n_syllables)}

    for seq in syllable_seqs:
        if len(seq) == 0:
            continue
        prev = int(seq[0])
        run = 1
        for t in range(1, len(seq)):
            curr = int(seq[t])
            if curr == prev:
                run += 1
            else:
                if 0 <= prev < n_syllables:
                    bouts[prev].append(run)
                prev = curr
                run = 1
        if 0 <= prev < n_syllables:
            bouts[prev].append(run)

    result: dict[int, dict] = {}
    for syl_id in range(n_syllables):
        bs = bouts[syl_id]
        if bs:
            result[syl_id] = {
                "n_bouts": len(bs),
                "mean_bout_length": float(np.mean(bs)),
                "median_bout_length": float(np.median(bs)),
            }
        else:
            result[syl_id] = {"n_bouts": 0, "mean_bout_length": 0.0, "median_bout_length": 0.0}
    return result


def _compute_nearest_centroid_distances(centroid_arr: np.ndarray) -> np.ndarray:
    """For each centroid, return its distance to the nearest other centroid."""
    n = len(centroid_arr)
    dists = np.full(n, np.nan)
    for i in range(n):
        ci = centroid_arr[i]
        if np.any(np.isnan(ci)):
            continue
        min_d = np.inf
        for j in range(n):
            if j == i:
                continue
            cj = centroid_arr[j]
            if np.any(np.isnan(cj)):
                continue
            d = float(np.linalg.norm(ci - cj))
            if d < min_d:
                min_d = d
        dists[i] = min_d if np.isfinite(min_d) else 0.0
    return dists


def _build_model_summary(
    umap_data: UMAPData,
    metrics: list[SyllableQCMetrics],
) -> dict:
    """Compute model-level summary statistics for the report."""
    n = umap_data.n_syllables
    if not metrics:
        return {"total_syllables": n, "total_frames_embedded": len(umap_data.xy)}

    low_occ = [m for m in metrics if m.occupancy_fraction < 0.005]
    spreads = [m.within_cluster_spread for m in metrics]
    compactness = [m.compactness_score for m in metrics]
    ncd = [m.nearest_centroid_distance for m in metrics if np.isfinite(m.nearest_centroid_distance)]

    return {
        "total_syllables": n,
        "total_frames_embedded": len(umap_data.xy),
        "fraction_low_occupancy": len(low_occ) / max(n, 1),
        "mean_compactness": float(np.mean(compactness)) if compactness else 0.0,
        "median_compactness": float(np.median(compactness)) if compactness else 0.0,
        "median_nearest_centroid_distance": float(np.median(ncd)) if ncd else 0.0,
        "mean_within_cluster_spread": float(np.mean(spreads)) if spreads else 0.0,
    }


def _safe_export(
    plot_fn: Callable,
    out_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    exported: list[Path],
) -> None:
    """Call *plot_fn()*, export the figure, close it, append paths to *exported*."""
    try:
        fig = plot_fn()
        paths = export_figure(fig, out_dir, stem, formats, dpi)
        exported.extend(paths)
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
            plt.close(fig)
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Could not generate/export figure '%s': %s", stem, exc)
