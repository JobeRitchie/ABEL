"""High-resolution multi-format figure export utilities for UMAP QC outputs.

Handles:
- Saving matplotlib figures to PNG / SVG / PDF at configurable DPI
- QC summary CSV table (one row per syllable)
- Machine-readable JSON / YAML report
- Structured output directory management
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QCExportConfig:
    """Controls how QC outputs are written to disk."""

    export_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
    """File formats to save for every figure.  Supported: png, svg, pdf."""

    dpi: int = 300
    """Raster DPI for PNG exports.  600 for publication, 150 for preview."""

    dark_theme: bool = True
    """Use dark background palette when True, light/white when False."""

    include_labels: bool = True
    """Draw syllable ID labels on scatter/highlight plots."""

    plot_density: bool = True
    plot_per_syllable: bool = True
    plot_transition_graph: bool = True
    plot_compactness: bool = True
    plot_dashboard: bool = True

    n_cols_highlight: int = 6
    """Columns in the per-syllable highlight grid."""

    top_n_per_syllable: int = 5
    """Representative frames / clips per syllable (future use)."""

    transition_threshold: float = 0.02
    """Minimum edge weight to draw in the transition graph."""


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_qc_output_dir(project_root: Path, model_name: str) -> Path:
    """Return (and create) the structured QC output folder for *model_name*."""
    qc_dir = project_root / "results" / "model_qc" / model_name
    for sub in ("figures", "tables", "reports"):
        (qc_dir / sub).mkdir(parents=True, exist_ok=True)
    return qc_dir


# ---------------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------------


def export_figure(
    fig: "matplotlib.figure.Figure",
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int = 300,
    facecolor: str | None = None,
) -> list[Path]:
    """Save *fig* in all requested *formats* under *output_dir/figures/*.

    Returns list of paths actually written.
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    fc = facecolor or fig.get_facecolor()

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt not in {"png", "svg", "pdf"}:
            logger.warning("Unsupported export format %r — skipping.", fmt)
            continue

        path = figures_dir / f"{stem}.{fmt}"
        try:
            kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": fc}
            if fmt == "png":
                kwargs["dpi"] = dpi
            elif fmt == "pdf":
                # rasterize raster layers; keep vector layers crisp
                kwargs["dpi"] = min(dpi, 300)
            fig.savefig(path, format=fmt, **kwargs)
            saved.append(path)
            logger.debug("Exported figure → %s", path)
        except Exception as exc:
            logger.warning("Could not export figure to %s: %s", path, exc)

    return saved


# ---------------------------------------------------------------------------
# UMAP table (frame-level metadata)
# ---------------------------------------------------------------------------


def save_umap_table(
    output_dir: Path,
    xy: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray | list | None = None,
    frame_indices: np.ndarray | list | None = None,
) -> Path:
    """Persist a CSV with columns: umap1, umap2, syllable_id, session_id, frame_index."""
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError:
        logger.warning("pandas not installed — skipping UMAP table export.")
        return output_dir / "tables" / "umap_embedding.csv"

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    n = len(xy)
    df = pd.DataFrame({
        "umap1": xy[:, 0].astype(np.float32),
        "umap2": xy[:, 1].astype(np.float32),
        "syllable_id": labels.astype(np.int32),
        "session_id": list(session_ids) if session_ids is not None else [""] * n,
        "frame_index": list(frame_indices) if frame_indices is not None else list(range(n)),
    })
    path = tables_dir / "umap_embedding.csv"
    df.to_csv(path, index=False)
    logger.info("Saved UMAP embedding table (%d rows) → %s", n, path)
    return path


# ---------------------------------------------------------------------------
# QC metrics CSV
# ---------------------------------------------------------------------------


def save_qc_csv(metrics: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write per-syllable QC metrics to a CSV table."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "syllable_qc_metrics.csv"

    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(metrics)
        df.to_csv(path, index=False)
        logger.info("Saved QC metrics table (%d rows) → %s", len(metrics), path)
    except ImportError:
        # Fallback: manual CSV write
        if not metrics:
            path.write_text("(no data)\n")
            return path
        keys = list(metrics[0].keys())
        lines = [",".join(str(k) for k in keys)]
        for row in metrics:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
        path.write_text("\n".join(lines))
        logger.info("Saved QC metrics table (no pandas) → %s", path)

    return path


# ---------------------------------------------------------------------------
# QC JSON/YAML report
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_qc_report(
    per_syllable_metrics: list[dict[str, Any]],
    model_summary: dict[str, Any],
    warnings: list[str],
    output_dir: Path,
    umap_settings: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON QC report to *output_dir/reports/qc_report.json*."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "qc_report.json"

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_summary": _make_json_safe(model_summary),
        "umap_settings": _make_json_safe(umap_settings or {}),
        "warnings": warnings,
        "per_syllable_metrics": [_make_json_safe(m) for m in per_syllable_metrics],
    }
    path.write_text(json.dumps(report, indent=2))
    logger.info("Saved QC report → %s", path)
    return path
