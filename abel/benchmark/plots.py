"""Visualization generators for ablation benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.benchmark.runner import RunResult
from abel.benchmark.metrics import (
    apply_behavior_names,
    results_to_dataframe,
    compute_deltas,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Colour palette ────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3",  # blue — baseline
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
    "#E91E63",  # pink
    "#CDDC39",  # lime
]


def _colour(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ── Bar chart comparison ──────────────────────────────────────────────

def metric_bar_chart(
    results: list[RunResult],
    save_path: Path | None = None,
    behavior_names: dict[str, str] | None = None,
) -> Figure | None:
    """Grouped bar chart of F1, Precision, Recall, PR-AUC per run with SEM error bars.

    When multiple behaviors are present, a separate subplot row is produced per behavior.
    """
    if not _HAS_MPL:
        return None

    df = results_to_dataframe(results)
    if behavior_names:
        df = apply_behavior_names(df, behavior_names)
    behaviors = df["Behavior"].unique().tolist()
    metrics = ["Precision", "Recall", "F1", "PR-AUC"]
    sem_cols = ["Precision SEM", "Recall SEM", "F1 SEM", "PR-AUC SEM"]
    n_behaviors = len(behaviors)

    fig, axes = plt.subplots(
        n_behaviors, 1, figsize=(max(8, 10), max(4.5 * n_behaviors, 5)),
        squeeze=False,
    )

    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx, 0]
        bdf = df[df["Behavior"] == behavior]
        runs = bdf["Run"].tolist()
        n_runs = len(runs)
        n_metrics = len(metrics)
        x = np.arange(n_runs)
        width = 0.8 / n_metrics

        for j, (metric, sem_col) in enumerate(zip(metrics, sem_cols)):
            vals = bdf[metric].to_numpy(dtype=float)
            errs = bdf[sem_col].to_numpy(dtype=float)
            offset = (j - n_metrics / 2 + 0.5) * width
            ax.bar(
                x + offset, vals, width, yerr=errs, capsize=2,
                label=metric, color=_colour(j), alpha=0.85,
                error_kw={"linewidth": 0.8},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("without_", "w/o ").replace("baseline_", "") for r in runs],
            rotation=35, ha="right", fontsize=8,
        )
        ax.set_ylabel("Score")
        beh_label = behavior if behavior else "(all)"
        ax.set_title(f"Ablation — {beh_label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── Delta impact chart ────────────────────────────────────────────────

def delta_impact_chart(
    results: list[RunResult],
    save_path: Path | None = None,
    behavior_names: dict[str, str] | None = None,
) -> Figure | None:
    """Grouped horizontal bar chart showing F1 delta per toggle, with all
    behaviors side-by-side for easy cross-behavior comparison.
    """
    if not _HAS_MPL:
        return None

    df = results_to_dataframe(results)
    if behavior_names:
        df = apply_behavior_names(df, behavior_names)
    behaviors = df["Behavior"].unique().tolist()
    n_beh = len(behaviors)

    if n_beh == 0:
        return None

    # Collect deltas for each behavior — relative to all_on so we see cost of removing each feature
    per_beh: list[tuple[str, pd.DataFrame]] = []
    toggle_set: set[str] = set()
    standalone_set: set[str] = set()
    for behavior in behaviors:
        deltas = compute_deltas(df, baseline_name="baseline_all_on", behavior=behavior)
        if deltas.empty or "ΔF1" not in deltas.columns:
            continue
        # Standard ablation rows (without_*)
        mask = deltas["Run"].str.startswith("without_")
        abl = deltas[mask].copy()
        # Standalone rows (video_only, etc.)
        solo_mask = deltas["Run"].isin({"video_only"})
        solo = deltas[solo_mask].copy()
        if abl.empty and solo.empty:
            continue
        abl["Toggle"] = abl["Run"].str.replace("without_", "", n=1).str.replace("_", " ").str.title()
        solo["Toggle"] = solo["Run"].str.replace("_", " ").str.title()
        combined = pd.concat([abl, solo], ignore_index=True)
        per_beh.append((behavior, combined))
        toggle_set.update(abl["Toggle"].tolist())
        standalone_set.update(solo["Toggle"].tolist())

    if not per_beh:
        return None

    # Ablation toggles first, then standalone evaluations separated by a gap
    toggles = sorted(toggle_set)
    standalones = sorted(standalone_set)
    all_labels = toggles + standalones
    n_labels = len(all_labels)
    bar_height = 0.7 / max(1, len(per_beh))

    # Add a visual gap between ablation and standalone sections
    y_positions = list(range(len(toggles)))
    gap = 0.6 if standalones else 0
    for i in range(len(standalones)):
        y_positions.append(len(toggles) + gap + i)
    y_base = np.array(y_positions, dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.8 * n_labels * len(per_beh))))

    for b_idx, (beh_name, deltas) in enumerate(per_beh):
        beh_label = beh_name[:20] if beh_name else "(all)"
        vals: list[float] = []
        for label in all_labels:
            row = deltas[deltas["Toggle"] == label]
            vals.append(float(row["ΔF1"].iloc[0]) if not row.empty else 0.0)
        arr = np.array(vals)
        offset = (b_idx - len(per_beh) / 2 + 0.5) * bar_height
        colours = [_colour(b_idx)] * n_toggles
        bars = ax.barh(
            y_base + offset, arr, bar_height,
            label=beh_label, color=colours, alpha=0.85,
            edgecolor="white", linewidth=0.3,
        )
        for bar, val in zip(bars, arr):
            if not np.isnan(val) and abs(val) > 0.0001:
                ax.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}",
                    ha="left" if val >= 0 else "right",
                    va="center", fontsize=7,
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels(all_labels, fontsize=9)
    # Visually distinguish standalone rows with italic font
    for i, label in enumerate(ax.get_yticklabels()):
        if all_labels[i] in standalone_set:
            label.set_fontstyle("italic")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("ΔF1 (negative = removing feature hurts performance)")
    ax.set_title("Feature Impact — Effect of Removing Each Feature")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── PR curves overlay ─────────────────────────────────────────────────

def pr_curves_overlay(
    results: list[RunResult],
    save_path: Path | None = None,
    behavior_names: dict[str, str] | None = None,
) -> Figure | None:
    """Overlaid Precision-Recall curves for each ablation run."""
    if not _HAS_MPL:
        return None

    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, r in enumerate(results):
        if r.y_true is None or r.y_score is None or r.error:
            continue
        binary_true = (r.y_true == 1).astype(int) if r.y_true.max() > 1 else r.y_true
        prec, rec, _ = precision_recall_curve(binary_true, r.y_score)
        label = r.run_name.replace("without_", "w/o ").replace("baseline_", "")
        lw = 2.5 if "baseline" in r.run_name else 1.5
        ax.plot(rec, prec, label=f"{label} (AP={r.pr_auc_mean:.3f})",
                color=_colour(i), linewidth=lw, alpha=0.85)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Ablation Comparison")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── Confusion matrices grid ──────────────────────────────────────────

def confusion_matrix_grid(
    results: list[RunResult],
    save_path: Path | None = None,
    behavior_names: dict[str, str] | None = None,
) -> Figure | None:
    """Grid of confusion matrices, one per run."""
    if not _HAS_MPL:
        return None

    valid = [r for r in results if r.confusion_matrix and not r.error]
    if not valid:
        return None

    n = len(valid)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, r in enumerate(valid):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        cm = np.array(r.confusion_matrix)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(r.run_name.replace("without_", "w/o\n").replace("baseline_", ""),
                     fontsize=8)

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")

        if r.label_map:
            labels = [str(r.label_map.get(i, i)) for i in range(cm.shape[0])]
            if behavior_names:
                labels = [behavior_names.get(l, l) for l in labels]
            short_labels = [l[:12] for l in labels]
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(short_labels)))
            ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Confusion Matrices — Per Ablation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── Delta heatmap ─────────────────────────────────────────────────────

def delta_heatmap(
    results: list[RunResult],
    save_path: Path | None = None,
    behavior_names: dict[str, str] | None = None,
) -> Figure | None:
    """Heatmap of ΔF1 for every toggle × behavior combination.

    Rows = feature toggles, columns = behaviors, cells = ΔF1.
    """
    if not _HAS_MPL:
        return None

    df = results_to_dataframe(results)
    if behavior_names:
        df = apply_behavior_names(df, behavior_names)
    behaviors = sorted(df["Behavior"].unique().tolist())

    # Collect per-toggle per-behavior ΔF1
    toggle_names: list[str] = []
    matrix_rows: list[list[float]] = []
    first_beh = True
    for behavior in behaviors:
        deltas = compute_deltas(df, baseline_name="baseline_all_on", behavior=behavior)
        if deltas.empty or "ΔF1" not in deltas.columns:
            continue
        mask = deltas["Run"].str.startswith("without_")
        deltas = deltas[mask].copy()
        if deltas.empty:
            continue
        deltas["Toggle"] = (
            deltas["Run"]
            .str.replace("without_", "", n=1)
            .str.replace("_", " ")
            .str.title()
        )
        if first_beh:
            toggle_names = deltas["Toggle"].tolist()
            matrix_rows = [[] for _ in toggle_names]
            first_beh = False
        for t_idx, toggle in enumerate(toggle_names):
            row_match = deltas[deltas["Toggle"] == toggle]
            val = float(row_match["ΔF1"].iloc[0]) if not row_match.empty else 0.0
            if t_idx < len(matrix_rows):
                matrix_rows[t_idx].append(val)

    if not toggle_names or not matrix_rows or not matrix_rows[0]:
        return None

    mat = np.array(matrix_rows)
    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(behaviors)), max(3, 0.6 * len(toggle_names))))

    vmax = max(0.01, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels([b[:16] for b in behaviors], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(toggle_names)))
    ax.set_yticklabels(toggle_names, fontsize=9)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white",
            )

    ax.set_title("ΔF1 Heatmap — Effect of Removing Each Feature", fontsize=11)
    fig.colorbar(im, ax=ax, label="ΔF1", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── Save all plots ────────────────────────────────────────────────────

def save_all_plots(
    results: list[RunResult],
    output_dir: Path,
    behavior_names: dict[str, str] | None = None,
) -> list[Path]:
    """Generate and save all standard ablation plots. Returns paths of saved files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Aggregate plots (all behaviors together)
    bar_path = output_dir / "ablation_metric_bars.png"
    if metric_bar_chart(results, bar_path, behavior_names=behavior_names):
        saved.append(bar_path)

    delta_path = output_dir / "ablation_feature_impact.png"
    if delta_impact_chart(results, delta_path, behavior_names=behavior_names):
        saved.append(delta_path)

    heatmap_path = output_dir / "ablation_delta_heatmap.png"
    if delta_heatmap(results, heatmap_path, behavior_names=behavior_names):
        saved.append(heatmap_path)

    # Per-behavior PR curves and confusion matrices
    behaviors = sorted({r.behavior for r in results})
    for behavior in behaviors:
        beh_results = [r for r in results if r.behavior == behavior]
        short = behavior_names.get(behavior, behavior[:16]) if behavior_names else behavior[:16]
        tag = short.replace(" ", "_") if short else "all"

        pr_path = output_dir / f"ablation_pr_curves_{tag}.png"
        if pr_curves_overlay(beh_results, pr_path, behavior_names=behavior_names):
            saved.append(pr_path)

        cm_path = output_dir / f"ablation_confusion_matrices_{tag}.png"
        if confusion_matrix_grid(beh_results, cm_path, behavior_names=behavior_names):
            saved.append(cm_path)

    plt.close("all")
    return saved
