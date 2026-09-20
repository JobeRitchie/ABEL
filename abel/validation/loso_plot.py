"""Publication figure + raw-data export for leave-one-mouse-out (LOSO) CV.

The LOSO run (:mod:`abel.validation.loso`) trains one model per subject, each mouse
held out once. This module turns that per-behavior result list into the standard
cross-validation figure: one group of bars per behavior, showing **PR-AUC** and
**target-class F1**, both pooled over every mouse's held-out predictions, with a
**subject-level bootstrap 95% CI** on F1 (mice are the unit of independence).

Two deliberate choices, because the obvious alternatives mislead here:

* **Pooled, not per-fold mean ± SEM.** :mod:`abel.validation.loso` marks the
  per-fold spread ``fold_sem_valid: False``, folds are neither equally sized nor
  independent draws of one quantity, so their SEM is descriptive scatter rather
  than an error bar.
* **Target-class F1, not macro.** Macro averages the behavior with "not the
  behavior", which is ~85% of the data and scores ~0.97 on its own; that floors a
  never-detects-anything model near 0.50 and lifts every real score above the
  truth.

PR-AUC (average precision) is the primary metric because it is threshold-free and,
unlike ROC-AUC, stays informative under the strong class imbalance typical of
behavior data (Saito & Rehmsmeier 2015, PLOS ONE).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:  # pragma: no cover - matplotlib is a hard dep in practice
    _HAS_MPL = False

# Shared with the ablation benchmark chart for a consistent look.
_PRAUC_COLOR = "#2196F3"  # blue
_F1_COLOR = "#4CAF50"     # green


def _ok_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only behaviors that produced scorable folds (no ``error`` key)."""
    return [r for r in (results or []) if not r.get("error")]


def loso_bar_chart(
    results: list[dict[str, Any]],
    save_path: Path | str | None = None,
) -> "Figure | None":
    """Grouped bar chart of pooled PR-AUC and target-class F1, one group per behavior.

    ``results`` is the list returned by
    :func:`abel.validation.loso.leave_one_subject_out_all`. Returns the matplotlib
    ``Figure`` (also saved to ``save_path`` when given), or ``None`` if matplotlib
    is unavailable or there is nothing scorable to plot.
    """
    if not _HAS_MPL:
        return None
    rows = _ok_results(results)
    if not rows:
        return None

    labels = [str(r.get("behavior_name", r.get("behavior_id", "?"))) for r in rows]
    pr_means = np.array([_finite(r.get("pooled_prauc")) for r in rows], dtype=float)
    f1_means = np.array([_finite(r.get("pooled_f1_target")) for r in rows], dtype=float)
    # Asymmetric bootstrap CI as matplotlib's (lower, upper) offsets, clipped at
    # zero so a degenerate replicate cannot draw a whisker through its own bar.
    f1_lo = np.array([_finite(r.get("boot_f1_target_lo")) for r in rows], dtype=float)
    f1_hi = np.array([_finite(r.get("boot_f1_target_hi")) for r in rows], dtype=float)
    f1_err = np.vstack([
        np.clip(f1_means - f1_lo, 0.0, None),
        np.clip(f1_hi - f1_means, 0.0, None),
    ])
    n_subj = [int(r.get("n_subjects", 0) or 0) for r in rows]

    x = np.arange(len(rows))
    width = 0.38

    fig = Figure(figsize=(max(7.0, 1.6 * len(rows) + 2.5), 5.0))
    ax = fig.subplots()
    # PR-AUC carries no whisker: the bootstrap CI is computed for F1 only, and no
    # interval is better than one borrowed from a different quantity.
    ax.bar(
        x - width / 2, pr_means, width,
        label="PR-AUC (pooled)", color=_PRAUC_COLOR, alpha=0.9,
    )
    ax.bar(
        x + width / 2, f1_means, width, yerr=f1_err, capsize=3,
        label="F1, target class (pooled)", color=_F1_COLOR, alpha=0.9,
        error_kw={"linewidth": 0.9},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score (pooled across held-out mice)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leave-one-mouse-out cross-validation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Annotate the subject count under each behavior so the figure is self-describing.
    ns = ", ".join(f"{lbl}: n={n}" for lbl, n in zip(labels, n_subj))
    caption = (
        "Each mouse held out once, predictions pooled; error bars = subject-level "
        f"bootstrap 95% CI on F1  ({ns})"
    )
    # A run restricted to a subset of mice must say so on the figure itself.
    excluded = sorted({str(s) for r in rows for s in (r.get("excluded_subjects") or [])})
    if excluded:
        caption += f"\nMice excluded from this run: {', '.join(excluded)}"
    fig.text(
        0.01, 0.005, caption,
        fontsize=7, color="#666666", ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06 if excluded else 0.03, 1, 1))

    if save_path is not None:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def loso_results_to_csv(results: list[dict[str, Any]], path: Path | str) -> Path:
    """Write per-behavior summary + per-fold rows for reproducibility.

    Emits one ``summary`` row per behavior (pooled PR-AUC and target-class F1 with
    its bootstrap CI, the descriptive per-fold mean ± SEM, and the pooled refined
    F1) and one ``fold`` row per held-out subject (that subject's F1 and PR-AUC).
    This is the raw data behind :func:`loso_bar_chart`.
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "row_type", "behavior", "subject", "n_subjects",
            "pooled_prauc", "pooled_f1_target",
            "f1_target_ci_lo", "f1_target_ci_hi",
            "prauc_mean", "prauc_sem", "f1_macro_mean", "f1_macro_sem",
            "fold_pr_auc", "fold_f1", "pooled_refined_f1_target",
            "refined_evaluable", "included_subjects", "excluded_subjects", "error",
        ])
        for r in results or []:
            behavior = str(r.get("behavior_name", r.get("behavior_id", "?")))
            excluded = _joined(r.get("excluded_subjects"))
            if r.get("error"):
                w.writerow([
                    "summary", behavior, "", "", "", "", "", "", "", "", "", "",
                    "", "", "", "", "", excluded, r["error"],
                ])
                continue
            w.writerow([
                "summary", behavior, "", r.get("n_subjects", 0),
                _num(r.get("pooled_prauc")), _num(r.get("pooled_f1_target")),
                _num(r.get("boot_f1_target_lo")), _num(r.get("boot_f1_target_hi")),
                _num(r.get("fold_prauc_mean")), _num(r.get("fold_prauc_sem")),
                _num(r.get("fold_f1_mean")), _num(r.get("fold_f1_sem")),
                "", "", _num(r.get("refined_target_f1")),
                str(bool(r.get("refined_evaluable", True))),
                _joined(r.get("subjects")), excluded, "",
            ])
            for f in r.get("folds", []):
                if "f1" not in f and "pr_auc" not in f:
                    continue  # skipped fold (no positives / degenerate)
                w.writerow([
                    "fold", behavior, f.get("subject", ""), "",
                    "", "", "", "", "", "", "", "",
                    _num(f.get("pr_auc")), _num(f.get("f1")), "", "", "", "", "",
                ])
    return path


def _finite(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def _joined(values: object) -> str:
    """Semicolon-joined subject list ('; ' would split under a CSV comma)."""
    if not values:
        return ""
    try:
        return "; ".join(str(v) for v in values)  # type: ignore[union-attr]
    except TypeError:
        return str(values)


def _num(value: object) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(v) else f"{v:.6f}"
