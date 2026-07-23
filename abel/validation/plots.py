"""Matplotlib figures for the validation platform.

Operates on the validation dataclasses / tidy frames (not benchmark RunResults).
Shares the benchmark palette + Agg backend + dpi=200 conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39",
]


def _colour(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _save(fig, save_path: Path | None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ── Learning curve (headline) ──────────────────────────────────────────────

# The selectable learning-curve views: key -> human-readable label.
LEARNING_CURVE_VIEWS: dict[str, str] = {
    "f1_prauc": "F1 & PR-AUC",
    "precision_recall": "Precision & Recall",
    "counts": "TP / FP / FN counts",
    "kappa": "Cohen's κ",
}


def _knee_marker(ax, lc, y: float = 0.02) -> None:
    if lc.knee_clips is not None and np.isfinite(lc.knee_clips):
        ax.axvline(lc.knee_clips, color="#555", linestyle="--", linewidth=1.2)
        ax.text(lc.knee_clips, y, f"  optimal ≈ {int(round(lc.knee_clips))} clips",
                rotation=90, va="bottom", ha="left", fontsize=8, color="#555")


def learning_curve_plot(lc, save_path: Path | None = None,
                        view: str = "f1_prauc") -> "Figure | None":
    """One learning-curve view vs. # positive clips.

    ``view`` selects which metrics are drawn:
    ``f1_prauc`` (default, with knee marker), ``precision_recall``, ``counts``
    (held-out TP/FP/FN), or ``kappa``.
    """
    if not _HAS_MPL or not lc.points:
        return None
    xs = np.array([p.n_clips_mean for p in lc.points], dtype=float)
    title = f"Learning curve — {lc.behavior_name} ({lc.project_id})"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if view == "counts":
        tp = np.array([p.tp_mean for p in lc.points], dtype=float)
        fp = np.array([p.fp_mean for p in lc.points], dtype=float)
        fn = np.array([p.fn_mean for p in lc.points], dtype=float)
        ax.plot(xs, tp, "-o", color=_colour(2), label="True positives", linewidth=2)
        ax.plot(xs, fp, "-s", color=_colour(1), label="False positives", linewidth=2)
        ax.plot(xs, fn, "-^", color=_colour(3), label="False negatives", linewidth=2)
        ax.set_ylabel("Held-out clips (mean across seeds)")
        ax.set_ylim(bottom=0)
        _knee_marker(ax, lc, y=ax.get_ylim()[1] * 0.02)
        ax.set_title(f"{title}\nConfusion counts on the fixed held-out set")
    elif view == "precision_recall":
        prec = np.array([p.precision_mean for p in lc.points], dtype=float)
        rec = np.array([p.recall_mean for p in lc.points], dtype=float)
        ax.plot(xs, prec, "-o", color=_colour(0), label="Precision", linewidth=2)
        ax.plot(xs, rec, "-s", color=_colour(4), label="Recall", linewidth=2)
        ax.set_ylabel("Score (held-out subjects)")
        ax.set_ylim(0, 1.05)
        _knee_marker(ax, lc)
        ax.set_title(title)
    elif view == "kappa":
        kappa = np.array([p.kappa_mean for p in lc.points], dtype=float)
        ax.plot(xs, kappa, "-o", color=_colour(5), label="Cohen's κ", linewidth=2)
        ax.set_ylabel("Cohen's κ (held-out subjects)")
        ax.set_ylim(0, 1.05)
        _knee_marker(ax, lc)
        ax.set_title(title)
    else:  # "f1_prauc"
        f1 = np.array([p.f1_mean for p in lc.points], dtype=float)
        f1ci = np.array([p.f1_ci for p in lc.points], dtype=float)
        pr = np.array([p.pr_auc_mean for p in lc.points], dtype=float)
        prci = np.array([p.pr_auc_ci for p in lc.points], dtype=float)
        ax.plot(xs, f1, "-o", color=_colour(0), label="F1", linewidth=2)
        ax.fill_between(xs, f1 - f1ci, f1 + f1ci, color=_colour(0), alpha=0.18)
        ax.plot(xs, pr, "-s", color=_colour(2), label="PR-AUC", linewidth=2)
        ax.fill_between(xs, pr - prci, pr + prci, color=_colour(2), alpha=0.18)
        ax.set_ylabel("Score (held-out subjects)")
        ax.set_ylim(0, 1.05)
        _knee_marker(ax, lc)
        ax.set_title(title)

    ax.set_xlabel("# labeled positive clips")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ── Active learning vs. random ─────────────────────────────────────────────

def al_vs_random_plot(al_result, save_path: Path | None = None) -> "Figure | None":
    """Two panels: F1 vs. clips reviewed (AL vs random) and positives discovered.

    The left panel is the headline — active learning reaching a target F1 with
    fewer reviewed clips.  The right panel shows *why*: uncertainty sampling
    surfaces the rare positive clips faster than random review.
    """
    if not _HAS_MPL or (not al_result.al_points and not al_result.random_points):
        return None

    def _xy(points, attr):
        xs = np.array([p.n_clips for p in points], dtype=float)
        ys = np.array([getattr(p, attr) for p in points], dtype=float)
        return xs, ys

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: F1 vs clips reviewed
    for pts, name, col in (
        (al_result.al_points, "Active learning", _colour(0)),
        (al_result.random_points, "Random selection", _colour(1)),
    ):
        if not pts:
            continue
        xs, f1 = _xy(pts, "f1_mean")
        ci = np.array([p.f1_ci for p in pts], dtype=float)
        ax1.plot(xs, f1, "-o", color=col, linewidth=2, label=name)
        ax1.fill_between(xs, f1 - ci, f1 + ci, color=col, alpha=0.16)
    # Mark the clips-to-target advantage.
    al_n = al_result.clips_to_target(al_result.al_points)
    rnd_n = al_result.clips_to_target(al_result.random_points)
    if al_n is not None:
        ax1.axvline(al_n, color=_colour(0), linestyle=":", linewidth=1.2)
    if rnd_n is not None:
        ax1.axvline(rnd_n, color=_colour(1), linestyle=":", linewidth=1.2)
    sub = ""
    if al_n is not None and rnd_n is not None and al_n < rnd_n:
        sub = f"\n95% of peak F1: {int(al_n)} clips (AL) vs {int(rnd_n)} (random)"
    ax1.set_xlabel("# clips reviewed (labeling effort)")
    ax1.set_ylabel("F1 (held-out subjects)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Active learning vs. random — {al_result.behavior_name}{sub}", fontsize=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: positives discovered vs clips reviewed
    for pts, name, col in (
        (al_result.al_points, "Active learning", _colour(0)),
        (al_result.random_points, "Random selection", _colour(1)),
    ):
        if not pts:
            continue
        xs, npos = _xy(pts, "n_pos_mean")
        ax2.plot(xs, npos, "-o", color=col, linewidth=2, label=name)
    ax2.plot([0, max((p.n_clips for p in al_result.random_points), default=1)],
             [0, 0], alpha=0)  # keep origin
    ax2.set_xlabel("# clips reviewed")
    ax2.set_ylabel("Positive clips discovered")
    ax2.set_title("Label efficiency: positives surfaced", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, save_path)


# ── Ablation feature impact ────────────────────────────────────────────────

# Above this many pooled behaviors, grouped bars stop fitting on a page and
# :func:`ablation_impact_plot` falls back to the summary + matrix layout.
_ABLATION_BAR_LIMIT = 4


def _ablation_matrix_plot(abl_results: list, names: list[str], label_for, save_path,
                          budget_title: str | None) -> "Figure | None":
    """Pooled-ablation layout: headline summary bars + a behavior×config effect matrix.

    Left: the mean ΔF1 each enhancement buys, averaged over every behavior, with a
    95% CI *across behaviors* — the one number per feature the manuscript quotes.
    Right: the same gains per behavior, so a feature that helps on average but hurts
    one behavior can't hide inside the mean.  Cells are annotated only where the
    per-seed CI excludes zero (bold ✦); everything else is within baseline noise and
    is left unannotated rather than being read as a real effect.
    """
    from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415
    from matplotlib.gridspec import GridSpec  # noqa: PLC0415

    # Rows grouped by project, behaviors alphabetical within a project.
    rows = sorted(abl_results, key=lambda r: (str(r.project_id), str(r.behavior_name)))
    n_row, n_col = len(rows), len(names)
    gains = np.full((n_row, n_col), np.nan)
    sig = np.zeros((n_row, n_col), dtype=bool)
    for i, r in enumerate(rows):
        for j, nme in enumerate(names):
            gains[i, j] = float(r.gain.get(nme, np.nan))
            sig[i, j] = bool(r.is_significant(nme))

    pretty = [label_for(n) for n in names]
    fig_h = max(4.2, 0.26 * n_row + 2.6)
    fig = plt.figure(figsize=(6.2 + 0.72 * n_col, fig_h))
    # Explicit margins rather than tight_layout: the colorbar axes lives in the
    # gridspec, which tight_layout cannot lay out (it warns and mis-sizes).
    gs = GridSpec(1, 3, figure=fig, width_ratios=[3.1, 4.4, 0.16], wspace=0.16,
                  left=0.20, right=0.91, top=1.0 - 1.15 / fig_h, bottom=1.35 / fig_h)
    ax_sum = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    # ── Left: mean gain per config, CI across behaviors.
    means = np.nanmean(gains, axis=0)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(gains, axis=0, ddof=1)
    n_ok = np.sum(np.isfinite(gains), axis=0)
    ci = np.where(n_ok > 1, 1.96 * sd / np.sqrt(np.maximum(n_ok, 1)), 0.0)
    y = np.arange(n_col)
    # A config whose across-behavior CI clears zero is a feature that helps in
    # general, not just somewhere.
    solid = np.abs(means) > ci
    ax_sum.barh(y, means, 0.66, xerr=ci,
                color=[("#2E7D32" if m >= 0 else "#C62828") for m in means],
                alpha=1.0, edgecolor="white", linewidth=0.4,
                error_kw={"elinewidth": 0.9, "ecolor": "#455A64", "capsize": 2.5})
    for patch, s in zip(ax_sum.patches, solid):
        patch.set_alpha(0.92 if s else 0.3)
    ax_sum.axvline(0, color="#546E7A", linewidth=1.0)
    # Pad the axis *before* annotating: the value labels sit just past each bar's
    # error bar, and without headroom the long bars' labels fall outside the axes
    # (clipped) while the negative bars' labels land on top of the y tick labels.
    ends = np.concatenate([means - ci, means + ci])
    ends = ends[np.isfinite(ends)]
    if ends.size:
        lo_e, hi_e = float(ends.min()), float(ends.max())
        span = max(hi_e - lo_e, 1e-4)
        ax_sum.set_xlim(min(lo_e, 0.0) - 0.30 * span, max(hi_e, 0.0) + 0.30 * span)
    pad = 0.02 * (ax_sum.get_xlim()[1] - ax_sum.get_xlim()[0])
    for yi, m, c in zip(y, means, ci):
        if np.isfinite(m):
            ax_sum.text(m + (c + pad) * (1 if m >= 0 else -1), yi, f"{m:+.3f}",
                        va="center", ha="left" if m >= 0 else "right", fontsize=7.5,
                        color="#263238")
    ax_sum.set_yticks(y)
    ax_sum.set_yticklabels(pretty, fontsize=8.5)
    ax_sum.invert_yaxis()
    ax_sum.set_xlabel(f"mean ΔF1 vs. pose-only\n(95% CI across the {n_row} behaviors)",
                      fontsize=8.5)
    ax_sum.set_title("Average gain per enhancement", fontsize=9.5, loc="left")
    ax_sum.tick_params(axis="x", labelsize=7.5)
    ax_sum.grid(axis="x", alpha=0.22)
    for side in ("top", "right"):
        ax_sum.spines[side].set_visible(False)

    # ── Right: per-behavior matrix, diverging about 0.
    finite = gains[np.isfinite(gains)]
    lim = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 0.1
    lim = max(lim, 1e-3)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax_mat.imshow(np.ma.masked_invalid(gains), aspect="auto", cmap="RdBu_r",
                       norm=norm, interpolation="nearest")
    ax_mat.set_xticks(range(n_col))
    ax_mat.set_xticklabels(pretty, rotation=35, ha="right", fontsize=8)
    ax_mat.set_yticks(range(n_row))
    ax_mat.set_yticklabels([str(r.behavior_name)[:20] for r in rows], fontsize=7.5)
    ax_mat.set_facecolor("#ECEFF1")
    for i in range(n_row):
        for j in range(n_col):
            if sig[i, j] and np.isfinite(gains[i, j]):
                ax_mat.text(j, i, "✦", ha="center", va="center", fontsize=5.5,
                            color="#212121", alpha=0.75)

    # Project bands: separator + a label just outside the right edge of the matrix.
    # clip_on=False is required — the label sits past the last column, i.e. outside
    # the axes' data limits, and would otherwise be silently clipped away.
    proj_seq = [str(r.project_id) for r in rows]
    for start, end, pid in _contiguous_runs(proj_seq):
        if start > 0:
            ax_mat.axhline(start - 0.5, color="#37474F", linewidth=1.1)
        ax_mat.text(n_col - 0.35, (start + end - 1) / 2, str(pid)[:18], rotation=-90,
                    va="center", ha="center", fontsize=6.5, color="#37474F",
                    fontweight="bold", clip_on=False)
    ax_mat.set_title("Gain per behavior  (✦ = CI excludes 0)", fontsize=9.5, loc="left")

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("ΔF1 vs. pose-only baseline", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    btxt = f"  @ {budget_title}" if budget_title else ""
    fig.suptitle(f"Feature / pipeline ablation{btxt} — gain over the pose-only baseline",
                 fontsize=11.5, y=0.995)
    return _save(fig, save_path)


def ablation_impact_plot(abl_results: list, save_path: Path | None = None,
                         budget_title: str | None = None) -> "Figure | None":
    """Horizontal grouped bars of F1 gain over the pose-only baseline, with 95% CIs.

    Each bar is one enhancement added on its own (plus a final "all enhancements"
    bar), measured as the paired ``config F1 − baseline F1`` across seeds.  Error
    bars are the 95% CI of that paired difference; bars whose CI crosses zero are
    faded — they are *not* distinguishable from the baseline (so a small negative
    value is noise, not evidence the feature hurts).  Behaviors are side-by-side.
    """
    if not _HAS_MPL or not abl_results:
        return None
    # Config names (excluding the baseline) in build order, unioned across results.
    names: list[str] = []
    for r in abl_results:
        for n in r.order:
            if n in r.gain and n not in names:
                names.append(n)
    if not names:
        return None

    def label_for(n: str) -> str:
        for r in abl_results:
            if n in r.labels:
                return r.labels[n].replace("+ ", "")
        return n

    # Side-by-side bars scale the figure as configs × behaviors, so pooling every
    # project's behaviors into one chart produces a metres-tall strip. Past a
    # handful of behaviors, switch to the summary + effect-size matrix layout,
    # which stays one page regardless of how many behaviors are pooled.
    if len(abl_results) > _ABLATION_BAR_LIMIT:
        return _ablation_matrix_plot(abl_results, names, label_for, save_path, budget_title)

    pretty = [label_for(n) for n in names]
    n_beh = len(abl_results)
    bar_h = 0.8 / max(1, n_beh)
    y = np.arange(len(names), dtype=float)
    multi_project = len({r.project_id for r in abl_results}) > 1
    any_sig = False

    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.55 * len(names) * n_beh + 1.4)))
    for bi, r in enumerate(abl_results):
        vals = np.array([float(r.gain.get(n, np.nan)) for n in names])
        errs = np.array([float(r.gain_ci.get(n, 0.0)) for n in names])
        sig = [r.is_significant(n) for n in names]
        any_sig = any_sig or any(sig)
        offset = (bi - n_beh / 2 + 0.5) * bar_h
        series_label = (f"{r.project_id[:12]} · {r.behavior_name[:16]}"
                        if multi_project else r.behavior_name[:20])
        # Significant bars solid; non-significant faded (within noise of baseline).
        alphas = [0.9 if s else 0.32 for s in sig]
        ax.barh(y + offset, vals, bar_h, xerr=errs,
                color=_colour(bi), label=series_label, edgecolor="white", linewidth=0.3,
                error_kw={"elinewidth": 0.8, "ecolor": "#444", "capsize": 2})
        # Per-bar alpha (barh takes a single alpha, so recolour each patch).
        for patch, a in zip(ax.patches[-len(names):], alphas):
            patch.set_alpha(a)

    ax.axvline(0, color="gray", linewidth=0.9)
    ax.text(0, -0.6, "baseline (pose only)", va="center", ha="center",
            fontsize=8, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ΔF1 vs. pose-only baseline  (positive ⇒ feature improves accuracy)")

    btxt = f" @ {budget_title}" if budget_title else ""
    if len(abl_results) == 1:
        r = abl_results[0]
        base = r.baseline_f1
        allf = r.f1_means.get("all_features", float("nan"))
        sub = (f"\nbaseline F1 = {base:.3f}"
               + (f"   ·   all-enhancements F1 = {allf:.3f}" if np.isfinite(allf) else ""))
        ax.set_title(f"Feature ablation{btxt} — {r.behavior_name} ({r.project_id}){sub}",
                     fontsize=11)
    else:
        ax.set_title(f"Feature / pipeline ablation{btxt} — gain over pose-only baseline",
                     fontsize=11)

    # Interpretation note: error bars = 95% CI; faded = not distinguishable from baseline.
    note = ("Error bars: 95% CI across seeds.  Faded bars overlap 0 — "
            "not distinguishable from baseline (a small ± here is noise, not harm).")
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.5, color="#666")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, save_path)


# ── Cross-project dashboards ───────────────────────────────────────────────

def cross_project_bars(summary_df: pd.DataFrame, value_col: str, ci_col: str | None,
                       label_col: str, title: str, xlabel: str,
                       save_path: Path | None = None) -> "Figure | None":
    if not _HAS_MPL or summary_df is None or summary_df.empty:
        return None
    df = summary_df.copy()
    labels = df[label_col].astype(str).tolist()
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    errs = pd.to_numeric(df[ci_col], errors="coerce").to_numpy() if ci_col and ci_col in df else None
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(labels) + 1)))
    ax.barh(y, vals, xerr=errs, color=_colour(0), alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# κ interpretation landmarks (Landis & Koch): the reader should be able to grade a
# bar without a lookup table, so they are drawn as reference lines on every panel.
_KAPPA_BANDS: list[tuple[float, str]] = [(0.6, "substantial"), (0.8, "almost perfect")]


def cross_project_forest(
    by_behavior: pd.DataFrame,
    save_path: Path | None = None,
    *,
    metric: str = "f1",
    ceiling: float = float("nan"),
) -> "Figure | None":
    """THE headline meta-analysis figure: every (project × behavior) on one axis.

    A bar chart of 3-6 project means is nearly empty and — because every project
    lands in 0.91-0.94 on a 0-1 axis — conveys nothing.  The unit of evidence is
    really the **(project, behavior) pair**: ~25-30 rows for six assays.  This
    draws each as a point with its 95% CI, grouped and coloured by project, worst
    first within each block, with a **pooled mean diamond per project** and one
    grand diamond — i.e. the actual meta-analytic claim.

    ``by_behavior`` is the frame from
    :func:`cross_project.accuracy_by_behavior` (columns ``project_id``,
    ``behavior_name``, ``<metric>_mean``, ``<metric>_ci``, ``n``).  The x-axis is
    clipped to the populated range rather than 0-1, so differences are visible.
    """
    if not _HAS_MPL or by_behavior is None or by_behavior.empty:
        return None
    mcol, cicol = f"{metric}_mean", f"{metric}_ci"
    if mcol not in by_behavior.columns:
        return None
    df = by_behavior.dropna(subset=[mcol]).copy()
    if df.empty:
        return None
    df[cicol] = pd.to_numeric(df.get(cicol, 0.0), errors="coerce").fillna(0.0)

    projects = sorted(df["project_id"].astype(str).unique())
    rows: list[tuple] = []          # (label, mean, ci, colour, is_pooled)
    for pi, pid in enumerate(projects):
        sub = df[df["project_id"].astype(str) == pid].sort_values(mcol)
        for _, r in sub.iterrows():
            rows.append((str(r["behavior_name"])[:22], float(r[mcol]), float(r[cicol]),
                         _colour(pi), False))
        vals = sub[mcol].to_numpy(dtype=float)
        rows.append((f"{pid}  (pooled, {len(sub)} behaviors)", float(np.mean(vals)),
                     float(np.std(vals, ddof=1) / np.sqrt(len(vals))) * 1.96
                     if len(vals) > 1 else 0.0, _colour(pi), True))
    grand = df[mcol].to_numpy(dtype=float)
    rows.append((f"ALL PROJECTS  ({len(grand)} project×behavior)", float(np.mean(grand)),
                 float(np.std(grand, ddof=1) / np.sqrt(len(grand))) * 1.96
                 if len(grand) > 1 else 0.0, "#212121", True))

    n = len(rows)
    y = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(3.5, 0.30 * n + 1.6)))

    lo_all = min(m - c for _, m, c, _, _ in rows)
    xmin = max(0.0, min(0.5, lo_all - 0.05))

    for i, (label, mean, ci, col, pooled) in enumerate(rows):
        if pooled:
            # Diamond = pooled estimate, the meta-analytic summary for that block.
            ax.plot([mean - ci, mean + ci], [i, i], color=col, linewidth=2.0, zorder=3)
            ax.plot([mean], [i], marker="D", markersize=9, color=col,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=4)
            ax.axhspan(i - 0.5, i + 0.5, color=col, alpha=0.07, zorder=0)
        else:
            ax.plot([mean - ci, mean + ci], [i, i], color=col, linewidth=1.4,
                    alpha=0.85, zorder=2)
            ax.plot([mean], [i], "o", markersize=5.2, color=col,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    if np.isfinite(ceiling):
        ax.axvline(ceiling, color="#B71C1C", linestyle="--", linewidth=1.3, zorder=1)
        ax.text(ceiling, -0.9, " human ceiling", color="#B71C1C", fontsize=7.5,
                va="bottom", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    for tick, (_, _, _, _, pooled) in zip(ax.get_yticklabels(), rows):
        if pooled:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(xmin, 1.005)
    ax.set_xlabel(f"Held-out {metric.upper()}  (point = behavior, diamond = pooled; "
                  "bars = 95% CI)")
    ax.set_title("Does ABEL work across assays?\n"
                 "every project × behavior, with pooled estimates", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _save(fig, save_path)


def pool_generalization_by_behavior(gen_results: list) -> pd.DataFrame:
    """Generalization κ per (assay, behavior) — one row each, never pooled by name.

    Every behavior is scoped to its assay/project: ``EPM``'s Rear and ``OFT``'s Rear
    are *different* models and stay two rows, keyed ``"<project> · <behavior>"``. We
    do NOT merge same-named behaviors across projects — a "Rear" model trained in one
    assay says nothing about a "Rear" model trained in another, and averaging them
    would fabricate a behavior that no single model represents.

    Aggregation is over the *seed cells* within one project × behavior (one cell per
    seed), so the κ is the seed mean and the CI reflects seed noise. Returns columns:
    project, behavior, label, kappa, kappa_ci, f1, n_cells, human_ceiling_kappa.
    """
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    pooled: dict[tuple[str, str], dict] = {}
    for r in gen_results:
        key = (str(r.project_id), str(r.behavior_name))
        acc = pooled.setdefault(key, {"kappas": [], "f1s": [], "ceilings": []})
        cells = [c for c in (getattr(r, "cells", None) or [])
                 if not c.error and np.isfinite(c.cohen_kappa)]
        if cells:
            acc["kappas"].extend(float(c.cohen_kappa) for c in cells)
            acc["f1s"].extend(float(c.f1) for c in cells if np.isfinite(c.f1))
        elif np.isfinite(r.kappa_mean):
            # No retained cells (shouldn't happen) — fall back to the summary.
            acc["kappas"].append(float(r.kappa_mean))
            if np.isfinite(r.f1_mean):
                acc["f1s"].append(float(r.f1_mean))
        ceiling = float(r.human_ceiling_kappa)
        if np.isfinite(ceiling):
            acc["ceilings"].append(ceiling)

    rows = []
    for (proj, name), acc in pooled.items():
        ks, f1s, ceils = acc["kappas"], acc["f1s"], acc["ceilings"]
        rows.append({
            "project": proj,
            "behavior": name,
            "label": f"{proj} · {name}",
            "kappa": float(np.mean(ks)) if ks else float("nan"),
            "kappa_ci": vmetrics.ci95(ks) if ks else 0.0,
            "f1": float(np.mean(f1s)) if f1s else float("nan"),
            "n_cells": len(ks),
            "human_ceiling_kappa": float(np.mean(ceils)) if ceils else float("nan"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Worst-first: the behaviors that need interrogating are the ones you read first.
    return df.sort_values("kappa", ascending=True, na_position="first").reset_index(drop=True)


def human_ceiling_plot(gen_results: list, save_path: Path | None = None) -> "Figure | None":
    """Model κ per (assay, behavior) — ONE panel, one bar per behavior model.

    Every bar is a single assay's behavior model (see
    :func:`pool_generalization_by_behavior`); same-named behaviors from different
    assays are kept as separate bars, never merged. Bars are sorted by κ, weakest at
    the bottom, so the figure answers "how well does ABEL learn this behavior in this
    assay".

    ``human_ceiling_kappa`` is optional.  When no result carries one, the figure
    does not claim a comparison it cannot show: no ceiling series appears.
    """
    if not _HAS_MPL or not gen_results:
        return None

    df = pool_generalization_by_behavior(gen_results)
    if df.empty:
        return None
    has_ceiling = bool(np.isfinite(df["human_ceiling_kappa"]).any())

    y = np.arange(len(df))
    vals = df["kappa"].to_numpy(dtype=float)
    errs = df["kappa_ci"].to_numpy(dtype=float)
    # Colour carries the same grading as the reference lines, so the bar itself
    # already says "substantial / almost perfect" before you read the axis.
    colors = ["#C62828" if not np.isfinite(v) or v < 0.6
              else "#F9A825" if v < 0.8 else "#2E7D32" for v in vals]

    fig, ax = plt.subplots(figsize=(7.4, max(2.2, 0.42 * len(df) + 1.4)))
    ax.barh(y, vals, 0.68, xerr=errs if np.any(errs > 0) else None,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 0.9, "ecolor": "#37474F", "capsize": 2.5})
    for level, _lab in _KAPPA_BANDS:
        ax.axvline(level, color="#90A4AE", linestyle=":", linewidth=1.0, zorder=0)
    if has_ceiling:
        for yi, c in zip(y, df["human_ceiling_kappa"].to_numpy(dtype=float)):
            if np.isfinite(c):
                ax.vlines(c, yi - 0.34, yi + 0.34, color="#212121", linewidth=1.8)

    for yi, v, e in zip(y, vals, errs):
        if not np.isfinite(v):
            continue
        ax.text(v + e + 0.015, yi, f"{v:.2f}",
                va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels([str(b)[:34] for b in df["label"]], fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Cohen's κ (held-out subjects)", fontsize=9)
    ax.grid(axis="x", alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2E7D32", label="κ ≥ 0.80 (almost perfect)"),
        plt.Rectangle((0, 0), 1, 1, color="#F9A825", label="0.60 ≤ κ < 0.80 (substantial)"),
        plt.Rectangle((0, 0), 1, 1, color="#C62828", label="κ < 0.60 (moderate or worse)"),
    ]
    if has_ceiling:
        handles.append(plt.Line2D([0], [0], color="#212121", linewidth=1.8,
                                  label="Human inter-rater ceiling"))
    # Distinct projects in the RUN — not df['n_projects'].max(), which is the widest
    # single behavior and undercounts whenever no one behavior spans every project.
    n_proj = len({str(r.project_id) for r in gen_results})
    sub = (f"{len(df)} behavior model{'s' if len(df) != 1 else ''} "
           f"across {n_proj} assay{'s' if n_proj != 1 else ''}")
    ax.set_title("Generalization: model agreement with held-out human labels\n"
                 + sub, fontsize=11, loc="left")
    fig.legend(handles=handles, loc="lower center", ncol=2 if has_ceiling else 3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save_path)


# ── Held-out confusion counts by behavior ──────────────────────────────────

def _clip_unit(df: pd.DataFrame) -> str:
    """Name the counted unit, with its measured duration when the run agrees on one.

    ``clip_sec`` is per-project and projects genuinely differ (most use ~0.5 s but
    it is configurable), so a single duration is only printed when every row in
    the figure shares it — otherwise the axis would quietly attribute one assay's
    clip length to all of them.
    """
    if "clip_sec" not in df.columns:
        return "clips"
    secs = pd.to_numeric(df["clip_sec"], errors="coerce").dropna().round(2).unique()
    if len(secs) != 1 or secs[0] <= 0:
        return "clips"
    s = float(secs[0])
    return f"clips (~{s:.1f} s)" if s >= 0.1 else f"clips (~{s:.2f} s)"



def confusion_counts_by_behavior(conf_df: pd.DataFrame,
                                 save_path: Path | None = None) -> "Figure | None":
    """The counts behind the rates: found / missed / false-alarmed, per behavior.

    Consumes :func:`cross_project.confusion_by_behavior`.  One stacked horizontal
    bar per project·behavior with three segments — TP (found), FN (missed), FP
    (false alarms) — so the bar's length is the union of what the reviewer marked
    and what the model called, and the green share *is* the agreement.

    TN is deliberately **not** drawn.  Under this imbalance it is 10-100× the
    other three segments; including it would compress every bar into a sliver and
    imply an accuracy the positives do not support.  The axis label names the
    unit as reviewer-scored windows, because a count invites a reader to picture
    bouts, which these are not.
    """
    if not _HAS_MPL or conf_df is None or conf_df.empty:
        return None
    need = {"project_id", "behavior_name", "tp", "fp", "fn"}
    if not need <= set(conf_df.columns):
        return None

    df = conf_df.copy()
    for col in ("tp", "fp", "fn"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["tp", "fp", "fn"])
    if df.empty:
        return None
    # Worst recall at the top of the reading order. Matplotlib's y axis runs
    # bottom-up, so that is a *descending* sort; NaN recall (no held-out
    # positives at all) is the worst case of the lot and leads.
    pos = (df["tp"] + df["fn"]).replace(0, np.nan)
    # (na_position="last" = last row = highest y = top of the drawn figure.)
    df = df.assign(_recall=df["tp"] / pos).sort_values(
        "_recall", ascending=False, na_position="last")

    labels = [f"{r.project_id} · {r.behavior_name}" for r in df.itertuples()]
    y = np.arange(len(df))
    tp = df["tp"].to_numpy(float)
    fn = df["fn"].to_numpy(float)
    fp = df["fp"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.5, max(2.4, 0.42 * len(df) + 1.6)))
    ax.barh(y, tp, color="#4CAF50", label="Found (TP)")
    ax.barh(y, fn, left=tp, color="#FF9800", label="Missed (FN)")
    ax.barh(y, fp, left=tp + fn, color="#F44336", label="False alarm (FP)")

    span = float((tp + fn + fp).max()) or 1.0
    for i, (t, f_n, f_p) in enumerate(zip(tp, fn, fp)):
        ax.text(t + f_n + f_p + 0.01 * span, i,
                f"  {int(round(t))} / {int(round(t + f_n))} found"
                f"  ·  {int(round(f_p))} FP",
                va="center", fontsize=8, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, span * 1.42)
    ax.set_xlabel(f"Held-out {_clip_unit(df)} scored by the reviewer (not bouts)")
    ax.set_title("What the model got right, missed, and over-called\n"
                 "Bar = reviewer positives + the model's extra calls; "
                 "true negatives omitted", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    # Below the axes, not inside them: the per-bar annotations run out into the
    # right margin, which is exactly where an in-axes legend would sit. The strip
    # reserved for it is a fixed 0.5 in, converted to the figure fraction the
    # rect wants, because the figure's height grows with the behavior count.
    strip = 0.5 / float(fig.get_figheight())
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, strip, 1, 1))
    return _save(fig, save_path)


# ── PR curve + confusion from retained arrays ──────────────────────────────

def pr_curve(labelled_results: list[tuple[str, object]], title: str,
             save_path: Path | None = None) -> "Figure | None":
    """labelled_results: list of (label, ConfigEvalResult) with y_true/y_score set."""
    if not _HAS_MPL or not labelled_results:
        return None
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plotted = 0
    for i, (label, r) in enumerate(labelled_results):
        if getattr(r, "y_true", None) is None or getattr(r, "y_score", None) is None:
            continue
        prec, rec, _ = precision_recall_curve(r.y_true, r.y_score)
        ax.plot(rec, prec, color=_colour(i), linewidth=2,
                label=f"{label} (AP={r.pr_auc:.3f})")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


# ── Pairwise behavior discrimination ───────────────────────────────────────

def discrimination_matrices(pair_results: list, save_path: Path | None = None,
                            feature_set: str = "pose_video") -> "Figure | None":
    """Behavior×behavior separability, and what a feature family adds to it.

    Left: ROC-AUC of telling each behavior pair apart using **pose only** — dark
    cells are the pairs the model conflates (0.5 = coin flip, 1.0 = perfectly
    separable). Right: the **change** in that AUC when ``feature_set`` is added,
    on a diverging scale — bright red cells are the pairs the feature family
    rescues.  Together they answer "which behaviors look alike, and which features
    fix that", pair by pair.
    """
    if not _HAS_MPL or not pair_results:
        return None
    from abel.validation.analyses import discrimination as disc  # noqa: PLC0415

    base = disc.separability_matrix(pair_results, feature_set=disc.BASELINE_FEATURE_SET)
    # Plot the *ceiling-corrected* gain: raw ΔAUC is all "+0.00" once the pose
    # baseline is at 0.99, which hides real improvements. Share-of-remaining-error
    # removed is the number that actually distinguishes the pairs.
    delta = disc.error_reduction_matrix(pair_results, feature_set=feature_set)
    if base.empty or base.shape[0] < 2:
        return None
    names = list(base.index)
    n = len(names)

    label = next((r.labels.get(feature_set, feature_set)
                  for r in pair_results if feature_set in r.labels), feature_set)

    size = max(5.5, min(0.62 * n + 3.4, 13.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * size, size))

    # Held-out n per pair, so a "hard" pair resting on 199 clips can be told from
    # one resting on 973 — it varies >3x within a project.
    n_hold = {frozenset({r.name_a, r.name_b}): (r.n_hold_a + r.n_hold_b)
              for r in pair_results if not r.error}
    # Pairs with no trained result must not look like pairs the baseline solved —
    # three different meanings were sharing one grey. This covers BOTH ways a pair
    # can lack a number: dropped by the max_pairs cap (absent from pair_results), and
    # selected but skipped for too few clips (present, with r.error set). The latter
    # used to render as plain grey, which on the left-hand AUC panel could only ever
    # be read as "already solved" — a reading that is impossible for an unscored pair.
    ran = {frozenset({r.name_a, r.name_b}) for r in pair_results if not r.error}

    def _ink(cmap, norm_val: float) -> str:
        """Pick ink by the cell's actual luminance, not a guess about the colormap.

        The old heuristic assumed a colormap dark at both ends and put white text on
        viridis's bright-yellow top — 1.26:1 contrast, i.e. invisible, on exactly the
        near-ceiling cells that dominate this figure.
        """
        r, g, b, _ = cmap(float(np.clip(norm_val, 0, 1)))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#212121" if lum > 0.55 else "white"

    b_arr = base.to_numpy(dtype=float)
    # Anchor the low end just under the worst pair so the spread among the
    # near-ceiling pairs stays visible (a fixed 0.5 floor paints them all one colour).
    lo = float(np.nanmin(b_arr)) if np.isfinite(b_arr).any() else 0.5
    vmin = max(0.5, min(0.95, lo - 0.02))
    cmap1 = plt.get_cmap("viridis")
    im1 = ax1.imshow(np.ma.masked_invalid(b_arr), cmap=cmap1, vmin=vmin, vmax=1.0)
    ax1.set_title("Pose-only separability (ROC-AUC)\n"
                  "dark = the model confuses this pair", fontsize=10)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label(f"A-vs-B ROC-AUC (scale starts at {vmin:.2f})", fontsize=8)

    # Data-driven symmetric scale. A hardcoded ±1 rendered the whole panel white
    # when the real values span −3% to +22%, leaving the figure with zero visual signal.
    d_arr = delta.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(d_arr))) if np.isfinite(d_arr).any() else 0.1
    lim = max(0.10, min(1.0, lim))
    cmap2 = plt.get_cmap("RdBu_r")
    im2 = ax2.imshow(np.ma.masked_invalid(d_arr), cmap=cmap2, vmin=-lim, vmax=lim)
    ax2.set_title(f"Error removed by {label}\n"
                  "red = this feature family disambiguates the pair", fontsize=10)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label(f"share of pose-only error removed by {label}", fontsize=8)

    for ax, arr, is_delta in ((ax1, b_arr, True), (ax2, d_arr, False)):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_facecolor("#eceff1")
        cmap = cmap1 if is_delta else cmap2
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # self-comparison: leave blank
                key = frozenset({names[i], names[j]})
                if key not in ran:
                    # Never trained — mark it, don't let it read as "solved".
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                               edgecolor="#B0BEC5", hatch="///",
                                               linewidth=0.0))
                    continue
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                if is_delta:  # left panel = AUC
                    txt = f"{v:.3f}"
                    nh = n_hold.get(key)
                    if nh:
                        txt += f"\nn={nh}"
                    norm = (v - vmin) / max(1e-9, 1.0 - vmin)
                else:         # right panel = error reduction
                    txt = f"{v:+.0%}"
                    norm = (v + lim) / (2 * lim)
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                        linespacing=0.95, color=_ink(cmap, norm))

    from matplotlib.patches import Patch  # noqa: PLC0415
    fig.legend(
        handles=[
            Patch(facecolor="#eceff1", edgecolor="#B0BEC5", hatch="///",
                  label="pair not run (past the max-pairs cap, or too few clips)"),
            Patch(facecolor="#eceff1", edgecolor="none",
                  label="already solved by the baseline (no error left to remove)"),
        ],
        loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )

    proj = pair_results[0].project_id if pair_results else ""
    fig.suptitle(f"Can ABEL tell these behaviors apart? — {proj}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, save_path)


def discrimination_gain_plot(pair_results: list, save_path: Path | None = None
                             ) -> "Figure | None":
    """Per-pair ΔROC-AUC over the pose-only baseline, one bar group per pair.

    Sorted so the pairs the pose baseline finds hardest sit at the top — the ones
    where a feature family has the most to prove.  Error bars are the 95% CI of the
    paired per-seed difference; bars whose CI crosses zero are faded (not
    distinguishable from pose-only).  The left-hand annotation carries each pair's
    baseline AUC, so a big Δ on an already-easy pair can't be mistaken for a rescue.
    """
    if not _HAS_MPL or not pair_results:
        return None
    from abel.validation.analyses import discrimination as disc  # noqa: PLC0415

    scored = [r for r in pair_results if not r.error and np.isfinite(r.baseline_auc)]
    # Drop pairs the baseline already solves perfectly: they contribute an empty row
    # of full height and nothing else (5 of 21 on Novel object).
    scored = [r for r in scored
              if any(np.isfinite(r.gain.get(k, np.nan)) and abs(r.gain[k]) > 1e-6
                     for k in r.gain)]
    if not scored:
        return None
    # Hardest pairs (lowest pose-only AUC) first.
    scored = sorted(scored, key=lambda r: r.baseline_auc)

    sets: list[str] = []
    for r in scored:
        for nme in r.order:
            if nme != disc.BASELINE_FEATURE_SET and nme in r.gain and nme not in sets:
                sets.append(nme)
    if not sets:
        return None

    n_pairs, n_sets = len(scored), len(sets)
    bar_h = 0.8 / n_sets
    y = np.arange(n_pairs, dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.52 * n_pairs * n_sets + 1.6)))
    for si, sname in enumerate(sets):
        vals = np.array([float(r.gain.get(sname, np.nan)) for r in scored])
        errs = np.array([float(r.gain_ci.get(sname, 0.0)) for r in scored])
        sig = [r.is_significant(sname) for r in scored]
        offset = (si - n_sets / 2 + 0.5) * bar_h
        label = next((r.labels.get(sname, sname) for r in scored if sname in r.labels), sname)
        # Fixed family colour, not _colour(si): the positional palette recoloured
        # "video" whenever a project lacked the context rung, so two figures in the
        # same report disagreed about which colour meant which modality.
        ax.barh(y + offset, vals, bar_h, xerr=errs, color=_family_colour(sname), label=label,
                edgecolor="white", linewidth=0.3,
                error_kw={"elinewidth": 0.8, "ecolor": "#444", "capsize": 2})
        for patch, s in zip(ax.patches[-n_pairs:], sig):
            patch.set_alpha(0.9 if s else 0.3)

    # Raw ΔAUC understates a gain made against an already-high baseline, so label
    # each significant bar with the share of the remaining error it removed.
    headline = VIDEO_SET if (VIDEO_SET := "pose_video") in sets else sets[0]
    xmax = max((abs(float(r.gain.get(s, 0.0) or 0.0)) + float(r.gain_ci.get(s, 0.0) or 0.0)
                for r in scored for s in sets), default=0.01)
    for yi, r in zip(y, scored):
        er = r.error_reduction(headline)
        if np.isfinite(er) and r.is_significant(headline) and abs(er) >= 0.05:
            g = float(r.gain.get(headline, np.nan))
            # Always annotate to the RIGHT of zero: a right-aligned label on a
            # negative bar ran back over the y-tick text and garbled it.
            ax.text(xmax * 1.04, yi, f"{er:+.0%} of error", va="center", ha="left",
                    fontsize=6.8, color="#37474F")

    ax.axvline(0, color="gray", linewidth=0.9)
    ax.set_yticks(y)
    # 3 dp: the rows are SORTED by this number, and at 2 dp six near-ceiling pairs
    # all printed "1.00" while being ordered differently — the labels contradicted
    # the ordering.
    ax.set_yticklabels([f"{r.pair_label}\n(pose AUC {r.baseline_auc:.3f})" for r in scored],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ ROC-AUC vs. pose-only  (positive ⇒ the feature family separates this pair better)")
    ax.set_title("What each feature family adds to telling behavior pairs apart\n"
                 "hardest pairs (lowest pose-only AUC) at the top", fontsize=11)
    note = ("Error bars: 95% CI of the paired per-seed difference.  Faded bars overlap 0 — "
            "not distinguishable from the pose-only baseline.  Annotations give the share of "
            "the baseline's remaining error removed (raw ΔAUC understates gains near ceiling).")
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.5, color="#666")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, save_path)


# ── Pooled discrimination landscape (every project in one figure) ──────────
#
# The per-project matrices above are the archive: a full run writes one per project
# PER add-on family, so a reader has to mentally join a dozen heatmaps to answer the
# question the analysis exists for — across every assay, which behavior pairs does
# pose already separate, which does it confuse, and which modality rescues the ones
# it confuses.  That is one scatter, and these two panels are it.

# Marker per project. Colour is spent on the feature family (the result), so the
# assay has to be carried by shape.
_PROJECT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")

# A pair the pose baseline separates perfectly has 1 − AUC == 0, which a log axis
# cannot place. Pin those to one decade below, inside the "already solved" band.
_MIN_POSE_ERROR = 1e-4

# Display clamps. error_reduction is unbounded below (a family CAN hurt), and one
# −4.0 outlier on a near-zero-headroom pair would flatten the informative 0–1 range.
_ER_LO, _ER_HI = -0.5, 1.0

_ALL_FEATURES_COLOUR = "#37474F"   # the union rung: neutral, not a modality
_UNRESCUED_COLOUR = "#B0BEC5"      # no family measurably helps this pair


def _family_colour(feature_set: str) -> str:
    """Colour for one discrimination feature family.

    Reuses the behaviorscape modality palette rather than this module's positional
    ``_colour(i)``: with a per-index palette the same family draws in a different
    colour whenever a project happens to lack one rung, and three figures in one
    report then disagree about what "video" looks like.
    """
    from abel.validation import features as vfeat  # noqa: PLC0415
    from abel.validation.analyses.behaviorscape import MODALITY_COLORS  # noqa: PLC0415

    modality = {
        "pose_only": vfeat.MODALITY_POSE,
        "pose_context": vfeat.MODALITY_CONTEXT,
        "pose_video": vfeat.MODALITY_VIDEO,
        "pose_social": vfeat.MODALITY_SOCIAL,
    }.get(str(feature_set))
    if modality is not None:
        return MODALITY_COLORS.get(modality, "#607D8B")
    return _ALL_FEATURES_COLOUR if feature_set == "all_features" else "#607D8B"


def _clip_for_display(values, lo: float, hi: float):
    """Clip to the drawable range, reporting which points had to be moved.

    Returns ``(clipped, was_clipped)`` so the caller can draw the moved points with
    a caret — a clamped point silently redrawn at the axis limit is a lie about
    where the datum sits.
    """
    arr = np.asarray(values, dtype=float)
    out_of_range = np.isfinite(arr) & ((arr < lo) | (arr > hi))
    return np.clip(arr, lo, hi), out_of_range


def _short_pair_label(row, limit: int = 34) -> str:
    """Just the pair — the assay is already carried by the marker shape.

    Prefixing the project made every label ~50 characters, and at 6pt two of them
    overprinted each other into an unreadable smear on the first render.
    """
    text = str(row["pair"])
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _annotate_points(ax, xs, ys, labels, order, *, max_labels: int = 8,
                     x_gap: float = 0.22, y_gap: float = 0.05) -> None:
    """Label points in ``order`` (best first), skipping ones that would overprint.

    Two guards, both learned from the first render of this figure: a label near the
    right edge is flipped to sit on the *left* of its point (otherwise it runs off
    the axes), and a label whose box would land on one already placed is dropped
    rather than drawn on top of it.  Proximity is judged in axes fractions, so it
    behaves the same on the log x-axis as the linear one.

    Call this AFTER the axis limits are final — it reads them to normalise.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    is_log = ax.get_xscale() == "log"
    lx0, lx1 = (np.log10(max(x0, 1e-12)), np.log10(max(x1, 1e-12))) if is_log else (x0, x1)

    def _norm(x, y):
        vx = np.log10(max(float(x), 1e-12)) if is_log else float(x)
        return ((vx - lx0) / max(1e-9, lx1 - lx0), (float(y) - y0) / max(1e-9, y1 - y0))

    placed: list[tuple[float, float]] = []
    for idx in order:
        if len(placed) >= max_labels:
            break
        if not (np.isfinite(xs[idx]) and np.isfinite(ys[idx])):
            continue
        nx, ny = _norm(xs[idx], ys[idx])
        if any(abs(nx - px) < x_gap and abs(ny - py) < y_gap for px, py in placed):
            continue
        flip = nx > 0.55
        ax.annotate(labels[idx], (xs[idx], ys[idx]), textcoords="offset points",
                    xytext=(-7 if flip else 7, 4), ha="right" if flip else "left",
                    fontsize=6.2, color="#37474F")
        placed.append((nx, ny))


def discrimination_landscape(disc_df: "pd.DataFrame", save_path: Path | None = None
                             ) -> "Figure | None":
    """The whole run's discrimination result in two panels, all projects pooled.

    **Left — the landscape.**  One point per behavior pair.  x is the pose-only error
    (``1 − ROC-AUC``) on a log axis, because most pairs sit above 0.98 AUC and a
    linear axis stacks them all on the wall.  y is the share of that error removed by
    the best *single* feature family, coloured by which family that was — grey where
    no family's paired gain clears its own CI.  Pairs with no headroom left are still
    drawn, as hollow dots in the shaded band: the wall of pose-solved pairs is a
    result, and dropping it would overstate how much the extra modalities matter.

    **Right — the volcano.**  One point per pair × family, so a pair that both video
    and context rescue is distinguishable from one only video rescues — information
    the left panel's "best family" necessarily collapses.  x is the same error-removed
    scale; y is ``−log10 p`` of the paired t-test on the per-seed ΔAUC.  The two axes
    are consistent by construction: dividing every paired seed difference by the
    constant ``1 − pose AUC`` rescales the effect without touching the t statistic, so
    this p IS the p of the error-reduction on the x-axis.

    Takes the tidy frame from
    :func:`abel.validation.analyses.discrimination.discrimination_rows` (not the
    ``PairResult`` objects) so it can be re-rendered from a finished run's CSV, and
    tested, without retraining anything.
    """
    if not _HAS_MPL or disc_df is None or getattr(disc_df, "empty", True):
        return None
    from abel.validation.analyses import discrimination as disc  # noqa: PLC0415

    df = disc_df.copy()
    for col in ("pose_only_auc", "error_reduction", "p_value", "n_holdout"):
        if col not in df.columns:
            return None  # pre-0.9.1 CSV without the landscape columns
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["pose_only_auc"].notna()]
    if df.empty:
        return None
    df["significant"] = df["significant"].astype(str).str.lower().eq("true")
    if "project_name" not in df.columns:
        df["project_name"] = df["project"].astype(str)

    projects = list(dict.fromkeys(df["project_name"].astype(str)))
    marker_of = {p: _PROJECT_MARKERS[i % len(_PROJECT_MARKERS)]
                 for i, p in enumerate(projects)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ── Panel A: difficulty × rescue ───────────────────────────────────────
    pairs = df[df["best_family"].astype(bool)] if "best_family" in df.columns \
        else df.iloc[0:0]
    # Pairs with no headroom never get a best_family row (error_reduction is NaN by
    # design), so recover them from the baseline rows — they are the pose-solved wall.
    base_rows = df[df["feature_set"] == disc.BASELINE_FEATURE_SET]
    rescued_keys = set(zip(pairs["project"].astype(str), pairs["pair"].astype(str)))
    solved = base_rows[[(p, q) not in rescued_keys for p, q in
                        zip(base_rows["project"].astype(str), base_rows["pair"].astype(str))]]

    def _pose_error(frame):
        return np.maximum(1.0 - frame["pose_only_auc"].to_numpy(dtype=float),
                          _MIN_POSE_ERROR)

    if not solved.empty:
        ax1.scatter(_pose_error(solved), np.zeros(len(solved)),
                    s=18, facecolor="none", edgecolor=_UNRESCUED_COLOUR, linewidth=0.9,
                    zorder=2)

    y_clip, y_moved = _clip_for_display(pairs["error_reduction"], _ER_LO, _ER_HI)
    x_vals = _pose_error(pairs)
    # Held-out n spans >3x within a single project; a pair resting on 199 clips must
    # not read as firmly as one resting on 973.
    n_hold = pairs["n_holdout"].to_numpy(dtype=float)
    sizes = 22.0 + 3.2 * np.sqrt(np.nan_to_num(n_hold, nan=0.0))
    for i, (_, row) in enumerate(pairs.iterrows()):
        sig = bool(row["significant"])
        colour = _family_colour(row["feature_set"]) if sig else _UNRESCUED_COLOUR
        ax1.scatter(x_vals[i], y_clip[i], s=sizes[i],
                    marker=("^" if y_moved[i] and y_clip[i] >= _ER_HI else
                            "v" if y_moved[i] else marker_of[str(row["project_name"])]),
                    facecolor=colour if sig else "none",
                    edgecolor=colour if sig else _UNRESCUED_COLOUR,
                    linewidth=0.9, alpha=0.85 if sig else 0.9, zorder=3)

    ax1.set_xscale("log")
    ax1.set_xlim(_MIN_POSE_ERROR * 0.7, 0.75)
    # Only reach below zero as far as a family actually hurt: a fixed -0.5 floor left
    # the bottom 40% of the panel empty on runs where nothing hurt.
    finite_y = y_clip[np.isfinite(y_clip)] if len(y_clip) else np.array([0.0])
    ax1.set_ylim(min(-0.06, float(finite_y.min()) - 0.06) if finite_y.size else -0.06,
                 _ER_HI + 0.08)
    ax1.axvspan(ax1.get_xlim()[0], disc.MIN_HEADROOM, color="#ECEFF1", zorder=0)
    ax1.axvline(disc.MIN_HEADROOM, color="#90A4AE", linewidth=0.9, linestyle=":")
    ax1.axhline(0, color="#90A4AE", linewidth=0.9)
    ax1.set_xlabel("Pose-only error, 1 − ROC-AUC  (right ⇒ pose confuses this pair)")
    ax1.set_ylabel("Share of that error removed by the best feature family")
    ax1.set_title("Discrimination landscape — every behavior pair, every assay\n"
                  "colour = the modality that disambiguates the pair", fontsize=10)
    ax1.text(float(np.sqrt(_MIN_POSE_ERROR * disc.MIN_HEADROOM)),
             ax1.get_ylim()[1] * 0.5, "pose alone already solves these pairs",
             fontsize=7, color="#78909C", va="center", ha="center", rotation=90)
    ax1.grid(alpha=0.25, which="both")

    # Label the pairs that carry the result: hard AND substantially rescued.
    if not pairs.empty:
        interest = x_vals * np.clip(y_clip, 0, None)
        labels_a = [_short_pair_label(row) for _, row in pairs.iterrows()]
        order_a = [int(i) for i in np.argsort(interest)[::-1]
                   if np.isfinite(interest[i]) and interest[i] > 0]
        _annotate_points(ax1, x_vals, y_clip, labels_a, order_a, max_labels=8)

    # ── Panel B: volcano ──────────────────────────────────────────────────
    vol = df[(df["feature_set"] != disc.BASELINE_FEATURE_SET)
             & df["p_value"].notna() & df["error_reduction"].notna()]
    n_untestable = int(((df["feature_set"] != disc.BASELINE_FEATURE_SET)
                        & (df["p_value"].isna() | df["error_reduction"].isna())).sum())
    ax2.axvline(0, color="#90A4AE", linewidth=0.9)
    ax2.set_xlim(_ER_LO - 0.06, _ER_HI + 0.06)
    if not vol.empty:
        vx, vx_moved = _clip_for_display(vol["error_reduction"], _ER_LO, _ER_HI)
        # p can underflow to 0 at 5 seeds; floor it so -log10 stays finite.
        vy = -np.log10(np.maximum(vol["p_value"].to_numpy(dtype=float), 1e-12))
        for i, (_, row) in enumerate(vol.iterrows()):
            ax2.scatter(vx[i], vy[i], s=34,
                        marker=("<" if vx_moved[i] and vx[i] <= _ER_LO else
                                ">" if vx_moved[i] else marker_of[str(row["project_name"])]),
                        facecolor=_family_colour(row["feature_set"]),
                        edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)

        ax2.set_ylim(bottom=0, top=max(2.0, float(np.nanmax(vy))) * 1.12)
        ax2.axhline(-np.log10(0.05), color="#607D8B", linestyle="--", linewidth=1.0)
        # BELOW its own line: the BH line sits just above it when few tests were run,
        # and a label growing upward had the dotted line struck through it.
        ax2.text(_ER_LO - 0.03, -np.log10(0.05), "p = 0.05", fontsize=6.8,
                 color="#607D8B", va="top", ha="left")
        # A full run runs 40-100 of these tests, so the bare 0.05 line expects a
        # handful of false positives by construction. BH is the honest second line.
        from abel.validation.metrics import benjamini_hochberg_threshold  # noqa: PLC0415

        bh = benjamini_hochberg_threshold(vol["p_value"].tolist())
        if np.isfinite(bh):
            ax2.axhline(-np.log10(bh), color="#37474F", linestyle=":", linewidth=1.0)
            # Opposite edge from the p=0.05 label: with few tests BH lands close to
            # 0.05 and two labels at the same x overprinted into a smear.
            ax2.text(_ER_HI + 0.04, -np.log10(bh), f"BH 5% FDR (p = {bh:.3g})",
                     fontsize=6.8, color="#37474F", va="bottom", ha="right")

        # Name the extremes on both sides — a family that HURTS a pair is a finding.
        # One label per PAIR, not per point: a pair rescued by video AND by the
        # all-features union printed its own name twice, side by side.
        labels_b, order_b, seen = [], [], set()
        for _, row in vol.iterrows():
            labels_b.append(_short_pair_label(row))
        for idx in np.argsort(np.abs(vx))[::-1]:
            key = str(vol.iloc[int(idx)]["pair"])
            if key in seen or vy[idx] < -np.log10(0.05):
                continue
            seen.add(key)
            order_b.append(int(idx))
        _annotate_points(ax2, vx, vy, labels_b, order_b, max_labels=6, x_gap=0.28)

    ax2.set_xlabel("Share of pose-only error removed  (left of 0 ⇒ the family hurts)")
    ax2.set_ylabel("−log₁₀ p  (paired t-test across seeds)")
    ax2.set_title("Effect vs. evidence — every pair × feature family\n"
                  "up = reproducible across seeds; right = large", fontsize=10)
    ax2.grid(alpha=0.25)

    # ── Shared legend: what the colours mean, what the shapes mean ────────
    from matplotlib.lines import Line2D  # noqa: PLC0415

    fams = [f for f in dict.fromkeys(df["feature_set"].astype(str))
            if f and f != disc.BASELINE_FEATURE_SET]
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=_family_colour(f), markeredgecolor="white",
                      label=disc.FEATURE_SET_LABELS.get(f, f).lstrip("+ ").strip())
               for f in fams]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor="none", markeredgecolor=_UNRESCUED_COLOUR,
                          label="no family measurably helps"))
    handles += [Line2D([], [], marker=marker_of[p], linestyle="none", markersize=6.5,
                       markerfacecolor="#78909C", markeredgecolor="white", label=p)
                for p in projects]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, max(2, len(handles) // 2 + 1)),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    note = ("Left: point size ∝ held-out clips; hollow = the best family's paired 95% CI "
            "includes zero. Shaded band = pairs with under "
            f"{disc.MIN_HEADROOM:.1%} pose-only error left, where a 'share removed' is "
            "division by noise and is not computed. "
            "Carets mark points clipped to the axis.")
    if n_untestable:
        note += (f"  {n_untestable} pair × family combination(s) omitted from the volcano: "
                 "no pose-only error left to remove, under 2 usable seeds, or identical "
                 "across every seed.")
    fig.text(0.5, -0.075, note, ha="center", va="top", fontsize=7, color="#666",
             wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, save_path)


# ── Biological readout: time-budget & bout-count agreement ─────────────────

def time_budget_plot(tb_result, save_path: Path | None = None) -> "Figure | None":
    """Model-vs-reviewer agreement on per-session behavior prevalence.

    Left: prevalence of the behavior among each held-out session's *reviewed
    segments*, model (y) vs. reviewer (x), on the identity line.  Right: the
    matching Bland-Altman panel, whose limits of agreement say whether the measure
    is usable for a single animal — the number that actually decides whether the
    model can replace the scorer, so it is promoted into the title rather than
    buried.  Points beyond the limits are labeled.

    The axes deliberately say "reviewed segments", not "time": ABEL's labeled set
    covers only a small, active-learning-selected slice of each session (see
    :mod:`abel.validation.analyses.time_budget`), so this is a prevalence
    agreement, not a time budget.  The median labeled coverage is printed on the
    figure so the caveat cannot be lost.
    """
    if not _HAS_MPL or tb_result is None:
        return None
    tf = np.asarray(tb_result.true_prevalence, dtype=float)
    pf = np.asarray(tb_result.pred_prevalence, dtype=float)
    labels = list(tb_result.unit_labels)
    good = np.isfinite(tf) & np.isfinite(pf)
    tf, pf = tf[good], pf[good]
    labels = [l for l, k in zip(labels, good) if k]
    if tf.size == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # Left: identity-line agreement scatter. Overplotted points (e.g. a pile at the
    # origin for a rare behavior) are sized by multiplicity so the reader sees them.
    hi = float(max(tf.max(), pf.max(), 0.05)) * 1.08
    ax1.plot([0, hi], [0, hi], color="#888", linestyle="--", linewidth=1.2,
             label="perfect agreement")
    pts, counts = np.unique(np.column_stack([tf, pf]), axis=0, return_counts=True)
    ax1.scatter(pts[:, 0], pts[:, 1], s=38 + 26 * (counts - 1), color=_colour(0),
                edgecolor="white", linewidth=0.6, zorder=3, alpha=0.9)
    if (counts > 1).any():
        ax1.plot([], [], "o", color=_colour(0), markersize=7,
                 label="larger dot = several sessions overlap")
    ax1.set_xlim(0, hi)
    ax1.set_ylim(0, hi)
    ax1.set_xlabel(f"Reviewer: share of reviewed segments scored {tb_result.behavior_name}")
    ax1.set_ylabel("Model: same share")
    bits = []
    for lbl, val in (("r", tb_result.prev_pearson_r), ("CCC", tb_result.prev_ccc),
                     ("R²", tb_result.prev_r2)):
        if np.isfinite(val):
            bits.append(f"{lbl} = {val:.3f}")
    bits.append(f"n = {tb_result.n_units} sessions")
    ax1.set_title(f"{tb_result.behavior_name} ({tb_result.project_id}) — "
                  f"per-session prevalence agreement\n" + "   ·   ".join(bits), fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.grid(alpha=0.3)

    # Right: Bland-Altman. The LoA width — not r — decides single-animal usability.
    mean_xy = (tf + pf) / 2.0
    diff = pf - tf
    bias = tb_result.prev_bias
    lo, up = tb_result.prev_loa_lower, tb_result.prev_loa_upper
    ax2.scatter(mean_xy, diff, s=40, color=_colour(0), edgecolor="white",
                linewidth=0.6, zorder=3)
    if np.isfinite(bias):
        ax2.axhline(bias, color=_colour(1), linewidth=1.4)
        blo, bhi = tb_result.prev_bias_ci
        if np.isfinite(blo) and np.isfinite(bhi):
            # 95% CI of the bias: if it straddles 0 there is no systematic offset.
            ax2.axhspan(blo, bhi, color=_colour(1), alpha=0.15, zorder=1)
    for yv, lab in ((lo, "−1.96 SD"), (up, "+1.96 SD")):
        if np.isfinite(yv):
            ax2.axhline(yv, color="#888", linestyle=":", linewidth=1.1)
    ax2.axhline(0, color="#bbb", linewidth=0.8)
    # Label the sessions outside the limits of agreement — the ones a reader will ask about.
    for x, d, lab in zip(mean_xy, diff, labels):
        if (np.isfinite(lo) and d < lo) or (np.isfinite(up) and d > up):
            ax2.annotate(str(lab)[:18], (x, d), fontsize=6, color="#546E7A",
                         xytext=(4, 3), textcoords="offset points")
    # Left-margin line labels (right margin collided with the legend).
    x0 = ax2.get_xlim()[0]
    for yv, lab in ((lo, "−1.96 SD"), (up, "+1.96 SD"), (bias, "bias")):
        if np.isfinite(yv):
            ax2.text(x0, yv, f" {lab}", fontsize=7, va="bottom", ha="left", color="#666")
    ax2.set_xlabel("Mean of model & reviewer prevalence")
    ax2.set_ylabel("Model − reviewer (prevalence)")
    width = tb_result.loa_width
    ttl = "Bland-Altman — can this be trusted for ONE animal?"
    if np.isfinite(width):
        ttl += (f"\nbias {bias:+.3f}   ·   95% limits of agreement "
                f"{lo:+.3f} to {up:+.3f}  (width {width:.3f})")
    ax2.set_title(ttl, fontsize=10)
    ax2.grid(alpha=0.3)

    cov = tb_result.median_coverage
    caveat = ("Prevalence is over REVIEWED segments, not session time — "
              "this is not a time budget.")
    if np.isfinite(cov):
        caveat += f"  Labeled segments cover a median {cov:.1%} of each session."
    fig.text(0.5, 0.005, caveat, ha="center", va="bottom", fontsize=7.5, color="#B71C1C")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, save_path)


def time_budget_forest(tb_results: list, save_path: Path | None = None) -> "Figure | None":
    """All behaviors' agreement on ONE panel — the figure that goes in the paper.

    Left: a forest plot of the per-session bias with its 95% CI (thick bar) and the
    95% limits of agreement (thin whiskers), one row per behavior.  The LoA — not
    r — decide whether the model can stand in for the scorer on a *single animal*,
    and putting every behavior on a shared axis makes an unusable one impossible to
    miss.  A behavior whose bias CI straddles 0 has no systematic offset; a behavior
    whose whiskers are wide is unreliable per-animal no matter how good its r looks.

    Right: Lin's CCC per behavior (agreement with the identity line), with n.
    Replaces N separate Bland-Altman panels with one comparable view.
    """
    if not _HAS_MPL or not tb_results:
        return None
    rs = [r for r in tb_results if r is not None and np.isfinite(r.prev_bias)]
    if not rs:
        return None
    # Widest limits of agreement (least usable) at the top — that is the risk.
    rs = sorted(rs, key=lambda r: (-(r.loa_width if np.isfinite(r.loa_width) else -1)))
    names = [f"{r.behavior_name}\n(n={r.n_units})" for r in rs]
    y = np.arange(len(rs), dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, max(2.8, 0.62 * len(rs) + 1.8)),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )

    for i, r in enumerate(rs):
        lo, hi = r.prev_loa_lower, r.prev_loa_upper
        blo, bhi = r.prev_bias_ci
        if np.isfinite(lo) and np.isfinite(hi):
            ax1.plot([lo, hi], [i, i], color="#90A4AE", linewidth=1.4, zorder=2,
                     solid_capstyle="butt")
            for x in (lo, hi):
                ax1.plot([x, x], [i - 0.16, i + 0.16], color="#90A4AE", linewidth=1.4, zorder=2)
        if np.isfinite(blo) and np.isfinite(bhi):
            ax1.plot([blo, bhi], [i, i], color=_colour(0), linewidth=5.0, alpha=0.85,
                     zorder=3, solid_capstyle="butt")
        ax1.plot([r.prev_bias], [i], "o", color=_colour(1), markersize=6.5,
                 markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    ax1.axvline(0, color="#555", linestyle="--", linewidth=1.1, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Model − reviewer, per-session prevalence")
    ax1.set_title("Agreement per behavior — bias, its 95% CI, and the limits of agreement\n"
                  "thin whiskers = 95% LoA (single-animal reliability) · thick bar = 95% CI of bias",
                  fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.plot([], [], "o", color=_colour(1), label="bias")
    ax1.plot([], [], color=_colour(0), linewidth=5, alpha=0.85, label="95% CI of bias")
    ax1.plot([], [], color="#90A4AE", linewidth=1.4, label="95% limits of agreement")
    ax1.legend(loc="lower right", fontsize=7.5, frameon=False)

    ccc = np.array([r.prev_ccc for r in rs], dtype=float)
    ax2.barh(y, ccc, 0.62, color=_colour(2), alpha=0.85, edgecolor="white", linewidth=0.4)
    for i, v in enumerate(ccc):
        if np.isfinite(v):
            ax2.text(min(v + 0.02, 1.02), i, f"{v:.2f}", va="center", fontsize=7.5,
                     color="#37474F")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Lin's CCC (agreement with identity)")
    ax2.set_title("Concordance", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    cov = np.nanmedian([r.median_coverage for r in rs])
    caveat = "Prevalence is over REVIEWED segments, not session time — this is not a time budget."
    if np.isfinite(cov):
        caveat += f"  Labeled segments cover a median {cov:.1%} of each session."
    fig.text(0.5, 0.005, caveat, ha="center", va="bottom", fontsize=7.5, color="#B71C1C")
    proj = rs[0].project_id
    fig.suptitle(f"Model vs. reviewer — per-session agreement ({proj})", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    return _save(fig, save_path)


def time_budget_grid(tb_results: list, save_path: Path | None = None) -> "Figure | None":
    """Small-multiples of the per-session identity scatters — one panel per behavior.

    One shared figure instead of one file per behavior, so the whole project reads
    as a single result. Each panel keeps its own axis range (behaviors differ by an
    order of magnitude in prevalence) but the identity line makes them comparable.
    """
    if not _HAS_MPL or not tb_results:
        return None
    rs = [r for r in tb_results if r is not None and r.true_prevalence]
    if not rs:
        return None
    n = len(rs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.7 * nrow), squeeze=False)

    for i, r in enumerate(rs):
        ax = axes[i // ncol][i % ncol]
        tf = np.asarray(r.true_prevalence, dtype=float)
        pf = np.asarray(r.pred_prevalence, dtype=float)
        good = np.isfinite(tf) & np.isfinite(pf)
        tf, pf = tf[good], pf[good]
        hi = float(max(tf.max(initial=0.0), pf.max(initial=0.0), 0.05)) * 1.1
        ax.plot([0, hi], [0, hi], color="#888", linestyle="--", linewidth=1.0)
        pts, cnts = np.unique(np.column_stack([tf, pf]), axis=0, return_counts=True)
        ax.scatter(pts[:, 0], pts[:, 1], s=26 + 20 * (cnts - 1), color=_colour(i % 10),
                   edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        bits = []
        if np.isfinite(r.prev_ccc):
            bits.append(f"CCC {r.prev_ccc:.2f}")
        if np.isfinite(r.loa_width):
            bits.append(f"LoA ±{r.loa_width / 2:.03f}")
        ax.set_title(f"{r.behavior_name}  (n={r.n_units})\n" + "  ·  ".join(bits), fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.supxlabel("Reviewer: share of reviewed segments in behavior", fontsize=9)
    fig.supylabel("Model: same share", fontsize=9)
    fig.suptitle(f"Per-session prevalence, model vs. reviewer — {rs[0].project_id}",
                 fontsize=12)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    return _save(fig, save_path)


# ── Probability calibration: reliability diagram ───────────────────────────

# Bins holding fewer than this many held-out samples are drawn hollow: their
# empirical rate is noise (a 1-sample bin reads as exactly 0.0 or 1.0).
MIN_RELIABLE_BIN_N = 10


def reliability_diagram(cal_result, save_path: Path | None = None) -> "Figure | None":
    """Reliability diagram + the bin-count histogram that makes it readable.

    Behavior-model probabilities are strongly **bimodal** — nearly every held-out
    segment scores close to 0 or close to 1, and the middle bins hold a handful of
    samples each.  Drawing those sparse bins as equal-weight vertices of a
    connected line produces a violent zigzag that screams "miscalibrated" while ECE
    reads ~0.01, because a single-sample bin necessarily lands on 0.0 or 1.0.

    So: bins are drawn as **bars on a fixed grid** (never connected), bins with
    fewer than :data:`MIN_RELIABLE_BIN_N` samples are drawn hollow and hatched, and
    a lower panel shows the sample count per bin on a log axis — the standard
    presentation, and the only way to see that the zigzag is 2% of the data.
    """
    if not _HAS_MPL or cal_result is None:
        return None
    from matplotlib.gridspec import GridSpec  # noqa: PLC0415

    curve = cal_result.curve
    if not curve.bin_center:
        return None
    centre = np.asarray(curve.bin_center, dtype=float)
    acc = np.asarray(curve.bin_accuracy, dtype=float)
    cnt = np.asarray(curve.bin_count, dtype=float)
    width = 1.0 / max(1, curve.n_bins)
    solid = cnt >= MIN_RELIABLE_BIN_N

    fig = plt.figure(figsize=(6.0, 6.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1.2, zorder=1,
            label="perfect calibration")
    # Well-populated bins: solid. Sparse bins: hollow + hatched (do not trust).
    ax.bar(centre[solid], acc[solid], width=width * 0.92, color=_colour(0), alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=2,
           label=f"bin with ≥{MIN_RELIABLE_BIN_N} samples")
    if (~solid).any():
        ax.bar(centre[~solid], acc[~solid], width=width * 0.92, facecolor="none",
               edgecolor=_colour(0), linewidth=0.9, hatch="///", alpha=0.7, zorder=2,
               label=f"bin with <{MIN_RELIABLE_BIN_N} samples (noise)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical positive rate")
    sub = (f"ECE = {curve.ece:.3f}   ·   MCE = {curve.mce:.3f}   ·   "
           f"Brier = {curve.brier:.3f}   ·   n = {curve.n}")
    ax.set_title(f"Reliability — {cal_result.behavior_name} ({cal_result.project_id})\n{sub}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelbottom=False)

    # Lower panel: where the samples actually are. Log scale because the extremes
    # outnumber the middle by ~2 orders of magnitude.
    axh.bar(centre, np.maximum(cnt, 0.6), width=width * 0.92, color="#90A4AE",
            edgecolor="white", linewidth=0.5)
    axh.axhline(MIN_RELIABLE_BIN_N, color=_colour(1), linestyle=":", linewidth=1.0)
    axh.text(0.995, MIN_RELIABLE_BIN_N, f" n={MIN_RELIABLE_BIN_N}", fontsize=6.5,
             color=_colour(1), va="bottom", ha="right")
    axh.set_yscale("log")
    axh.set_xlim(0, 1)
    axh.set_xlabel("Predicted probability of the behavior")
    axh.set_ylabel("samples\n(log)", fontsize=8)
    axh.grid(alpha=0.25, axis="y")

    mid = cnt[(centre > 0.1) & (centre < 0.9)].sum()
    if curve.n > 0:
        fig.text(0.5, 0.005,
                 f"Scores are bimodal: only {mid / curve.n:.1%} of held-out segments fall "
                 f"between p=0.1 and p=0.9. Hatched bars are too sparse to interpret.",
                 ha="center", va="bottom", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, save_path)


# ── Behaviorscape: feature-landscape figures ───────────────────────────────
#
# All four operate on a ``BehaviorscapeData`` (see
# :mod:`abel.validation.analyses.behaviorscape`): a feature×behavior importance
# matrix plus a ``feature -> modality`` map and the modality colour palette.


def _modality_of(data, feat: str) -> str:
    return data.modality.get(feat, data.modality_order[0])


def _dominant_modality_per_behavior(data) -> dict:
    """Behavior -> the modality holding the largest share of its importance."""
    frac = data.modality_fraction_by_behavior()  # behaviors × modalities
    out = {}
    for beh in frac.index:
        row = frac.loc[beh]
        out[beh] = str(row.idxmax()) if float(row.max()) > 0 else data.modality_order[0]
    return out


def _linkage(x: np.ndarray):
    """Average-linkage hierarchy over the rows of ``x`` (correlation distance,
    euclidean fallback).  Returns the SciPy linkage matrix, or ``None``."""
    if x.shape[0] < 2:
        return None
    try:
        from scipy.cluster.hierarchy import linkage  # noqa: PLC0415
        from scipy.spatial.distance import pdist  # noqa: PLC0415
    except Exception:
        return None
    d = None
    try:
        d = pdist(x, metric="correlation")
        if not np.all(np.isfinite(d)):
            d = None
    except Exception:
        d = None
    if d is None:
        try:
            d = pdist(x, metric="euclidean")
        except Exception:
            return None
    if d is None or not np.all(np.isfinite(d)):
        return None
    try:
        return linkage(d, method="average")
    except Exception:
        return None


def _leaf_order(z, n: int) -> list[int]:
    if z is None:
        return list(range(n))
    try:
        from scipy.cluster.hierarchy import leaves_list  # noqa: PLC0415
        return [int(i) for i in leaves_list(z)]
    except Exception:
        return list(range(n))


def _contiguous_runs(labels: list) -> list[tuple[int, int, object]]:
    """[(start, end_exclusive, label), ...] for contiguous equal-label runs."""
    runs: list[tuple[int, int, object]] = []
    if not labels:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def behaviorscape_heatmap(data, save_path: Path | None = None) -> "Figure | None":
    """Clustered feature×behavior importance heatmap with a modality side-strip.

    Behaviors (columns) are clustered by importance-profile similarity and a
    dendrogram is drawn above them, so neighbouring columns are genuinely
    similar.  Features (rows) are grouped into four contiguous **modality
    bands** (pose / kinematics / video / context) — labelled directly on the
    left strip — and clustered within each band.  The result reads as: which
    *kinds* of features (the bands) light up for which behaviors.
    """
    if not _HAS_MPL or data is None or data.is_empty():
        return None
    from matplotlib.colors import ListedColormap, PowerNorm  # noqa: PLC0415
    from matplotlib.gridspec import GridSpec  # noqa: PLC0415

    mat = data.matrix.to_numpy(dtype=float)
    features = list(data.matrix.index)
    behaviors = list(data.matrix.columns)

    # Columns: cluster + keep the linkage so we can draw the dendrogram.
    col_link = _linkage(mat.T)
    col_order = _leaf_order(col_link, len(behaviors))

    # Rows: group by modality (fixed order), cluster within each band.
    row_order: list[int] = []
    for m in data.modality_order:
        idx = [i for i, f in enumerate(features) if data.modality.get(f) == m]
        if not idx:
            continue
        sub_order = _leaf_order(_linkage(mat[idx]), len(idx)) if len(idx) >= 3 else range(len(idx))
        row_order.extend(idx[k] for k in sub_order)

    mat = mat[np.ix_(row_order, col_order)]
    features = [features[i] for i in row_order]
    behaviors = [behaviors[i] for i in col_order]
    row_mod = [data.modality.get(f, data.modality_order[0]) for f in features]

    n_feat, n_beh = len(features), len(behaviors)
    fig_h = max(5.0, min(0.02 * n_feat + 4.5, 18.0))
    fig_w = max(7.0, min(0.55 * n_beh + 4.2, 20.0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[0.42, 6.0, 0.22], height_ratios=[0.9, 6.0],
        wspace=0.03, hspace=0.02,
    )
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])
    ax_dendro = fig.add_subplot(gs[0, 1])

    # Column dendrogram.  SciPy places leaves at x = 10·i + 5 (range 0..10n);
    # the heatmap spans -0.5..n-0.5.  Both fill the same gridspec cell, so leaf
    # i lands at fraction (i+0.5)/n in each — they line up WITHOUT sharex (which
    # would otherwise collapse the heatmap into the dendrogram's 0..10n range).
    if col_link is not None:
        try:
            from scipy.cluster.hierarchy import dendrogram  # noqa: PLC0415
            dendrogram(col_link, ax=ax_dendro, color_threshold=0,
                       above_threshold_color="#90A4AE", no_labels=True)
            ax_dendro.set_xlim(0, 10 * len(behaviors))
        except Exception:
            pass
    ax_dendro.axis("off")

    # Modality strip with one band label per contiguous modality run.
    strip = np.array([[data.modality_order.index(m)] for m in row_mod])
    strip_cmap = ListedColormap([data.modality_colors[m] for m in data.modality_order])
    ax_strip.imshow(strip, aspect="auto", cmap=strip_cmap,
                    vmin=0, vmax=len(data.modality_order) - 1, interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for start, end, m in _contiguous_runs(row_mod):
        if end - start < max(2, n_feat * 0.015):
            continue  # too thin to label
        ax_strip.text(0, (start + end) / 2 - 0.5, data.modality_labels[m].split(" (")[0],
                      rotation=90, ha="center", va="center", fontsize=8.5,
                      fontweight="bold", color="white")
    ax_strip.set_ylabel("feature modality  (grouped, then clustered within group)",
                        fontsize=8.5)

    # Heatmap. PowerNorm lifts the many small importances so structure is visible
    # rather than a near-black field with a few bright streaks.
    pos_vals = mat[mat > 0]
    vmax = float(np.quantile(pos_vals, 0.98)) if pos_vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax_heat.imshow(mat, aspect="auto", cmap="magma",
                        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax))
    ax_heat.set_xticks(range(n_beh))
    ax_heat.set_xticklabels(behaviors, rotation=40, ha="right", fontsize=8)
    ax_heat.set_yticks([])
    # Thin separators between modality bands.
    for start, end, _m in _contiguous_runs(row_mod)[:-1]:
        ax_heat.axhline(end - 0.5, color="white", linewidth=0.8, alpha=0.7)

    cbar = fig.colorbar(im, cax=ax_cbar)
    norm_lbl = "share of model gain" if data.normalize == "fraction" else "importance"
    cbar.set_label(f"feature importance ({norm_lbl}, √-scaled)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    kept = data.n_features_kept or n_feat
    sub = (f"{kept} of {data.n_features_total} features "
           f"(kept if importance ≥ {data.threshold:g} in ≥1 project) · "
           f"columns clustered by profile similarity"
           if data.n_features_total else f"{kept} features")
    fig.suptitle(f"Behaviorscape — feature importance across behaviors\n{sub}",
                 fontsize=11, y=0.98)
    return _save(fig, save_path)


def behaviorscape_modality_bars(data, save_path: Path | None = None) -> "Figure | None":
    """Per-behavior stacked bars of importance share by data modality.

    Behaviors are grouped by their **dominant** modality (in the fixed pose →
    kinematics → video → context order) and, within each group, sorted by how
    strongly they rely on that dominant family (high → low).  Within each bar the
    four modality segments are ordered largest-first, so the strongest data
    source always sits at the left edge (colour still encodes modality).  Faint
    group bands + right-hand labels make the clusters explicit.
    """
    if not _HAS_MPL or data is None or data.is_empty():
        return None
    from matplotlib.patches import Patch  # noqa: PLC0415

    frac = data.modality_fraction_by_behavior()  # behaviors × modalities
    rank = {m: i for i, m in enumerate(data.modality_order)}
    dom = frac.idxmax(axis=1)
    dom_share = frac.max(axis=1)
    order = sorted(
        frac.index,
        key=lambda b: (rank.get(str(dom[b]), 99), -float(dom_share[b])),
    )
    frac = frac.loc[order]
    dom = dom.loc[order]
    behaviors = list(frac.index)

    n = len(behaviors)
    fig, ax = plt.subplots(figsize=(9.0, max(2.8, 0.42 * n + 1.6)))
    # Plot top-to-bottom in group order: invert so order[0] is at the top.
    y = np.arange(n)[::-1]
    for yi, beh in zip(y, behaviors):
        row = frac.loc[beh]
        left = 0.0
        # Fixed modality order — NOT largest-segment-first. Re-sorting the stack
        # per bar means a given modality starts at a different x in every row, so
        # its width can no longer be compared across behaviors, which is the one
        # question this figure exists to answer.
        for m in data.modality_order:
            if m not in row.index:
                continue
            v = float(row[m]) * 100.0
            if v <= 0:
                continue
            ax.barh(yi, v, left=left, color=data.modality_colors[m],
                    edgecolor="white", linewidth=0.4)
            left += v

    # Group bands + labels for each dominant-modality cluster.
    dom_seq = [str(dom[b]) for b in behaviors]
    for start, end, m in _contiguous_runs(dom_seq):
        y_top = y[start] + 0.5
        y_bot = y[end - 1] - 0.5
        ax.axhspan(y_bot, y_top, color=data.modality_colors[m], alpha=0.06, zorder=0)
        if start > 0:
            ax.axhline(y_top, color="#B0BEC5", linewidth=0.6, alpha=0.6)
        ax.text(101, (y_top + y_bot) / 2, data.modality_labels[m].split(" (")[0],
                rotation=90, va="center", ha="left", fontsize=8,
                color=data.modality_colors[m], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(behaviors, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of feature importance (%)")
    ax.set_title("Where each behavior's signal comes from\n"
                 "(grouped by dominant modality · fixed segment order)",
                 fontsize=11)
    handles = [Patch(facecolor=data.modality_colors[m], label=data.modality_labels[m])
               for m in data.present_modalities if m in frac.columns]
    # Below the axes, not inside them — an in-axes legend sits on top of the
    # bottom bars, which are data.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=8, frameon=False, ncol=min(len(handles), 3))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, save_path)


def behaviorscape_clusters(data, save_path: Path | None = None) -> "Figure | None":
    """Behavior×behavior similarity matrix (shared feature-importance profile).

    For every pair of behaviors, the correlation of their feature-importance
    vectors says how much they draw on the *same* features.  The matrix is
    clustered (a dendrogram on top justifies the order) so similar behaviors form
    bright blocks on the diagonal — e.g. the Sniff/Approach variants — while a
    mostly-dark off-diagonal is itself the message: most behaviors recruit
    distinct feature sets.  Tick labels are coloured by dominant modality.
    """
    if not _HAS_MPL or data is None or data.is_empty():
        return None
    behaviors = list(data.matrix.columns)
    if len(behaviors) < 3:
        return None
    z = _linkage(data.matrix.to_numpy(dtype=float).T)
    if z is None:
        return None
    from matplotlib.gridspec import GridSpec  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415
    from scipy.cluster.hierarchy import dendrogram  # noqa: PLC0415

    # Pairwise profile correlation (clip negatives to 0 — anti-correlation of
    # sparse non-negative importance vectors isn't meaningful here).
    corr = np.corrcoef(data.matrix.to_numpy(dtype=float).T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, 0.0, 1.0)

    order = _leaf_order(z, len(behaviors))
    corr = corr[np.ix_(order, order)]
    labels = [behaviors[i] for i in order]
    dom = _dominant_modality_per_behavior(data)

    n = len(behaviors)
    size = max(6.0, min(0.42 * n + 2.4, 16.0))
    fig = plt.figure(figsize=(size, size))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[6.0, 0.25],
                  height_ratios=[1.0, 6.0], wspace=0.04, hspace=0.03)
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    if z is not None:
        try:
            dendrogram(z, ax=ax_dendro, color_threshold=0,
                       above_threshold_color="#90A4AE", no_labels=True)
            ax_dendro.set_xlim(0, 10 * n)
        except Exception:
            pass
    ax_dendro.axis("off")

    # The diagonal is 1.0 by construction and carries no information, but leaving it
    # in pins the colour scale to 1.0 and crushes every real off-diagonal block
    # (which top out far lower) into near-black. Mask it and scale to the strongest
    # actual pair, so the shared-profile blocks are the brightest thing on the plot.
    corr_off = corr.copy()
    np.fill_diagonal(corr_off, np.nan)
    off_max = float(np.nanmax(corr_off)) if np.isfinite(corr_off).any() else 1.0
    vmax = max(0.1, min(1.0, off_max))
    im = ax_mat.imshow(np.ma.masked_invalid(corr_off), aspect="equal",
                       cmap="rocket" if "rocket" in plt.colormaps() else "magma",
                       vmin=0.0, vmax=vmax)
    ax_mat.set_facecolor("#ECEFF1")  # the masked diagonal
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_mat.set_yticklabels(labels, fontsize=8)
    for ticklabels, axis_key in ((ax_mat.get_xticklabels(), "x"),
                                 (ax_mat.get_yticklabels(), "y")):
        for lbl in ticklabels:
            lbl.set_color(data.modality_colors.get(dom.get(lbl.get_text()), "#333333"))
            lbl.set_fontweight("bold")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(f"profile correlation\n(shared feature usage; scale to max pair "
                   f"= {vmax:.2f})", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Only the modalities with features behind them — see `present_modalities`.
    handles = [Patch(facecolor=data.modality_colors[m], label=data.modality_labels[m])
               for m in data.present_modalities]
    ax_dendro.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                     fontsize=8, frameon=False, title="dominant modality",
                     title_fontsize=8.5)
    fig.suptitle("Behavior similarity — do behaviors share feature profiles?\n"
                 "bright blocks = behaviors using the same features "
                 "(self-similarity diagonal masked)", fontsize=11, y=0.99)
    return _save(fig, save_path)


def _packed_component_layout(g, nx) -> dict:
    """Lay out each connected component on its own and tile the components into a
    compact grid, so a graph that is mostly disjoint modules reads as neat tiles
    instead of a few clusters flung across an ocean of whitespace.

    Tiles are not all the same size.  Components differ wildly in node count — the
    feature-sharing Sniff/Approach cluster can hold six behaviors and thirty features
    while most tiles hold one behavior and four — and forcing that component into a
    1×1 tile piles its behavior chips on top of each other.  A component much larger
    than the median therefore gets a full-width row to itself.
    """
    import math  # noqa: PLC0415

    comps = sorted((g.subgraph(c).copy() for c in nx.connected_components(g)),
                   key=lambda s: s.number_of_nodes(), reverse=True)
    sizes = [s.number_of_nodes() for s in comps]
    median = float(np.median(sizes)) if sizes else 1.0
    cols = max(1, int(math.ceil(math.sqrt(len(comps)))))
    spread = 0.78  # intra-tile fill fraction (leaves a gutter between tiles)

    # Place each component, widening the oversized ones to the full grid width.
    places: list[tuple] = []  # (subgraph, col, row, width)
    row = col = 0
    for sub, n in zip(comps, sizes):
        if len(comps) > 1 and n >= 2.0 * median and cols > 1:
            if col:
                row, col = row + 1, 0
            places.append((sub, 0, row, cols))
            row += 1
        else:
            places.append((sub, col, row, 1))
            col += 1
            if col >= cols:
                row, col = row + 1, 0

    pos: dict = {}
    for sub, c0, r0, width in places:
        if sub.number_of_nodes() == 1:
            local = {next(iter(sub.nodes)): np.array([0.5, 0.5])}
        else:
            # Repulsion has to grow with the tile, or a wide tile just spreads the
            # same tight ball of nodes across more empty space.
            k = 0.9 * math.sqrt(width)
            local = nx.spring_layout(sub, k=k, iterations=250, seed=42, weight="weight")
            xs = np.array([p[0] for p in local.values()])
            ys = np.array([p[1] for p in local.values()])
            xr = (xs.max() - xs.min()) or 1.0
            yr = (ys.max() - ys.min()) or 1.0
            local = {nd: np.array([(p[0] - xs.min()) / xr, (p[1] - ys.min()) / yr])
                     for nd, p in local.items()}
        for nd, p in local.items():
            pos[nd] = np.array([c0 + width * (0.5 + (p[0] - 0.5) * spread),
                                -r0 - 0.5 + (p[1] - 0.5) * spread])
    return pos


def _separate_chips(pos: dict, chips, *, x_ext: float, y_ext: float,
                    fig_w: float, fig_h: float, iterations: int = 260) -> None:
    """Push overlapping behavior chips apart, in place.

    Each behavior renders as a rounded text box far wider than the point the graph
    layout positioned it at, so a purely node-based layout happily overlaps them.
    This is a small collision relaxation on the chips' estimated bounding boxes
    (converted from rendered points into data units via the figure's scale): on each
    pass, any overlapping pair is pushed apart along the axis of least penetration.
    Feature nodes are left alone — only the chips move.
    """
    names = [c for c in chips if c in pos]
    if len(names) < 2:
        return
    # Chip footprint in data units. ~0.62 * fontsize is a good mean glyph advance
    # for this face; the +0.56 accounts for the box padding on both sides.
    data_per_in_x = x_ext / max(fig_w, 1e-6)
    data_per_in_y = y_ext / max(fig_h, 1e-6)
    half_w = {n: ((len(str(n)) * 0.62 + 0.56) * 9.0 / 72.0) * data_per_in_x / 2.0
              for n in names}
    half_h = {n: ((9.0 * 1.9) / 72.0) * data_per_in_y / 2.0 for n in names}

    for _ in range(iterations):
        moved = False
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pa, pb = pos[a], pos[b]
                dx, dy = float(pb[0] - pa[0]), float(pb[1] - pa[1])
                ox = (half_w[a] + half_w[b]) - abs(dx)   # x-overlap
                oy = (half_h[a] + half_h[b]) - abs(dy)   # y-overlap
                if ox <= 0 or oy <= 0:
                    continue  # boxes already clear on at least one axis
                moved = True
                # Separate along whichever axis needs the smaller correction.
                if ox * data_per_in_y < oy * data_per_in_x:
                    push = (ox / 2.0 + 1e-3) * (1.0 if dx >= 0 else -1.0)
                    pa[0] -= push
                    pb[0] += push
                else:
                    push = (oy / 2.0 + 1e-3) * (1.0 if dy >= 0 else -1.0)
                    pa[1] -= push
                    pb[1] += push
        if not moved:
            break


def behaviorscape_network(data, save_path: Path | None = None,
                          top_k_per_behavior: int = 4) -> "Figure | None":
    """Feature↔behavior network, drawn as tidy per-module tiles.

    Each behavior links to its top-``k`` features (edge width/opacity ∝
    importance).  The graph is split into connected components — each a behavior
    (or a few feature-sharing behaviors) with its driving features — and the
    components are packed into a grid so the modular structure is legible rather
    than diffuse.  Greedy-modularity Q is reported as the overall 'how modular'
    summary.  Feature dots keep the modality colours used across every figure.
    """
    if not _HAS_MPL or data is None or data.is_empty():
        return None
    mat = data.matrix
    behaviors = set(mat.columns)

    edges: list[tuple[str, str, float]] = []
    feat_used: set[str] = set()
    for beh in mat.columns:
        col = mat[beh].sort_values(ascending=False)
        col = col[col > 0].head(top_k_per_behavior)
        for feat, w in col.items():
            edges.append((str(feat), str(beh), float(w)))
            feat_used.add(str(feat))
    if not edges:
        return None
    features = sorted(feat_used)

    try:
        import networkx as nx  # noqa: PLC0415
    except Exception:
        return None

    g = nx.Graph()
    g.add_nodes_from(features)
    g.add_nodes_from(mat.columns)
    for f, b, w in edges:
        g.add_edge(f, b, weight=w)
    wdeg = dict(g.degree(weight="weight"))

    modularity_q = float("nan")
    try:
        from networkx.algorithms.community import (  # noqa: PLC0415
            greedy_modularity_communities, modularity,
        )
        comms = [set(c) for c in greedy_modularity_communities(g, weight="weight")]
        modularity_q = float(modularity(g, comms, weight="weight"))
    except Exception:
        pass

    pos = _packed_component_layout(g, nx)

    # Size the canvas from the layout's real extent — tiles are no longer uniform,
    # so a fixed square would letterbox a wide layout and crush a tall one.
    n_tiles = nx.number_connected_components(g)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_ext = max(float(xs.max() - xs.min()), 1e-6)
    y_ext = max(float(ys.max() - ys.min()), 1e-6)
    base = max(8.0, min(1.6 * np.sqrt(max(1, n_tiles)) + 5.0, 16.0))
    scale = base / max(x_ext, y_ext)
    fig_w = max(7.0, x_ext * scale)
    fig_h = max(6.0, y_ext * scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # The spring layout places behavior nodes as *points*, but each is drawn as a
    # wide text chip — so behaviors that share features (which is exactly what this
    # figure is about) end up with their chips stacked on top of each other. Nudge
    # the chips apart using their real rendered footprint.
    _separate_chips(pos, behaviors, x_ext=x_ext, y_ext=y_ext, fig_w=fig_w, fig_h=fig_h)

    # Edges: width + opacity ∝ importance.
    wmax = max(w for _, _, w in edges) or 1.0
    for f, b, w in edges:
        x0, y0 = pos[f]
        x1, y1 = pos[b]
        ax.plot([x0, x1], [y0, y1], color="#90A4AE",
                linewidth=0.4 + 2.4 * (w / wmax), alpha=0.45, zorder=1)

    # Feature nodes: colour = modality, size = weighted degree.
    fmax = max((wdeg.get(f, 0.0) for f in features), default=1.0) or 1.0
    for m in data.modality_order:
        fs = [f for f in features if data.modality.get(f) == m]
        if not fs:
            continue
        ax.scatter([pos[f][0] for f in fs], [pos[f][1] for f in fs],
                   s=[24 + 150 * (wdeg.get(f, 0.0) / fmax) for f in fs],
                   color=data.modality_colors[m], edgecolor="white", linewidth=0.5,
                   zorder=2, label=data.modality_labels[m])

    # Behavior nodes are rounded label boxes (no opaque squares); the label IS the
    # node, font-scaled by importance.
    bmax = max((wdeg.get(b, 0.0) for b in behaviors), default=1.0) or 1.0
    for b in behaviors:
        fs = 7.0 + 4.0 * (wdeg.get(b, 0.0) / bmax)
        ax.annotate(b, pos[b], fontsize=fs, color="white", ha="center", va="center",
                    zorder=5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#263238",
                              edgecolor="white", linewidth=0.8, alpha=0.92))

    # Label EVERY feature node, not just the top hubs. An unlabelled dot carries no
    # information in a feature-landscape figure, and the old top-8 rule dropped the
    # labels on the node directly under a behavior chip. Each label is pushed
    # radially outward from its tile centre, so it points away from the chip (which
    # sits near the centre) instead of landing on top of it.
    tile_of: dict[tuple[int, int], list[str]] = {}
    for nd, p in pos.items():
        tile_of.setdefault((round(float(p[0])), round(float(p[1]))), []).append(nd)
    centroid = {t: np.mean([pos[nd] for nd in nds], axis=0) for t, nds in tile_of.items()}

    for f in features:
        p = pos[f]
        tile = (round(float(p[0])), round(float(p[1])))
        d = np.asarray(p, dtype=float) - centroid[tile]
        norm = float(np.hypot(*d))
        u = d / norm if norm > 1e-6 else np.array([0.0, 1.0])
        ax.annotate(
            str(f), p, fontsize=4.6, color="#37474F", zorder=4,
            xytext=(9.0 * u[0], 9.0 * u[1]), textcoords="offset points",
            ha="left" if u[0] > 0.15 else "right" if u[0] < -0.15 else "center",
            va="bottom" if u[1] >= 0 else "top",
        )

    ax.axis("off")
    # Extra headroom at the top: the two legends live up there and would otherwise
    # sit on top of the topmost tile's feature labels.
    ax.margins(x=0.06, y=0.06)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
    # Node size encodes importance, so the legend has to say what a size means —
    # otherwise the biggest dot on the page is uninterpretable.
    size_handles = [
        plt.scatter([], [], s=24 + 150 * frac_, facecolor="#B0BEC5",
                    edgecolor="white", linewidth=0.5,
                    label=f"{frac_:.0%} of max importance")
        for frac_ in (0.25, 1.0)
    ]
    mod_handles = [
        plt.scatter([], [], s=60, facecolor=data.modality_colors[m], edgecolor="white",
                    linewidth=0.5, label=data.modality_labels[m])
        for m in data.modality_order if any(data.modality.get(f) == m for f in features)
    ]
    # Opaque backing: these sit over the plot area and would otherwise be read
    # through the feature labels underneath them.
    lg_kw = {"frameon": True, "facecolor": "white", "framealpha": 0.92,
             "edgecolor": "#CFD8DC"}
    first = ax.legend(handles=mod_handles, loc="upper left", fontsize=8,
                      title="feature modality", title_fontsize=8.5, **lg_kw)
    ax.add_artist(first)
    ax.legend(handles=size_handles, loc="upper right", fontsize=7.5, title="dot size",
              title_fontsize=8.5, labelspacing=1.1, borderpad=0.9, **lg_kw)
    q_txt = f" · modularity Q={modularity_q:.2f}" if np.isfinite(modularity_q) else ""
    ax.set_title(f"Behaviorscape network — {n_tiles} feature modules{q_txt}\n"
                 "each tile = a behavior + its top features · "
                 "dot size = importance · edge width = link strength", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, save_path)


def behaviorscape_distinctiveness(data, save_path: Path | None = None,
                                  stats=None) -> "Figure | None":
    """Per-behavior profile distinctiveness, with a PERMANOVA significance headline.

    Quantifies the heatmap's qualitative story.  Each bar is a behavior's mean
    cosine distance from its (per-project) replicates to *every other* behavior's
    importance centroid — i.e. how much that behavior's feature reliance differs
    from the rest; error bars are the SE across project replicates.  The title
    carries the PERMANOVA test of whether behavior identity explains the
    importance-profile variance (the formal 'different behaviors rely on
    different features' result).
    """
    if not _HAS_MPL or data is None or data.is_empty():
        return None
    from matplotlib.patches import Patch  # noqa: PLC0415

    from abel.validation.analyses.behaviorscape import (  # noqa: PLC0415
        behavior_distinctiveness_stats,
    )
    if stats is None:
        stats = behavior_distinctiveness_stats(data)
    if stats is None:
        return None

    order = sorted(stats.behaviors, key=lambda b: stats.distinctiveness[b], reverse=True)
    n = len(order)
    vals = [stats.distinctiveness[b] for b in order]
    errs = [stats.err[b] for b in order]
    colors = [data.modality_colors.get(stats.dominant_modality.get(b), "#888888")
              for b in order]

    fig, ax = plt.subplots(figsize=(8.8, max(2.8, 0.40 * n + 1.9)))
    y = np.arange(n)
    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="white", linewidth=0.4,
            error_kw={"elinewidth": 0.8, "ecolor": "#444", "capsize": 2})
    ax.axvline(stats.mean_distinctiveness, color="#555", linestyle="--", linewidth=1.0)
    ax.text(stats.mean_distinctiveness, -0.7, "mean", fontsize=7.5, color="#555",
            ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b}  (n={stats.n_replicates[b]})" for b in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("distinctiveness — mean profile distance to other behaviors (cosine)")
    # Cosine distinctiveness lives in a narrow band near 1.0; anchoring the axis at
    # 0 squeezes every bar's differences into the last sliver of the panel. Frame
    # the observed range instead (the axis break is explicit in the label).
    finite = [v for v in vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.01, (hi - lo) * 0.25)
        left = max(0.0, lo - pad)
        if left > 0.02:
            ax.set_xlim(left, min(1.0, hi + pad * 0.6))
            ax.set_xlabel("distinctiveness — mean profile distance to other behaviors "
                          f"(cosine; axis starts at {left:.2f})")
        else:
            ax.set_xlim(left=0)

    if stats.permanova:
        pm = stats.permanova
        p_txt = "p<0.001" if pm["p"] < 0.001 else f"p={pm['p']:.3f}"
        sub = (f"PERMANOVA: behavior explains {pm['R2'] * 100:.0f}% of importance "
               f"variance (pseudo-F={pm['pseudo_F']:.1f}, {p_txt}; "
               f"{pm['n_groups']} behaviors × ≥2 projects, {pm['n_perm']} permutations)")
    else:
        sub = ("descriptive only — the PERMANOVA test needs ≥2 behaviors with ≥2 "
               "projects each")
    ax.set_title(f"Do different behaviors rely on different features?\n{sub}", fontsize=10.5)

    handles = [Patch(facecolor=data.modality_colors[m], label=data.modality_labels[m])
               for m in data.present_modalities]
    # Outside the axes: an in-axes legend lands on the shortest bars.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, frameon=False, title="dominant modality", title_fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, save_path)


def behaviorscape_figures(data, out_dir: Path) -> list[Path]:
    """Render all behaviorscape figures into ``out_dir``; return saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("behaviorscape_heatmap.png", behaviorscape_heatmap),
        ("behaviorscape_modality_bars.png", behaviorscape_modality_bars),
        ("behaviorscape_distinctiveness.png", behaviorscape_distinctiveness),
        ("behaviorscape_clusters.png", behaviorscape_clusters),
        ("behaviorscape_network.png", behaviorscape_network),
    ]
    saved: list[Path] = []
    for fname, fn in specs:
        path = out_dir / fname
        try:
            fig = fn(data, save_path=path)
        except Exception:  # noqa: BLE001 — one bad figure shouldn't sink the rest
            fig = None
        if fig is not None and path.exists():
            saved.append(path)
        if fig is not None:
            plt.close(fig)
    return saved


def close_all() -> None:
    if _HAS_MPL:
        plt.close("all")
