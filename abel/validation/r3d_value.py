"""Focused analysis: does the **R3D appearance embedding** earn its cost?

R3D-18 is its own user-facing toggle (Features tab → "R3D appearance embeddings"),
sitting *under* ``use_video_features``: it runs a pretrained 3-D CNN over a
pose-centred crop of every segment and appends 512 dense dimensions to the feature
table.  That is by far the most expensive feature family ABEL computes — it wants a
GPU, it caches per session under ``derived/r3d_features/``, and it needs the video
to still be reachable.  So it deserves the question the toggle implies: *given
everything else ABEL already extracts, what does the embedding add?*

The suite's existing ablation deliberately does **not** answer this.  Its
"+ Video features" bar is one lump (flow + surface motion + R3D) and is kept that
way on purpose, so its numbers stay comparable with runs made before the embedding
existed.  This module is the separate, explicit test.

**Primary comparison (always run) — the toggle, exactly.**
    ``full_no_r3d`` vs ``full_with_r3d``: every feature ABEL would use for this
    project (pose + kinematics + context/ROI + handcrafted video ± social), once
    without the ``r3d_*`` columns and once with them.  Same held-out split, same
    training pool, paired per seed.  The ΔF1 *is* what flipping the toggle buys.

**Decomposition (optional) — is R3D redundant with optical flow, or complementary?**
    A positive primary delta says the embedding adds something; it does not say
    whether it is doing the *same* job as the handcrafted video features or a
    different one.  Three extra arms settle it, all over the same pose baseline:

    * ``pose_only``        — pose + kinematics (+ context, + social), no pixels at all
    * ``pose_handcrafted`` — that, plus optical flow / surface motion only
    * ``pose_r3d``         — that, plus the R3D embedding only

    If ``pose_r3d`` ≈ ``pose_handcrafted`` and the primary delta is ~0, the two are
    redundant and the toggle buys nothing but runtime.  If ``pose_r3d`` beats
    ``pose_handcrafted`` *and* the primary delta is positive, the embedding is
    carrying signal the named features never had.

Nothing is re-implemented: every arm calls the shared ``engine.run_one_config``
primitive with a ``feature_cols_override``, so all of them are ABEL's real
classifier, fit the way the product fits it.

Run standalone::

    python -m abel.validation.r3d_value \\
        --projects c:/Users/jober/CIE_NSF --behaviors Groom Freeze \\
        --seeds 5 --decompose --out ./r3d_value_out

Writes ``r3d_value.csv`` (tidy, one row per project·behavior, per-seed F1 retained)
plus ``r3d_value.png`` (paired dumbbell + gain CI) and, with ``--decompose``,
``r3d_decomposition.png``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import features, holdout, subsample
from abel.validation.datamodel import ProjectRef
from abel.validation.engine import run_one_config

# Arm names — stable, they are CSV column stems and plot keys.
ARM_NO_R3D = "full_no_r3d"
ARM_WITH_R3D = "full_with_r3d"
ARM_POSE_ONLY = "pose_only"
ARM_POSE_HANDCRAFTED = "pose_handcrafted"
ARM_POSE_R3D = "pose_r3d"

#: The two arms of the primary paired test, in (baseline, treatment) order.
PRIMARY_ARMS = (ARM_NO_R3D, ARM_WITH_R3D)
#: Extra arms enabled by ``decompose=True``.
DECOMPOSITION_ARMS = (ARM_POSE_ONLY, ARM_POSE_HANDCRAFTED, ARM_POSE_R3D)

ARM_LABELS = {
    ARM_NO_R3D: "All features, R3D off",
    ARM_WITH_R3D: "All features, R3D on",
    ARM_POSE_ONLY: "Pose (no pixels)",
    ARM_POSE_HANDCRAFTED: "Pose + flow/surface",
    ARM_POSE_R3D: "Pose + R3D",
}


@dataclass
class R3DValueResult:
    """Paired with/without-R3D accuracy for one (project, behavior)."""

    project_id: str
    behavior_id: str
    behavior_name: str
    n_seeds: int = 0
    n_pos_holdout: int = 0

    # Feature counts per arm — the cost side of the ledger (R3D adds 512 columns).
    n_features_no_r3d: int = 0
    n_features_with_r3d: int = 0
    n_r3d_cols: int = 0

    # Per-seed F1 on the two primary arms (aligned by seed → paired).
    f1_no_r3d_seeds: list[float] = field(default_factory=list)
    f1_with_r3d_seeds: list[float] = field(default_factory=list)

    f1_no_r3d: float = float("nan")
    f1_with_r3d: float = float("nan")
    precision_no_r3d: float = float("nan")
    precision_with_r3d: float = float("nan")
    recall_no_r3d: float = float("nan")
    recall_with_r3d: float = float("nan")
    # Held-out error counts (mean across seeds) — what the gain costs/saves in clips.
    fp_no_r3d: float = float("nan")
    fp_with_r3d: float = float("nan")
    fn_no_r3d: float = float("nan")
    fn_with_r3d: float = float("nan")

    gain: float = float("nan")
    gain_ci95: float = float("nan")
    p_value: float = float("nan")
    significant: bool = False

    # Decomposition arms (NaN when decompose=False).
    f1_pose_only: float = float("nan")
    f1_pose_handcrafted: float = float("nan")
    f1_pose_r3d: float = float("nan")
    # R3D-alone minus handcrafted-alone, both over the same pose baseline. Positive
    # → the embedding sees something the named video features do not.
    r3d_over_handcrafted: float = float("nan")

    error: str = ""

    def to_row(self) -> dict:
        """Flat CSV row, with the per-seed F1 of both primary arms retained.

        The paired test must be reproducible from the CSV alone (Prism, R, a
        reviewer) — a mean, a CI half-width and a significance boolean are not
        enough to re-run it.
        """
        d = asdict(self)
        no_seeds = d.pop("f1_no_r3d_seeds", None) or []
        yes_seeds = d.pop("f1_with_r3d_seeds", None) or []
        for i, v in enumerate(no_seeds, start=1):
            d[f"f1_no_r3d_seed{i}"] = v
        for i, v in enumerate(yes_seeds, start=1):
            d[f"f1_with_r3d_seed{i}"] = v
        return d


def _ci95(values) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return 0.0
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.ci95(vals)  # t-based 95% CI half-width


def _paired_p(deltas) -> float:
    """Two-sided paired t-test on the per-seed differences (shared implementation)."""
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.paired_p(deltas)


def _mean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _arm_columns(pool: pd.DataFrame, *, has_social: bool,
                 has_context: bool) -> dict[str, list[str]]:
    """Feature columns for every arm, from one pool.

    ``include_r3d`` is passed explicitly on every arm here — that is what switches
    :func:`features.select_feature_cols` out of its default "R3D rides with video"
    mode and into the split the whole module exists to make.
    """
    common = dict(include_social=has_social, include_context=has_context)
    return {
        ARM_NO_R3D: features.select_feature_cols(
            pool, include_video=True, include_r3d=False, **common),
        ARM_WITH_R3D: features.select_feature_cols(
            pool, include_video=True, include_r3d=True, **common),
        ARM_POSE_ONLY: features.select_feature_cols(
            pool, include_video=False, include_r3d=False, **common),
        ARM_POSE_HANDCRAFTED: features.select_feature_cols(
            pool, include_video=True, include_r3d=False, **common),
        ARM_POSE_R3D: features.select_feature_cols(
            pool, include_video=False, include_r3d=True, **common),
    }


def run_r3d_value(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: holdout.HoldoutSplit,
    *,
    n_seeds: int = 5,
    decompose: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> R3DValueResult:
    """Train with vs. without the R3D embedding (paired per seed) for one behavior."""
    name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    res = R3DValueResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=name, n_seeds=int(n_seeds),
        n_pos_holdout=int(subsample.count_positives(holdout_split.holdout, behavior_id)),
    )

    r3d_cols = features.r3d_only_cols(pool)
    res.n_r3d_cols = len(r3d_cols)
    if not r3d_cols:
        # Not an error the user should chase in the log: it just means this project
        # was extracted with the toggle off, or predates it.
        res.error = ("no r3d_* columns in this project's features — re-extract with "
                     "'R3D appearance embeddings' enabled to compare")
        return res

    has_social = bool(features.social_only_cols(pool))
    has_context = bool(features.context_only_cols(pool))
    arm_cols = _arm_columns(pool, has_social=has_social, has_context=has_context)
    res.n_features_no_r3d = len(arm_cols[ARM_NO_R3D])
    res.n_features_with_r3d = len(arm_cols[ARM_WITH_R3D])

    total_pos = subsample.count_positives(pool, behavior_id)
    if total_pos == 0:
        res.error = "no positive examples for this behavior in the training pool"
        return res

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    arms = list(PRIMARY_ARMS) + (list(DECOMPOSITION_ARMS) if decompose else [])
    # arm -> metric name -> per-seed values.  Every arm is fit on the SAME pool and
    # the SAME held-out rows, differing only in which columns it may look at, so any
    # difference between two arms is attributable to the feature families alone.
    per_arm: dict[str, dict[str, list[float]]] = {
        a: {k: [] for k in ("f1", "precision", "recall", "fp", "fn")} for a in arms}

    for rep in range(n_seeds):
        seed = 3000 + rep
        for arm in arms:
            _log(f"{project.project_id}/{name}: {ARM_LABELS[arm]} "
                 f"seed {rep + 1}/{n_seeds}…")
            r = run_one_config(
                trainer, project, behavior_id, pool, holdout_split.holdout,
                seed=seed, feature_cols_override=arm_cols[arm],
                n_pos_train=int(total_pos), n_neg_train=int(len(pool) - total_pos),
            )
            ok = (not r.error) and np.isfinite(r.f1)
            sink = per_arm[arm]
            sink["f1"].append(r.f1 if ok else float("nan"))
            sink["precision"].append(r.precision if ok else float("nan"))
            sink["recall"].append(r.recall if ok else float("nan"))
            sink["fp"].append(float(r.fp) if ok else float("nan"))
            sink["fn"].append(float(r.fn) if ok else float("nan"))

    f1_no = per_arm[ARM_NO_R3D]["f1"]
    f1_yes = per_arm[ARM_WITH_R3D]["f1"]
    res.f1_no_r3d_seeds, res.f1_with_r3d_seeds = f1_no, f1_yes
    res.f1_no_r3d, res.f1_with_r3d = _mean(f1_no), _mean(f1_yes)
    res.precision_no_r3d = _mean(per_arm[ARM_NO_R3D]["precision"])
    res.precision_with_r3d = _mean(per_arm[ARM_WITH_R3D]["precision"])
    res.recall_no_r3d = _mean(per_arm[ARM_NO_R3D]["recall"])
    res.recall_with_r3d = _mean(per_arm[ARM_WITH_R3D]["recall"])
    res.fp_no_r3d = _mean(per_arm[ARM_NO_R3D]["fp"])
    res.fp_with_r3d = _mean(per_arm[ARM_WITH_R3D]["fp"])
    res.fn_no_r3d = _mean(per_arm[ARM_NO_R3D]["fn"])
    res.fn_with_r3d = _mean(per_arm[ARM_WITH_R3D]["fn"])

    paired = [y - n for y, n in zip(f1_yes, f1_no)
              if np.isfinite(y) and np.isfinite(n)]
    res.gain = float(np.mean(paired)) if paired else float("nan")
    res.gain_ci95 = _ci95(paired)
    res.p_value = _paired_p(paired)
    res.significant = bool(
        len(paired) >= 2 and np.isfinite(res.gain) and abs(res.gain) > res.gain_ci95)

    if decompose:
        res.f1_pose_only = _mean(per_arm[ARM_POSE_ONLY]["f1"])
        res.f1_pose_handcrafted = _mean(per_arm[ARM_POSE_HANDCRAFTED]["f1"])
        res.f1_pose_r3d = _mean(per_arm[ARM_POSE_R3D]["f1"])
        if np.isfinite(res.f1_pose_r3d) and np.isfinite(res.f1_pose_handcrafted):
            res.r3d_over_handcrafted = float(
                res.f1_pose_r3d - res.f1_pose_handcrafted)
    return res


def run_analysis(
    project_roots: list[str],
    behavior_names: list[str],
    *,
    n_seeds: int = 5,
    decompose: bool = False,
    min_confidence: float = 1.0,
    holdout_test_size: float = 0.25,
    holdout_seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> list[R3DValueResult]:
    """Run the with/without-R3D comparison for the named behaviors across projects."""
    trainer = ActiveLearningTrainerService()
    results: list[R3DValueResult] = []
    for root in project_roots:
        project = ProjectRef.load(root)
        if not project.is_valid():
            if progress_cb:
                progress_cb(f"SKIP (no training set): {root}")
            continue
        bids = project.behavior_ids_matching(behavior_names)
        if not bids:
            if progress_cb:
                progress_cb(f"SKIP (no matching behaviors {behavior_names}): {project.name}")
            continue
        try:
            hsplit = holdout.split(
                project, min_confidence=min_confidence,
                test_size=holdout_test_size, seed=holdout_seed)
        except Exception as exc:  # noqa: BLE001
            for bid in bids:
                results.append(R3DValueResult(
                    project_id=project.project_id, behavior_id=str(bid),
                    behavior_name=project.behavior_label(bid),
                    error=f"holdout failed: {exc}"))
            continue
        for bid in bids:
            results.append(run_r3d_value(
                trainer, project, str(bid), hsplit, n_seeds=n_seeds,
                decompose=decompose, progress_cb=progress_cb))
    return results


def results_to_frame(results: list[R3DValueResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def summarize(results: list[R3DValueResult]) -> str:
    """Human-readable verdict — the text the GUI status line and the CLI both show.

    Deliberately ASCII: this string is ``print``-ed by :func:`main`, and a Windows
    console on cp1252 raises ``UnicodeEncodeError`` on a literal delta.  The figures
    are free to use one — matplotlib is not writing to a terminal.
    """
    usable = [r for r in results if not r.error and np.isfinite(r.gain)]
    if not usable:
        return "No usable comparisons (see per-behavior errors)."
    wins = [r for r in usable if r.significant and r.gain > 0]
    losses = [r for r in usable if r.significant and r.gain < 0]
    mean_gain = float(np.mean([r.gain for r in usable]))
    head = (f"{len(usable)} comparison(s): R3D helps significantly in {len(wins)}, "
            f"hurts in {len(losses)}, mean dF1 {mean_gain:+.3f}.")
    if not wins and not losses:
        head += ("  No behavior's CI excludes zero — on this evidence the embedding "
                 "is not paying for its extraction cost.")
    return head


# ── Figures ────────────────────────────────────────────────────────────────


def plot_r3d_value(results: list[R3DValueResult], save_path: Path) -> Path:
    """Paired dumbbells — F1 with the R3D toggle off → on, per (project, behavior).

    Left panel is the pairing itself (the design's whole point); right panel is the
    paired gain with its 95% CI across seeds, which is what the significance claim
    is about.  Mirrors :func:`abel.validation.video_value.plot_video_value` so the
    two figures read as one family.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    usable = [r for r in results if not r.error and np.isfinite(r.f1_no_r3d)]
    if not usable:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                "No usable results\n(re-extract with R3D embeddings enabled)",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    usable = sorted(usable, key=lambda r: (not np.isfinite(r.gain), -float(r.gain)))
    n = len(usable)
    labels = [f"{r.project_id[:16]} · {r.behavior_name}" for r in usable]
    y = np.arange(n)[::-1]
    off = np.array([r.f1_no_r3d for r in usable], dtype=float)
    on = np.array([r.f1_with_r3d for r in usable], dtype=float)
    gains = np.array([r.gain for r in usable], dtype=float)
    gain_ci = np.array([r.gain_ci95 for r in usable], dtype=float)
    sig = [bool(r.significant) for r in usable]

    C_OFF, C_ON = "#9E9E9E", "#4C72B0"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, max(3.2, 0.30 * n + 1.9)),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.06}, sharey=True)

    for yi, a, b, s in zip(y, off, on, sig):
        ax1.plot([a, b], [yi, yi], color=(C_ON if s else "#CFD8DC"),
                 linewidth=2.0 if s else 1.4, alpha=0.9 if s else 0.7, zorder=1,
                 solid_capstyle="round")
    ax1.scatter(off, y, s=34, color=C_OFF, edgecolor="white", linewidth=0.7,
                zorder=3, label="R3D off (all other features on)")
    ax1.scatter(on, y, s=34, color=C_ON, edgecolor="white", linewidth=0.7,
                zorder=3, label="R3D on")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    lo = float(np.nanmin([off.min(), on.min()]))
    ax1.set_xlim(max(0.0, lo - 0.06), 1.0)
    ax1.set_xlabel("F1 (held-out, target-vs-rest)", fontsize=9)
    ax1.set_title("Paired: R3D embeddings off → on", fontsize=10, loc="left")
    # Open a blank band below the last row for the legend.  Letting it default into
    # "lower left" drops it straight on top of the bottom behavior's dumbbell as soon
    # as the run has only a handful of behaviors — which is the common case here,
    # since you point this analysis at the few behaviors you suspect need pixels.
    ax1.set_ylim(-1.25, n - 0.45)
    ax1.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    ax1.grid(axis="x", alpha=0.22)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    ax2.barh(y, gains, 0.62, xerr=gain_ci,
             color=[(C_ON if s else "#CFD8DC") for s in sig],
             edgecolor="white", linewidth=0.4,
             error_kw={"elinewidth": 0.8, "ecolor": "#455A64", "capsize": 2})
    ax2.axvline(0, color="#546E7A", linewidth=1.0)
    span = float(np.nanmax(np.abs(gains) + gain_ci)) if len(gains) else 1.0
    pad = max(span * 0.03, 1e-3)
    for yi, g, c, s in zip(y, gains, gain_ci, sig):
        if np.isfinite(g):
            side = 1.0 if g >= 0 else -1.0
            ax2.text(g + side * (c + pad), yi, f"{g:+.3f}" + ("*" if s else ""),
                     va="center", ha="left" if g >= 0 else "right", fontsize=7,
                     color="#263238" if s else "#90A4AE")
    ax2.set_xlabel("ΔF1 from R3D  (paired, 95% CI)", fontsize=9)
    ax2.set_title("Gain  (* = CI excludes 0)", fontsize=10, loc="left")
    ax2.grid(axis="x", alpha=0.22)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ends = np.concatenate([gains - gain_ci, gains + gain_ci])
    ends = ends[np.isfinite(ends)]
    if ends.size:
        lo_e, hi_e = float(ends.min()), float(ends.max())
        rng = max(hi_e - lo_e, 1e-4)
        ax2.set_xlim(min(lo_e, 0.0) - 0.34 * rng, max(hi_e, 0.0) + 0.34 * rng)

    n_cols = usable[0].n_r3d_cols if usable else 0
    fig.suptitle(f"Value of the R3D appearance embedding ({n_cols} dims) — "
                 "paired, same split & training pool",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_r3d_decomposition(results: list[R3DValueResult], save_path: Path) -> Path:
    """Grouped bars: pose / pose+flow / pose+R3D / everything, per behavior.

    Answers the redundancy question the primary figure cannot.  Reading it: if the
    ``Pose + R3D`` and ``Pose + flow/surface`` bars land together and ``All`` adds
    nothing over either, the two pixel families are substitutes.  If ``Pose + R3D``
    clears ``Pose + flow/surface`` and ``All`` clears both, the embedding carries
    signal the handcrafted features never had.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    usable = [r for r in results
              if not r.error and np.isfinite(r.f1_pose_only)]
    if not usable:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No decomposition results\n(run with 'decompose' enabled)",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    usable = sorted(usable, key=lambda r: -float(
        r.r3d_over_handcrafted if np.isfinite(r.r3d_over_handcrafted) else -np.inf))
    series = [
        (ARM_POSE_ONLY, "Pose (no pixels)", "#9E9E9E",
         [r.f1_pose_only for r in usable]),
        (ARM_POSE_HANDCRAFTED, "Pose + flow/surface", "#C44E52",
         [r.f1_pose_handcrafted for r in usable]),
        (ARM_POSE_R3D, "Pose + R3D", "#4C72B0",
         [r.f1_pose_r3d for r in usable]),
        (ARM_WITH_R3D, "All features", "#55A868",
         [r.f1_with_r3d for r in usable]),
    ]
    n = len(usable)
    y = np.arange(n)[::-1]
    height = 0.78 / len(series)

    fig, ax = plt.subplots(figsize=(9.2, max(3.0, 0.52 * n + 1.8)))
    for i, (_key, label, colour, vals) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * height
        ax.barh(y + offset, vals, height, label=label, color=colour,
                edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.project_id[:16]} · {r.behavior_name}" for r in usable],
                       fontsize=8)
    ax.set_xlabel("F1 (held-out, target-vs-rest)", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.22)
    # Blank band below the last group so the 4-entry legend never lands on a bar.
    ax.set_ylim(-1.35, n - 0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("R3D vs. handcrafted video — are they redundant?",
                 fontsize=11.5, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--projects", nargs="+", required=True,
                    help="ABEL project roots")
    ap.add_argument("--behaviors", nargs="+", required=True,
                    help="Behavior names to compare")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--decompose", action="store_true",
                    help="Also run pose-only / pose+flow / pose+R3D arms")
    ap.add_argument("--min-confidence", type=float, default=1.0)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--holdout-seed", type=int, default=42)
    ap.add_argument("--out", default="./r3d_value_out")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_analysis(
        args.projects, args.behaviors, n_seeds=args.seeds,
        decompose=args.decompose, min_confidence=args.min_confidence,
        holdout_test_size=args.test_size, holdout_seed=args.holdout_seed,
        progress_cb=print)
    df = results_to_frame(results)
    df.to_csv(out / "r3d_value.csv", index=False)
    plot_r3d_value(results, out / "r3d_value.png")
    if args.decompose:
        plot_r3d_decomposition(results, out / "r3d_decomposition.png")
    print(summarize(results))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
