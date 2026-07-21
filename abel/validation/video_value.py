"""Focused analysis: how much do the *video motion features* buy us?

The headline case is **Groom vs Freeze** — two behaviors that pose-only kinematics
struggle to separate (freezing is defined by the *absence* of motion, grooming by
small *rhythmic* motion; both are low-locomotion, so body-part geometry alone
confuses them).  ABEL's video-derived motion features (optical flow magnitude/
entropy, local surface motion, R3D appearance) are exactly the signal that should
disambiguate them.

For each requested ``(project, behavior)`` this trains ABEL's real classifier twice
on the *same* held-out split and the *same* per-seed training subsample — once
**without** the video features (pose + kinematics + context ± social) and once
**with** them — so the F1 difference is a clean paired estimate of what the video
motion features add.  Nothing is re-implemented: both arms call the shared
``engine.run_one_config`` primitive with a ``feature_cols_override``.

Run it on whatever projects you like::

    python -m abel.validation.video_value \\
        --projects c:/Users/jober/CIE_NSF c:/Users/jober/NSF-NewLights \\
        --behaviors Groom Freeze --seeds 5 --out ./video_value_out

Writes ``video_value.csv`` (tidy, paste-ready) + ``video_value.png`` (paired bars).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation import features, holdout, prism, subsample
from abel.validation.datamodel import ProjectRef
from abel.validation.engine import run_one_config

# Config names used on the two arms (kept stable for CSV / plot consumers).
ARM_NO_VIDEO = "no_video"
ARM_WITH_VIDEO = "with_video"


@dataclass
class VideoValueResult:
    """Paired with/without-video accuracy for one (project, behavior)."""

    project_id: str
    behavior_id: str
    behavior_name: str
    n_seeds: int = 0
    n_pos_holdout: int = 0
    n_features_no_video: int = 0
    n_features_with_video: int = 0

    # Per-seed F1 on each arm (aligned by seed → paired).
    f1_no_video_seeds: list[float] = field(default_factory=list)
    f1_with_video_seeds: list[float] = field(default_factory=list)

    # Means + the paired gain (with − without) and its 95% CI half-width.
    f1_no_video: float = float("nan")
    f1_with_video: float = float("nan")
    precision_no_video: float = float("nan")
    precision_with_video: float = float("nan")
    recall_no_video: float = float("nan")
    recall_with_video: float = float("nan")
    # Held-out error counts (mean across seeds) — the reduction is the story.
    fp_no_video: float = float("nan")
    fp_with_video: float = float("nan")
    fn_no_video: float = float("nan")
    fn_with_video: float = float("nan")

    gain: float = float("nan")
    gain_ci95: float = float("nan")
    p_value: float = float("nan")
    significant: bool = False
    error: str = ""

    def to_row(self) -> dict:
        """Flat CSV row — including the per-seed F1 on each arm.

        The seeds used to be dropped here, which left the export with a mean, a CI
        half-width and a significance *boolean* but no way to re-run the paired test
        downstream (Prism, R, a reviewer).  They are emitted as ``f1_no_video_seed1…N``
        / ``f1_with_video_seed1…N`` so the paired comparison is reproducible from the
        CSV alone.
        """
        d = asdict(self)
        no_seeds = d.pop("f1_no_video_seeds", None) or []
        yes_seeds = d.pop("f1_with_video_seeds", None) or []
        for i, v in enumerate(no_seeds, start=1):
            d[f"f1_no_video_seed{i}"] = v
        for i, v in enumerate(yes_seeds, start=1):
            d[f"f1_with_video_seed{i}"] = v
        return d


def _ci95(values) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return 0.0
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.ci95(vals)  # t-based 95% CI half-width


def _paired_p(deltas) -> float:
    """Two-sided paired t-test p-value on the per-seed differences.

    A boolean ``significant`` is not something a manuscript can report; the exact p
    is.  Returns NaN when there are too few seeds or the differences are constant.
    """
    vals = np.asarray([v for v in deltas if np.isfinite(v)], dtype=float)
    if len(vals) < 2 or float(np.std(vals, ddof=1)) == 0.0:
        return float("nan")
    try:
        from scipy import stats  # noqa: PLC0415
    except ImportError:
        return float("nan")
    return float(stats.ttest_1samp(vals, 0.0).pvalue)


def _mean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def run_video_value(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: holdout.HoldoutSplit,
    *,
    n_seeds: int = 5,
    progress_cb: Callable[[str], None] | None = None,
) -> VideoValueResult:
    """Train with vs. without video features (paired per seed) for one behavior."""
    name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    res = VideoValueResult(
        project_id=project.project_id, behavior_id=str(behavior_id),
        behavior_name=name, n_seeds=int(n_seeds),
        n_pos_holdout=int(subsample.count_positives(holdout_split.holdout, behavior_id)),
    )

    has_social = bool(features.social_only_cols(pool))
    # Both arms hold everything else constant; they differ ONLY by the video
    # motion features, so the delta isolates the video features' contribution.
    cols_no_video = features.select_feature_cols(
        pool, include_video=False, include_social=has_social)
    cols_with_video = features.select_feature_cols(
        pool, include_video=True, include_social=has_social)
    if len(cols_with_video) <= len(cols_no_video):
        res.error = "project has no video features (use_video_features off or none extracted)"
        return res

    total_pos = subsample.count_positives(pool, behavior_id)
    if total_pos == 0:
        res.error = "no positive examples for this behavior in the training pool"
        return res

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    f1_no, f1_yes = [], []
    prec_no, prec_yes, rec_no, rec_yes = [], [], [], []
    fp_no, fp_yes, fn_no, fn_yes = [], [], [], []
    for rep in range(n_seeds):
        seed = 2000 + rep
        for arm, cols, sink in (
            (ARM_NO_VIDEO, cols_no_video,
             (f1_no, prec_no, rec_no, fp_no, fn_no)),
            (ARM_WITH_VIDEO, cols_with_video,
             (f1_yes, prec_yes, rec_yes, fp_yes, fn_yes)),
        ):
            _log(f"{project.project_id}/{name}: {arm} seed {rep + 1}/{n_seeds}…")
            r = run_one_config(
                trainer, project, behavior_id, pool, holdout_split.holdout,
                seed=seed, feature_cols_override=cols,
                n_pos_train=int(total_pos), n_neg_train=int(len(pool) - total_pos),
            )
            ok = (not r.error) and np.isfinite(r.f1)
            sink[0].append(r.f1 if ok else float("nan"))
            sink[1].append(r.precision if ok else float("nan"))
            sink[2].append(r.recall if ok else float("nan"))
            sink[3].append(float(r.fp) if ok else float("nan"))
            sink[4].append(float(r.fn) if ok else float("nan"))
            if arm == ARM_NO_VIDEO:
                res.n_features_no_video = int(r.n_features or len(cols_no_video))
            else:
                res.n_features_with_video = int(r.n_features or len(cols_with_video))

    res.f1_no_video_seeds = f1_no
    res.f1_with_video_seeds = f1_yes
    res.f1_no_video, res.f1_with_video = _mean(f1_no), _mean(f1_yes)
    res.precision_no_video, res.precision_with_video = _mean(prec_no), _mean(prec_yes)
    res.recall_no_video, res.recall_with_video = _mean(rec_no), _mean(rec_yes)
    res.fp_no_video, res.fp_with_video = _mean(fp_no), _mean(fp_yes)
    res.fn_no_video, res.fn_with_video = _mean(fn_no), _mean(fn_yes)

    paired = [y - n for y, n in zip(f1_yes, f1_no)
              if np.isfinite(y) and np.isfinite(n)]
    res.gain = float(np.mean(paired)) if paired else float("nan")
    res.gain_ci95 = _ci95(paired)
    res.p_value = _paired_p(paired)
    res.significant = bool(
        len(paired) >= 2 and np.isfinite(res.gain) and abs(res.gain) > res.gain_ci95)
    return res


def _match_behavior_ids(project: ProjectRef, names: list[str]) -> list[str]:
    return project.behavior_ids_matching(names)


def run_analysis(
    project_roots: list[str],
    behavior_names: list[str],
    *,
    n_seeds: int = 5,
    min_confidence: float = 1.0,
    holdout_test_size: float = 0.25,
    holdout_seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> list[VideoValueResult]:
    """Run the with/without-video comparison for the named behaviors across projects."""
    trainer = ActiveLearningTrainerService()
    results: list[VideoValueResult] = []
    for root in project_roots:
        project = ProjectRef.load(root)
        if not project.is_valid():
            if progress_cb:
                progress_cb(f"SKIP (no training set): {root}")
            continue
        bids = _match_behavior_ids(project, behavior_names)
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
                results.append(VideoValueResult(
                    project_id=project.project_id, behavior_id=str(bid),
                    behavior_name=project.behavior_label(bid),
                    error=f"holdout failed: {exc}"))
            continue
        for bid in bids:
            results.append(run_video_value(
                trainer, project, str(bid), hsplit,
                n_seeds=n_seeds, progress_cb=progress_cb))
    return results


def results_to_frame(results: list[VideoValueResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def plot_video_value(results: list[VideoValueResult], save_path: Path) -> Path:
    """Paired dumbbells — F1 without → with video features, per (project, behavior).

    Two panels.  Left: for each behavior, a line from the pose-only F1 to the
    +video F1, so the *pairing* — the whole point of the design — is the visual
    primitive rather than something the reader has to infer by comparing the
    heights of two adjacent bars.  Right: the paired gain with its 95% CI across
    seeds, which is the quantity the significance claim is actually about.

    Side-by-side bars were the previous layout; they scale the figure width with
    the number of behaviors (unreadable past ~10), spend most of the panel on the
    empty 0–0.5 region no F1 ever visits, and bury the paired Δ in a tiny
    annotation.  Rows are sorted by gain, so the behaviors video features rescue
    sit at the top.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in results if not r.error and np.isfinite(r.f1_no_video)]
    save_path = Path(save_path)
    if not usable:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No usable results\n(check labels / video features)",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    usable = sorted(usable, key=lambda r: (not np.isfinite(r.gain), -float(r.gain)))
    n = len(usable)
    labels = [f"{r.project_id[:16]} · {r.behavior_name}" for r in usable]
    y = np.arange(n)[::-1]
    no_v = np.array([r.f1_no_video for r in usable], dtype=float)
    with_v = np.array([r.f1_with_video for r in usable], dtype=float)
    gains = np.array([r.gain for r in usable], dtype=float)
    gain_ci = np.array([r.gain_ci95 for r in usable], dtype=float)
    sig = [bool(r.significant) for r in usable]

    C_OFF, C_ON = "#9E9E9E", "#C44E52"
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, max(3.2, 0.30 * n + 1.9)),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.06}, sharey=True)

    # ── Left: paired dumbbell, pose-only → +video.
    for yi, a, b, s in zip(y, no_v, with_v, sig):
        ax1.plot([a, b], [yi, yi], color=(C_ON if s else "#CFD8DC"),
                 linewidth=2.0 if s else 1.4, alpha=0.9 if s else 0.7, zorder=1,
                 solid_capstyle="round")
    ax1.scatter(no_v, y, s=34, color=C_OFF, edgecolor="white", linewidth=0.7,
                zorder=3, label="Pose only (no video motion)")
    ax1.scatter(with_v, y, s=34, color=C_ON, edgecolor="white", linewidth=0.7,
                zorder=3, label="+ Video motion features")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    # Frame the region F1 actually occupies — starting at 0 spends most of the
    # panel on empty space and flattens every difference the figure is about.
    lo = float(np.nanmin([no_v.min(), with_v.min()]))
    ax1.set_xlim(max(0.0, lo - 0.06), 1.0)
    ax1.set_xlabel("F1 (held-out, target-vs-rest)", fontsize=9)
    ax1.set_title("Paired: video features off → on", fontsize=10, loc="left")
    ax1.legend(loc="lower left", fontsize=8, frameon=False)
    ax1.grid(axis="x", alpha=0.22)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # ── Right: the paired gain and its CI — the quantity the asterisks are about.
    ax2.barh(y, gains, 0.62, xerr=gain_ci,
             color=[(C_ON if s else "#CFD8DC") for s in sig],
             edgecolor="white", linewidth=0.4,
             error_kw={"elinewidth": 0.8, "ecolor": "#455A64", "capsize": 2})
    ax2.axvline(0, color="#546E7A", linewidth=1.0)
    # Mirror the label to the far side for negative gains — always writing it to the
    # right of the bar puts it *on top of* a leftward bar and its error whisker.
    span = float(np.nanmax(np.abs(gains) + gain_ci)) if len(gains) else 1.0
    pad = max(span * 0.03, 1e-3)
    for yi, g, c, s in zip(y, gains, gain_ci, sig):
        if np.isfinite(g):
            side = 1.0 if g >= 0 else -1.0
            ax2.text(g + side * (c + pad), yi, f"{g:+.3f}" + ("*" if s else ""),
                     va="center", ha="left" if g >= 0 else "right", fontsize=7,
                     color="#263238" if s else "#90A4AE")
    ax2.set_xlabel("ΔF1 from video features  (paired, 95% CI across seeds)", fontsize=9)
    ax2.set_title("Gain  (* = CI excludes 0)", fontsize=10, loc="left")
    ax2.grid(axis="x", alpha=0.22)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    # Room for the value labels themselves, which sit *beyond* the error whiskers —
    # a plain margin sized to the bars alone clips the widest label.
    ends = np.concatenate([gains - gain_ci, gains + gain_ci])
    ends = ends[np.isfinite(ends)]
    if ends.size:
        lo_e, hi_e = float(ends.min()), float(ends.max())
        rng = max(hi_e - lo_e, 1e-4)
        ax2.set_xlim(min(lo_e, 0.0) - 0.34 * rng, max(hi_e, 0.0) + 0.34 * rng)

    fig.suptitle("Value of video motion features "
                 "(paired: same split & subsample, video features on vs. off)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--projects", nargs="+", required=True,
                    help="One or more ABEL project roots.")
    ap.add_argument("--behaviors", nargs="+", default=["Groom", "Freeze"],
                    help="Behavior display names to compare (default: Groom Freeze).")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=1.0)
    ap.add_argument("--out", default="./video_value_out",
                    help="Output directory for the CSV + PNG.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(
        args.projects, args.behaviors, n_seeds=args.seeds,
        min_confidence=args.min_confidence,
        progress_cb=lambda m: print(m, flush=True))

    df = results_to_frame(results)
    csv_path = out_dir / "video_value.csv"
    df.to_csv(csv_path, index=False)
    png_path = plot_video_value(results, out_dir / "video_value.png")
    prism.write_all(out_dir, video_df=df)

    print("\n=== Video-motion-feature value ===", flush=True)
    cols = ["project_id", "behavior_name", "f1_no_video", "f1_with_video",
            "gain", "gain_ci95", "significant", "error"]
    show = df[[c for c in cols if c in df.columns]] if not df.empty else df
    print(show.to_string(index=False) if not show.empty else "(no results)", flush=True)
    print(f"\nCSV : {csv_path}\nPlot: {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
