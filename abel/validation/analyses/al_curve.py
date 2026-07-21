"""Active learning vs. random clip selection — does uncertainty sampling win?

The headline efficiency experiment.  Both strategies start from the *same* small
random seed set (per seed), evaluate on the *same* fixed high-confidence held-out
set, and grow the labeled set in equal-size batches.  They differ only in the
acquisition rule:

- **random**: add a random batch of pool clips (group-aware shuffle).
- **active_learning**: train, score the *remaining pool*, and add the
  top-ranked clips — the realistic ABEL loop.  The ranking mirrors ABEL's shipped
  candidate generation, whose ``candidate_score`` is the model's predicted
  probability of the target behavior (``candidate_service._rank_segments``: it
  surfaces *likely instances* for the human to confirm, not boundary-ambiguous
  windows).  A ``"uncertainty"`` acquisition (|p−0.5| small) is also selectable
  for comparison; on rare classes it underperforms — the known cold-start
  failure of naive uncertainty sampling.

The x-axis is **total clips reviewed** (real human labeling effort), so the win
condition is: active learning reaches a target F1 with fewer reviewed clips, and
discovers the rare positive clips faster.  No held-out row is ever acquired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.utils import xgb_predict
from abel.validation import subsample
from abel.validation.datamodel import CellResult, ProjectRef
from abel.validation.engine import build_config
from abel.validation.holdout import HoldoutSplit

STRATEGY_AL = "active_learning"
STRATEGY_RANDOM = "random"


@dataclass
class ALPoint:
    n_clips: int                 # total reviewed/labeled clips
    n_pos_mean: float            # positives discovered among them (mean across seeds)
    f1_mean: float
    f1_ci: float
    pr_auc_mean: float
    pr_auc_ci: float
    n_seeds: int


@dataclass
class ALvsRandomResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    al_points: list[ALPoint] = field(default_factory=list)
    random_points: list[ALPoint] = field(default_factory=list)
    cells: list[CellResult] = field(default_factory=list)

    def clips_to_target(self, points: list[ALPoint], target_frac: float = 0.95) -> float | None:
        """Smallest #clips where mean F1 ≥ target_frac × max(F1) over all points."""
        allf = [p.f1_mean for p in (self.al_points + self.random_points) if np.isfinite(p.f1_mean)]
        if not allf:
            return None
        thr = target_frac * max(allf)
        for p in sorted(points, key=lambda q: q.n_clips):
            if np.isfinite(p.f1_mean) and p.f1_mean >= thr:
                return float(p.n_clips)
        return None


def _ci95(values) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return 0.0
    from abel.validation import metrics as vmetrics  # noqa: PLC0415

    return vmetrics.ci95(vals)  # t-based 95% CI half-width


def _fit(trainer, project, behavior, sub, holdout, seed):
    """Train on ``sub`` (labeled pool subset), evaluate on the fixed holdout."""
    df = pd.concat([sub, holdout], ignore_index=True)
    tr = np.arange(len(sub), dtype=int)
    va = np.arange(len(sub), len(sub) + len(holdout), dtype=int)
    cfg = build_config(project, behavior, seed)
    return trainer.train_and_evaluate(
        df, cfg, project_root=project.root, precomputed_split=(tr, va),
    )


def _seed_set(pool: pd.DataFrame, behavior: str, k0: int, seed_pos: int,
              rng: np.random.Generator) -> set[int]:
    """The shared warm seed set (mimics ABEL's seed-example step).

    Guarantees ``seed_pos`` positive clips so the first model is non-degenerate,
    then fills to ``k0`` total with random clips.  Both AL and random arms receive
    the *identical* seed (per seed value), so the comparison isolates acquisition.
    """
    n = len(pool)
    pos_mask = (pool["label"].astype(str).str.strip() == str(behavior).strip()).to_numpy()
    pos_idx = np.where(pos_mask)[0]
    labeled: set[int] = set()
    n_pos = min(seed_pos, len(pos_idx))
    if n_pos:
        for i in rng.choice(pos_idx, size=n_pos, replace=False):
            labeled.add(int(i))
    target = max(k0, len(labeled))
    others = [i for i in range(n) if i not in labeled]
    rng.shuffle(others)
    for i in others:
        if len(labeled) >= target:
            break
        labeled.add(int(i))
    return labeled


def _acquisition_order(p_tar: np.ndarray, acquisition: str) -> np.ndarray:
    """Rank remaining-pool clips for acquisition (best first).

    ``probability`` (default, ABEL-faithful): highest predicted target probability
    first — surfaces likely positives, matching ``candidate_score = prediction_prob``.
    ``uncertainty``: boundary-uncertain (|p−0.5| small) first.
    """
    if acquisition == "uncertainty":
        return np.argsort(np.abs(p_tar - 0.5))          # closest to 0.5 first
    return np.argsort(-p_tar)                            # highest target prob first


def _run_strategy(
    trainer, project, behavior, pool, holdout, *,
    strategy: str, seed: int, k0: int, batch: int, max_budget: int, group_col: str,
    acquisition: str, seed_pos: int, log: Callable[[str], None],
) -> tuple[list[tuple[int, int, float, float]], list[CellResult]]:
    """One strategy, one seed → trajectory of (n_clips, n_pos, f1, pr_auc) + cells."""
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True)
    n = len(pool)
    cap = min(max_budget, n)
    behavior_name = project.behavior_label(behavior)
    pos_strip = pool["label"].astype(str).str.strip() == str(behavior).strip()

    labeled = _seed_set(pool, behavior, k0, seed_pos, rng)
    traj: list[tuple[int, int, float, float]] = []
    cells: list[CellResult] = []

    while True:
        idx = sorted(labeled)
        sub = pool.iloc[idx]
        n_pos = int(pos_strip.iloc[idx].sum())
        f1 = pr = float("nan")
        ti = None
        res = None
        err = ""
        try:
            res = _fit(trainer, project, behavior, sub, holdout, seed)
            f1 = float(res.metrics.get("f1", float("nan")))
            pr = float(res.metrics.get("pr_auc", float("nan")))
            ti = res.target_idx
        except Exception as exc:  # noqa: BLE001 — early tiny sets can be degenerate
            err = f"{type(exc).__name__}: {exc}"
        log(f"{behavior_name}: {strategy} seed {seed} — {len(idx)} clips ({n_pos} pos) F1={f1:.3f}")

        traj.append((len(idx), n_pos, f1, pr))
        cells.append(CellResult(
            project_id=project.project_id, project_name=project.name,
            behavior_id=str(behavior), behavior_name=behavior_name,
            analysis="al_curve", config_name=strategy,
            n_clips=len(idx), seed=int(seed),
            f1=f1, pr_auc=pr, n_pos_train=n_pos, n_neg_train=len(idx) - n_pos,
            degenerate=bool(err), error=err,
        ))

        if len(labeled) >= cap:
            break
        remaining = [i for i in range(n) if i not in labeled]
        if not remaining:
            break
        n_choose = min(batch, len(remaining), cap - len(labeled))

        if strategy == STRATEGY_AL and res is not None and ti is not None:
            rem_df = pool.iloc[remaining]
            try:
                probs = xgb_predict.predict_proba(
                    res.calibrated_model,
                    rem_df[res.feature_cols].to_numpy(dtype=float))
                p_tar = probs[:, int(ti)] if int(ti) < probs.shape[1] else probs.max(axis=1)
                order = _acquisition_order(p_tar, acquisition)
            except Exception:
                order = rng.permutation(len(remaining))
            chosen = [remaining[k] for k in order[:n_choose]]
        else:  # random (or AL fallback when the model failed)
            rng.shuffle(remaining)
            chosen = remaining[:n_choose]
        labeled.update(int(c) for c in chosen)

    return traj, cells


def _aggregate(seed_trajs: list[list[tuple[int, int, float, float]]]) -> list[ALPoint]:
    """Aggregate per-seed trajectories (aligned by checkpoint index) into points."""
    if not seed_trajs:
        return []
    n_steps = min(len(t) for t in seed_trajs)
    points: list[ALPoint] = []
    for s in range(n_steps):
        nclips = [t[s][0] for t in seed_trajs]
        npos = [t[s][1] for t in seed_trajs]
        f1s = [t[s][2] for t in seed_trajs]
        prs = [t[s][3] for t in seed_trajs]
        finite_f1 = [v for v in f1s if np.isfinite(v)]
        finite_pr = [v for v in prs if np.isfinite(v)]
        points.append(ALPoint(
            n_clips=int(round(float(np.mean(nclips)))),
            n_pos_mean=float(np.mean(npos)),
            f1_mean=float(np.mean(finite_f1)) if finite_f1 else float("nan"),
            f1_ci=_ci95(f1s),
            pr_auc_mean=float(np.mean(finite_pr)) if finite_pr else float("nan"),
            pr_auc_ci=_ci95(prs),
            n_seeds=len(finite_f1),
        ))
    return points


def run_al_vs_random(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    n_seeds: int = 3,
    k0: int = 20,
    batch: int = 15,
    max_budget: int = 200,
    acquisition: str = "probability",
    seed_pos: int = 5,
    progress_cb: Callable[[str], None] | None = None,
) -> ALvsRandomResult:
    """Compare active-learning vs random clip acquisition for one (project, behavior).

    ``acquisition`` selects the AL ranking: ``"probability"`` (default, ABEL-faithful
    candidate_score = predicted target probability) or ``"uncertainty"``.
    """
    behavior_name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool
    group_col = holdout_split.group_col
    result = ALvsRandomResult(project.project_id, str(behavior_id), behavior_name)

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    al_trajs: list[list] = []
    rand_trajs: list[list] = []
    for rep in range(n_seeds):
        seed = 7000 + rep
        # Same seed → identical starting set for a fair paired comparison.
        al_t, al_cells = _run_strategy(
            trainer, project, behavior_id, pool, holdout_split.holdout,
            strategy=STRATEGY_AL, seed=seed, k0=k0, batch=batch,
            max_budget=max_budget, group_col=group_col, acquisition=acquisition,
            seed_pos=seed_pos, log=_log)
        rnd_t, rnd_cells = _run_strategy(
            trainer, project, behavior_id, pool, holdout_split.holdout,
            strategy=STRATEGY_RANDOM, seed=seed, k0=k0, batch=batch,
            max_budget=max_budget, group_col=group_col, acquisition=acquisition,
            seed_pos=seed_pos, log=_log)
        al_trajs.append(al_t)
        rand_trajs.append(rnd_t)
        result.cells.extend(al_cells)
        result.cells.extend(rnd_cells)

    result.al_points = _aggregate(al_trajs)
    result.random_points = _aggregate(rand_trajs)
    return result
