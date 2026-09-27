r"""How well does each clip-hunting tool work when the user starts with few positives?

The rare-discovery panel seeds every arm with 20 positives drawn at random from the
whole pool.  A user building a new behavior usually has far fewer, and the ones they
have tend to come from the one or two videos they happened to score first.  This
analysis re-runs the hunt from ``k0`` in {5, 10, 25, 50} starting positives, drawn
from as few sessions as can supply them (clustered, like a real start), and asks two
questions per arm:

* discovery: new positives found in the next 50/100/200 reviewed clips, and clips
  reviewed to find 25 more (censored at the budget, never dropped);
* model quality (``k0`` in {10, 25} only): held-out target-class F1 / PR-AUC of the
  model trained on the seed plus every clip reviewed so far.

Arms: Essence Miner, Active Learning (the AL tab's default queue, see
:func:`rare_discovery._shipped_al_order`), UMAP selection, random clips, and an
essence-then-AL hybrid that hunts with essence for the first ``hybrid_switch``
clips and hands everything it found (positives and rejects) to the AL queue.
Every arm shares the same seed positives and pool per replicate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
from abel.validation.analyses import al_curve
from abel.validation.analyses import rare_discovery as rd
from abel.validation.holdout import HoldoutSplit
from abel.validation.datamodel import ProjectRef

STRATEGY_HYBRID = "essence_then_al"
ARMS = (rd.STRATEGY_ESSENCE, rd.STRATEGY_AL, rd.STRATEGY_UMAP, rd.STRATEGY_RANDOM,
        STRATEGY_HYBRID)
ARM_LABELS = {rd.STRATEGY_ESSENCE: "Essence Miner", rd.STRATEGY_AL: "Active Learning",
              rd.STRATEGY_UMAP: "UMAP selection", rd.STRATEGY_RANDOM: "Random clips",
              STRATEGY_HYBRID: "Essence then AL"}


@dataclass
class StartingPositivesResult:
    project_id: str
    project_name: str
    behavior_id: str
    behavior_name: str
    n_pool: int
    n_pos_pool: int
    prevalence: float
    # rows: k0, arm, rep, n_seed_sessions, pos@50, pos@100, pos@200, clips_to_25,
    #       censored (did not reach +25 within the budget), n_pos_available
    discovery: list[dict] = field(default_factory=list)
    # rows: k0, arm, rep, n_clips, n_pos, f1, pr_auc
    quality: list[dict] = field(default_factory=list)
    skipped: dict = field(default_factory=dict)       # k0 -> reason


def _clustered_seed(pool: pd.DataFrame, pos_idx: np.ndarray, k0: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """``k0`` positives from as few sessions as possible, sessions in random order."""
    if "session_id" not in pool.columns:
        return rng.choice(pos_idx, size=k0, replace=False), 0
    sess = pool["session_id"].astype(str).to_numpy()
    by = pd.Series(pos_idx).groupby(sess[pos_idx]).apply(list).to_dict()
    names = list(by)
    rng.shuffle(names)
    chosen: list[int] = []
    used = 0
    for name in names:
        if len(chosen) >= k0:
            break
        idx = list(by[name])
        rng.shuffle(idx)
        chosen.extend(idx[:k0 - len(chosen)])
        used += 1
    return np.asarray(chosen, dtype=int), used


def _hybrid_order(trainer, project, behavior, cand, throwaway, seed_rows, metrics_all,
                  cand_idx, is_pos, seed_sel, *, seed, batch, budget, switch, log):
    e_order = rd._essence_discovery(
        metrics_all.iloc[cand_idx], is_pos, metrics_all.iloc[seed_sel],
        seed=seed, batch=batch, refit_budget=switch, log=log)
    if e_order is None:
        e_order = np.random.default_rng(seed).permutation(len(cand))
    head = np.asarray(e_order[:switch], dtype=int)
    rest = np.setdiff1d(np.arange(len(cand)), head)
    sub_cand = cand.iloc[rest].reset_index(drop=True)
    seed_plus = pd.concat([seed_rows, cand.iloc[head]], ignore_index=True)
    tail = rd._al_discovery(
        trainer, project, behavior, sub_cand, throwaway, seed_plus, seed=seed,
        batch=batch, max_reveal=max(0, budget - switch), log=log, warm_fill=0)
    return np.concatenate([head, rest[tail]])


def run_starting_positives(
    trainer: ActiveLearningTrainerService,
    project: ProjectRef,
    behavior_id: str,
    holdout_split: HoldoutSplit,
    *,
    k0_grid: tuple[int, ...] = (5, 10, 25, 50),
    quality_k0: tuple[int, ...] = (10, 25),
    n_reps: int = 3,
    batch: int = 25,
    budget: int = 200,
    quality_budget: int = 300,
    quality_reps: int = 3,
    quality_neg_fill: int = 10,
    hybrid_switch: int = 50,
    find_more: int = 25,
    arms: tuple[str, ...] = ARMS,
    cache: "rd.RareProjectCache | None" = None,
    progress_cb: Callable[[str], None] | None = None,
) -> StartingPositivesResult:
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    name = project.behavior_label(behavior_id)
    pool = holdout_split.train_pool.reset_index(drop=True)
    throwaway = holdout_split.holdout.reset_index(drop=True)
    pos_all = np.where(rd._pos_mask(pool, behavior_id))[0]
    out = StartingPositivesResult(
        project_id=getattr(project, "project_id", ""), project_name=project.name,
        behavior_id=behavior_id, behavior_name=name, n_pool=len(pool),
        n_pos_pool=len(pos_all), prevalence=len(pos_all) / max(1, len(pool)))

    metrics_all = (cache.metrics if cache is not None and cache.metrics is not None
                   else rd._essence_feature_frame(pool))
    if cache is not None and cache.embedding is not None:
        emb = cache.embedding
    else:
        cols = [c for c in pool.columns if c not in rd._META_COLS
                and pd.api.types.is_numeric_dtype(pool[c]) and bool(pool[c].notna().any())]
        emb = rd._embed(pool[cols].to_numpy(dtype=float), 9000)

    def at(c, k):
        return float(c[min(k, len(c)) - 1]) if len(c) else float("nan")

    for k0 in k0_grid:
        if len(pos_all) <= k0:
            out.skipped[k0] = f"only {len(pos_all)} positives in the pool"
            _log(f"{name}: k0={k0} skipped ({out.skipped[k0]})")
            continue
        for rep in range(n_reps):
            seed = 7000 + 100 * k0 + rep
            rng = np.random.default_rng(seed)
            seed_sel, n_sess = _clustered_seed(pool, pos_all, k0, rng)
            seed_set = set(int(i) for i in seed_sel)
            cand_idx = np.asarray([i for i in range(len(pool)) if i not in seed_set], dtype=int)
            cand = pool.iloc[cand_idx].reset_index(drop=True)
            is_pos = rd._pos_mask(cand, behavior_id)
            seed_rows = pool.iloc[seed_sel]
            _log(f"{name}: k0={k0} rep {rep + 1}/{n_reps} ({n_sess} sessions), "
                 f"{int(is_pos.sum())} positives left in {len(cand)}")
            for arm in arms:
                if arm == rd.STRATEGY_RANDOM:
                    order = rng.permutation(len(cand))
                elif arm == rd.STRATEGY_ESSENCE:
                    order = rd._essence_discovery(
                        metrics_all.iloc[cand_idx], is_pos, metrics_all.iloc[seed_sel],
                        seed=seed, batch=batch, refit_budget=budget, log=_log)
                    if order is None:
                        order = rng.permutation(len(cand))
                elif arm == rd.STRATEGY_UMAP:
                    order = rd._umap_discovery(emb[cand_idx], is_pos, emb[seed_sel],
                                               seed=seed, batch=batch, refit_budget=budget)
                elif arm == rd.STRATEGY_AL:
                    order = rd._al_discovery(
                        trainer, project, behavior_id, cand, throwaway, seed_rows,
                        seed=seed, batch=batch, max_reveal=budget, log=_log)
                else:
                    order = _hybrid_order(
                        trainer, project, behavior_id, cand, throwaway, seed_rows,
                        metrics_all, cand_idx, is_pos, seed_sel, seed=seed, batch=batch,
                        budget=budget, switch=hybrid_switch, log=_log)
                curve = rd._discovered_curve(is_pos[np.asarray(order, dtype=int)[:budget]])
                hit = np.nonzero(curve >= find_more)[0]
                out.discovery.append({
                    "k0": k0, "arm": arm, "rep": rep, "n_seed_sessions": n_sess,
                    "pos@50": at(curve, 50), "pos@100": at(curve, 100),
                    "pos@200": at(curve, 200),
                    "clips_to_25": float(hit[0] + 1) if len(hit) else float("nan"),
                    "censored": not len(hit), "n_pos_available": int(is_pos.sum()),
                    "curve": curve.astype(np.int32),
                })
                _log(f"{name}: k0={k0} rep {rep + 1} {arm}: +{at(curve, 100):.0f} in 100")

    for k0 in quality_k0:
        if len(pos_all) <= k0:
            continue
        for rep in range(quality_reps):
            seed = 8000 + 100 * k0 + rep
            rng = np.random.default_rng(seed)
            seed_sel, _ = _clustered_seed(pool, pos_all, k0, rng)
            neg = np.setdiff1d(np.where(~rd._pos_mask(pool, behavior_id))[0], seed_sel)
            fill = rng.choice(neg, size=min(quality_neg_fill, len(neg)), replace=False)
            for arm in arms:
                out.quality.extend(_quality_traj(
                    trainer, project, behavior_id, pool, throwaway, arm,
                    set(int(i) for i in np.concatenate([seed_sel, fill])),
                    metrics_all, emb, seed=seed, batch=batch, budget=quality_budget,
                    switch=hybrid_switch, k0=k0, rep=rep, log=_log))
    return out


def _quality_traj(trainer, project, behavior, pool, holdout, arm, labeled, metrics_all,
                  emb, *, seed, batch, budget, switch, k0, rep, log,
                  switch_pos: int | None = None, switch_cap: int = 100):
    rng = np.random.default_rng(seed + 1)
    pos_arr = rd._pos_mask(pool, behavior)
    start = len(labeled)
    cap = min(start + int(budget), len(pool))
    rows: list[dict] = []
    density_cache: dict = {}
    phase = "essence" if arm == STRATEGY_HYBRID else arm
    while True:
        idx = sorted(labeled)
        f1 = pr = float("nan")
        res = None
        try:
            res = al_curve._fit(trainer, project, behavior, pool.iloc[idx], holdout, seed)
            f1, pr = rd._target_metrics(res, holdout, behavior)
        except Exception as exc:  # noqa: BLE001
            log(f"quality {arm} k0={k0}: fit failed ({type(exc).__name__}) at {len(idx)}")
        rows.append({"k0": k0, "arm": arm, "rep": rep, "n_clips": len(idx) - start,
                     "n_pos": int(pos_arr[idx].sum()), "f1": f1, "pr_auc": pr,
                     "phase": phase})
        if len(labeled) >= cap:
            break
        remaining = [i for i in range(len(pool)) if i not in labeled]
        if not remaining:
            break
        strat = arm
        if arm == STRATEGY_HYBRID:
            if switch_pos is None:
                hunting = len(labeled) - start < switch
            else:
                # Hunt with essence until the labeled set holds switch_pos positives
                # (or switch_cap clips were spent hunting), then hand over to AL.
                hunting = (phase == "essence" and int(pos_arr[list(labeled)].sum()) < switch_pos
                           and len(labeled) - start < switch_cap)
            if not hunting:
                phase = "active_learning"
            strat = rd.STRATEGY_ESSENCE if hunting else rd.STRATEGY_AL
        lab_pos = [i for i in labeled if pos_arr[i]]
        lab_neg = [i for i in labeled if not pos_arr[i]]
        order = rd._acquire_next(
            strat, remaining, pool, lab_pos, res, metrics_all=metrics_all, umap_emb=emb,
            rng=rng, labeled_neg=lab_neg, project=project, behavior=behavior,
            density_cache=density_cache)
        take = min(batch, len(remaining), cap - len(labeled))
        labeled.update(int(remaining[k]) for k in order[:take])
    log(f"quality {arm} k0={k0} rep {rep + 1}: F1 {rows[-1]['f1']:.3f} after "
        f"{rows[-1]['n_clips']} clips")
    return rows


EFFORT_ARMS = (rd.STRATEGY_ESSENCE, rd.STRATEGY_AL, STRATEGY_HYBRID)


def run_effort_quality(trainer, project, behavior_id, holdout_split, *, k0: int = 10,
                       n_reps: int = 2, budget: int = 200, batch: int = 25,
                       neg_fill: int = 10, switch_pos: int = 50, switch_cap: int = 100,
                       cache=None, progress_cb=None,
                       arms: tuple[str, ...] = EFFORT_ARMS) -> list[dict]:
    """Held-out F1 vs clips reviewed for AL alone, essence alone and essence-then-AL.

    Every arm starts from the same ``k0`` clustered positives plus ``neg_fill`` random
    negatives. The hybrid hunts with essence until the labeled set holds
    ``switch_pos`` positives (or ``switch_cap`` clips were spent), then uses the AL
    tab queue. Rows: rep, arm, n_clips, n_pos, f1, pr_auc, phase.
    """
    log = progress_cb or (lambda _m: None)
    pool = holdout_split.train_pool.reset_index(drop=True)
    holdout = holdout_split.holdout.reset_index(drop=True)
    pos_all = np.where(rd._pos_mask(pool, behavior_id))[0]
    if len(pos_all) <= k0:
        return []
    metrics_all = (cache.metrics if cache is not None and cache.metrics is not None
                   else rd._essence_feature_frame(pool))
    rows: list[dict] = []
    for rep in range(n_reps):
        seed = 8000 + 100 * k0 + rep
        rng = np.random.default_rng(seed)
        seed_sel, _ = _clustered_seed(pool, pos_all, k0, rng)
        neg = np.setdiff1d(np.where(~rd._pos_mask(pool, behavior_id))[0], seed_sel)
        fill = rng.choice(neg, size=min(neg_fill, len(neg)), replace=False)
        start = set(int(i) for i in np.concatenate([seed_sel, fill]))
        for arm in arms:
            rows.extend(_quality_traj(
                trainer, project, behavior_id, pool, holdout, arm, set(start), metrics_all,
                None, seed=seed, batch=batch, budget=budget, switch=switch_cap, k0=k0,
                rep=rep, log=log, switch_pos=switch_pos, switch_cap=switch_cap))
    return rows
