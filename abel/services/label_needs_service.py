"""What each behavior model needs next: a leak-free label ablation.

For every chosen behavior, the project's labeled windows are split into
session-grouped folds. In each fold the training rows are thinned one way at a
time and the whole held-out fold is scored, so the drop in PR-AUC says how
much that kind of label is worth right now:

- ``pos50``   half the positives, drawn at random across sessions
- ``conc50``  half the positives, drawn as whole sessions (same count, fewer animals)
- ``hard0``   no hard negatives (windows labeled as the behaviors this model
              most often mistakes for its target, found from its own
              held-out false positives)
- ``nobeh50`` half the No Behavior negatives

A COACIECNO run of the full version (2026-09) found: No Behavior negatives
never mattered, positives spread over many animals beat the same count from a
few, and hard negatives were worth 0.03-0.08 PR-AUC for the confusable
behaviors. Differences under ~0.03-0.04 were run-to-run noise, hence
``MEANINGFUL_GAIN``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from abel.services.behavior_service import NO_BEHAVIOR_ID
from abel.storage.file_store import read_json
from abel.utils.cancellation import check_cancel

logger = logging.getLogger(__name__)

# 0.04, not 0.03: at the default 2 repeats, a COACIECNO run called 0.037-0.039 gains
# (Allogroom, Rear, Flee) that the 3-repeat full test put at 0.01-0.02.
MEANINGFUL_GAIN = 0.04
MIN_POSITIVES = 20
FLAT_PRAUC = 0.45          # below this with no gain from positives: labels are not the fix
COVERED_SESSION_SHARE = 0.8  # positives in >= this share of sessions: coverage is fine
TOP3_CONCENTRATED = 0.5      # top 3 sessions hold >= this share of positives: concentrated


@dataclass
class LabelNeedsConfig:
    folds: int = 5
    repeats: int = 2
    n_estimators: int = 150
    seed: int = 0


@dataclass
class BehaviorNeeds:
    behavior_id: str
    behavior_name: str
    positives: int
    sessions_with_positives: int
    sessions_total: int
    top3_session_share: float
    prauc_full: float | None = None
    gain_positives: float | None = None     # full - pos50
    gain_spread: float | None = None        # pos50 - conc50
    gain_hard: float | None = None          # full - hard0
    gain_no_behavior: float | None = None   # full - nobeh50
    confusers: list[str] = field(default_factory=list)
    verdict: str = ""
    actions: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        row = asdict(self)
        row["confusers"] = ", ".join(self.confusers)
        row["actions"] = " | ".join(self.actions)
        return row


def _label_sets(labels: pd.Series) -> list[set[str]]:
    return [set(str(v).split("|")) for v in labels]


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.sum() else float("nan")


def _feature_columns(project_root: Path, behavior_id: str, df: pd.DataFrame) -> list[str]:
    """The behavior's own trained feature set when there is one, else every numeric feature."""
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.services.behavior_representation_service import align_model_feature_columns

    models = project_root / "derived" / "models"
    if models.exists():
        for d in sorted(models.glob("behavior_model_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            settings = read_json(d / "run_settings.json", {})
            tb = str(settings.get("target_behavior") or settings.get("target_behavior_id") or "")
            if tb != behavior_id or not (d / "model_state.pkl").exists():
                continue
            try:
                with open(d / "model_state.pkl", "rb") as fh:
                    cols = [str(c) for c in pickle.load(fh).get("feature_cols") or []]
                aligned, _ = align_model_feature_columns(cols, set(df.columns))
                aligned = [c for c in aligned if c in df.columns]
                if aligned:
                    return aligned
            except Exception:
                logger.debug("Label needs: could not read %s", d, exc_info=True)
    return ActiveLearningTrainerService._numeric_feature_cols(df)


class _Fitter:
    """XGBoost on GPU when it works, CPU otherwise (decided once)."""

    def __init__(self, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        try:
            from abel.utils.gpu_feature_ops import gpu_available

            self.device = "cuda" if gpu_available() else "cpu"
        except Exception:
            self.device = "cpu"

    def __call__(self, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, rep: int) -> np.ndarray:
        import xgboost as xgb

        if y_tr.sum() == 0:
            return np.zeros(len(x_te))
        params = dict(
            n_estimators=self.n_estimators, max_depth=6, learning_rate=0.07, subsample=0.8,
            colsample_bytree=0.5, tree_method="hist", eval_metric="logloss", random_state=self.seed + rep,
        )
        try:
            m = xgb.XGBClassifier(device=self.device, **params)
            m.fit(x_tr, y_tr)
        except Exception:
            if self.device == "cpu":
                raise
            logger.info("Label needs: GPU fit failed, using CPU", exc_info=True)
            self.device = "cpu"
            m = xgb.XGBClassifier(device="cpu", **params)
            m.fit(x_tr, y_tr)
        m.set_params(device="cpu")
        return m.predict_proba(x_te)[:, 1]


def _false_positive_confusers(
    y: np.ndarray, p: np.ndarray, labsets: list[set[str]], names: dict[str, str], behavior_id: str, top: int = 2
) -> list[str]:
    """Behavior ids most often scored as this behavior among held-out negatives."""
    from abel.temporal_refinement.onset_autotune import _GRID, _f1_at, _plateau_midpoint

    f1 = np.array([_f1_at(y, p, t) for t in _GRID])
    fp = np.where((y == 0) & (p >= _plateau_midpoint(f1)))[0]
    counts: dict[str, int] = {}
    for i in fp:
        for lab in labsets[i]:
            if lab in names and lab not in (behavior_id, NO_BEHAVIOR_ID):
                counts[lab] = counts.get(lab, 0) + 1
    return [b for b, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def _verdict(n: BehaviorNeeds, names: dict[str, str]) -> None:
    g = MEANINGFUL_GAIN
    confuser_names = ", ".join(names.get(c, c) for c in n.confusers) or "its look-alikes"
    under_covered = (
        n.sessions_with_positives < COVERED_SESSION_SHARE * n.sessions_total
        or n.top3_session_share >= TOP3_CONCENTRATED
    )
    actions: list[str] = []
    if n.gain_positives is not None and n.gain_positives >= g:
        if under_covered:
            actions.append("More positives, from subjects that have none yet (coverage-gap hunt)")
        else:
            actions.append("More positives")
    elif (n.gain_spread is not None and n.gain_spread >= g and under_covered
          and (n.prauc_full is None or n.prauc_full >= FLAT_PRAUC)):
        # A weak model that more positives do not lift will not be rescued by
        # spreading them either (Allogroom: flat 0.29-0.33 from 25% to 100%).
        actions.append("Positives from subjects that have none yet (coverage-gap hunt)")
    if n.gain_hard is not None and n.gain_hard >= g:
        actions.append(f"Hard negatives: review clips of {confuser_names} and label them correctly")
    if n.gain_no_behavior is not None and n.gain_no_behavior >= g:
        actions.append("More No Behavior negatives")
    n.actions = actions
    if actions:
        n.verdict = "Label more"
    elif n.prauc_full is not None and n.prauc_full < FLAT_PRAUC:
        n.verdict = "Flat"
        n.actions = [
            "More of the same labels will not fix this model. Consider merging it with "
            f"{confuser_names}, tightening its definition, or dropping it."
        ]
    else:
        n.verdict = "Saturated"
        n.actions = ["Labeling more is unlikely to help. Spend review time on other behaviors."]


def analyze_label_needs(
    project_root: Path,
    behaviors: list[tuple[str, str]],
    config: LabelNeedsConfig | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
    cancel_flag: list[bool] | None = None,
    training_set: pd.DataFrame | None = None,
    behavior_names: dict[str, str] | None = None,
) -> list[BehaviorNeeds]:
    """Run the ablation for each ``(behavior_id, name)``; nothing in the project is written.

    ``behavior_names`` lists every project behavior, so look-alikes can be named
    even when they are not among the behaviors being tested.
    """
    from sklearn.model_selection import GroupKFold

    cfg = config or LabelNeedsConfig()
    project_root = Path(project_root)
    df = training_set
    if df is None:
        path = project_root / "derived" / "training_sets" / "training_set.parquet"
        if not path.exists():
            raise FileNotFoundError("No training set yet. Train a model first.")
        df = pd.read_parquet(path)
    df = df.reset_index(drop=True)
    labsets = _label_sets(df["label"])
    groups = df["session_id"].astype(str).to_numpy()
    sessions_total = len(np.unique(groups))
    no_beh = np.array([NO_BEHAVIOR_ID in s for s in labsets])
    names = {**(behavior_names or {}), **dict(behaviors)}
    fitter = _Fitter(cfg.n_estimators, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_folds = min(cfg.folds, sessions_total)
    folds = list(GroupKFold(n_folds).split(np.zeros(len(df)), groups=groups)) if n_folds >= 2 else []
    fits_per_behavior = n_folds * (1 + 4 * cfg.repeats)
    total = max(1, fits_per_behavior * len(behaviors))
    done = 0
    out: list[BehaviorNeeds] = []

    def tick(name: str) -> None:
        nonlocal done
        done += 1
        check_cancel(cancel_flag)
        if progress_cb is not None:
            progress_cb(name, done, total)

    for bid, name in behaviors:
        y = np.array([bid in s for s in labsets], dtype=int)
        per_sess = pd.Series(groups[y == 1]).value_counts()
        n = BehaviorNeeds(
            bid, name, int(y.sum()), int(len(per_sess)), sessions_total,
            float(per_sess.head(3).sum() / max(1, y.sum())),
        )
        if n.positives < MIN_POSITIVES or not folds:
            n.verdict = "Too few"
            n.actions = [f"Only {n.positives} positives; label at least {MIN_POSITIVES} across several subjects first."]
            done += fits_per_behavior
            if progress_cb is not None:
                progress_cb(name, done, total)
            out.append(n)
            continue
        X = df.reindex(columns=_feature_columns(project_root, bid, df)).to_numpy(np.float32)

        full = np.zeros(len(y))
        for tr, te in folds:
            full[te] = fitter(X[tr], y[tr], X[te], 0)
            tick(name)
        n.prauc_full = _pr_auc(y, full)
        n.confusers = _false_positive_confusers(y, full, labsets, names, bid)
        hard = np.array([bool(s & set(n.confusers)) for s in labsets]) & (y == 0)

        arms = {k: [np.zeros(len(y)) for _ in range(cfg.repeats)] for k in ("pos50", "conc50", "hard0", "nobeh50")}
        for tr, te in folds:
            pos_tr = tr[y[tr] == 1]
            for rep in range(cfg.repeats):
                for arm, preds in arms.items():
                    keep = np.ones(len(y), bool)
                    if arm == "pos50":
                        keep[rng.choice(pos_tr, len(pos_tr) // 2, replace=False)] = False
                    elif arm == "conc50":
                        kept, need = 0, len(pos_tr) - len(pos_tr) // 2
                        for s in rng.permutation(np.unique(groups[pos_tr])):
                            rows = pos_tr[groups[pos_tr] == s]
                            if kept >= need:
                                keep[rows] = False
                            else:
                                kept += len(rows)
                    elif arm == "hard0":
                        keep[tr[hard[tr]]] = False
                    else:
                        pool = tr[no_beh[tr] & (y[tr] == 0)]
                        keep[rng.choice(pool, len(pool) // 2, replace=False)] = False
                    k = tr[keep[tr]]
                    preds[rep][te] = fitter(X[k], y[k], X[te], rep + 1)
                    tick(name)
        score = {arm: float(np.nanmean([_pr_auc(y, p) for p in preds])) for arm, preds in arms.items()}
        n.gain_positives = n.prauc_full - score["pos50"]
        n.gain_spread = score["pos50"] - score["conc50"]
        n.gain_hard = n.prauc_full - score["hard0"] if hard.any() else None
        n.gain_no_behavior = n.prauc_full - score["nobeh50"] if no_beh.any() else None
        _verdict(n, names)
        out.append(n)
    return out
