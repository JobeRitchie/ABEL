"""Suggest per-behavior onset thresholds from held-out validation predictions.

Only ``validation_predictions.parquet`` is read: it holds the leak-free
per-window probabilities of the held-out sessions. The F1-versus-threshold
curve on those windows is usually flat near its peak, so its raw argmax fits
noise; the plateau midpoint (every threshold within ``PEAK_TOLERANCE`` of peak
F1) is the per-behavior signal.

That signal is only weakly related to the best threshold on the deployed trace.
Bouts are cut from a trace that averages overlapping windows, is smoothed and
has mutual inhibition subtracted, and it runs over whole sessions instead of
labeled windows. In a leak-free benchmark (2026-10, 120 behaviors in 20
projects, every model retrained 5-fold by subject, traces rebuilt with each
project's inference settings and scored on labeled stretches plus random probe
windows) the best trace threshold ranged from 0.02 to 0.85, and no per-behavior
estimate tracked it well:

- the window plateau midpoint alone lost 0.037 score on average (worst 0.30),
- the previous rule (midpoint capped at the 10th percentile of labeled bout
  peaks) lost 0.033 (worst 0.31), and its bout-peak cap was the worst part,
- tuning on 5-fold rebuilt traces, cross-fitted, lost 0.021 to 0.026,
- a constant 0.2 lost 0.021 (worst 0.15),
- ``SHRINK_PRIOR`` blended with ``SHRINK_WEIGHT`` of the plateau midpoint lost
  0.017 (worst 0.09), chosen leave-one-project-out, and it also won when scored
  on the random probe windows alone.

So the suggestion is the window midpoint shrunk hard toward a global prior. The
best constant was about 0.2 with or without inhibition, so the prior does not
depend on the inhibition weight. ``recall_beta`` > 1 tunes the window curve for
F-beta instead of F1 before shrinking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from abel.storage.file_store import read_json, read_yaml

PEAK_TOLERANCE = 0.01
# Suggestions closer than this to the current threshold are not worth a change.
MIN_THRESHOLD_MOVE = 0.01
RECALL_BETA = 1.5
MIN_VAL_POSITIVES = 5
LOW_POSITIVES_WARNING = 30
# Suggested = (1 - SHRINK_WEIGHT) * SHRINK_PRIOR + SHRINK_WEIGHT * window midpoint.
SHRINK_PRIOR = 0.1
SHRINK_WEIGHT = 0.25
# A window optimum that flags this many times more windows than are truly
# positive comes from a weak model; the suggestion still stands, with a warning.
MAX_PREDICTED_OVER_TRUE = 3.0
_GRID = np.round(np.arange(0.01, 0.96, 0.005), 3)
# Bout cleanup from the leak-free 108-behavior audit (2026-09): the surface is flat
# for merge 0-1 s and min bout 0-0.6 s, and only min bout >= 1 s clearly hurts.
# Sparse window labels cannot support fitting these per behavior.
AUDITED_MIN_BOUT_SEC = 0.4
AUDITED_MERGE_GAP_SEC = 0.2


@dataclass(frozen=True)
class OnsetSuggestion:
    behavior_id: str
    behavior_name: str
    current: float
    suggested: float | None
    f1_current: float | None
    f1_suggested: float | None
    val_positives: int
    note: str

    @property
    def is_change(self) -> bool:
        return self.suggested is not None


def _fbeta_at(y: np.ndarray, p: np.ndarray, thr: float, beta: float = 1.0) -> float:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / denom if denom else 0.0


def _f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    return _fbeta_at(y, p, thr, 1.0)


def _plateau_midpoint(f1: np.ndarray) -> float:
    """Middle of the contiguous run of near-peak thresholds that holds the peak."""
    near = f1 >= f1.max() - PEAK_TOLERANCE
    peak = int(np.argmax(f1))
    lo = hi = peak
    while lo > 0 and near[lo - 1]:
        lo -= 1
    while hi < len(f1) - 1 and near[hi + 1]:
        hi += 1
    return float(_GRID[(lo + hi) // 2])


def _model_dirs_by_behavior(project_root: Path) -> dict[str, Path]:
    """Map target behavior id to its model directory via ``run_settings.json``.

    Directory names are display names (or custom names), so the recorded
    target id is the only reliable key.
    """
    out: dict[str, Path] = {}
    models_root = project_root / "derived" / "models"
    if not models_root.exists():
        return out
    for p in sorted(models_root.iterdir()):
        if not (p.is_dir() and p.name.startswith("behavior_model_")):
            continue
        settings = read_json(p / "run_settings.json", {})
        tb = str(settings.get("target_behavior") or settings.get("target_behavior_id") or "").strip()
        if tb and (p / "validation_predictions.parquet").exists():
            prev = out.get(tb)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                out[tb] = p
    return out


def project_fps(project_root: Path) -> float:
    raw = read_yaml(Path(project_root) / "project.yaml", {})
    try:
        fps = float(raw.get("default_fps") or 30.0)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0 else 30.0


def suggest_bout_cleanup(fps: float) -> tuple[int, int]:
    """``(min_bout_frames, merge_gap_frames)`` at the audited optimum."""
    return max(1, round(AUDITED_MIN_BOUT_SEC * fps)), max(0, round(AUDITED_MERGE_GAP_SEC * fps))


def shrink_threshold(window_midpoint: float) -> float:
    return round((1.0 - SHRINK_WEIGHT) * SHRINK_PRIOR + SHRINK_WEIGHT * float(window_midpoint), 3)


def suggest_for_predictions(
    behavior_id: str,
    behavior_name: str,
    current: float,
    predictions: pd.DataFrame,
    recall_beta: float = 1.0,
) -> OnsetSuggestion:
    def result(suggested, f1_cur, f1_new, n_pos, note):
        return OnsetSuggestion(behavior_id, behavior_name, float(current), suggested, f1_cur, f1_new, n_pos, note)

    needed = {"label_true", "prediction_prob", "target_index"}
    if predictions.empty or not needed.issubset(predictions.columns):
        return result(None, None, None, 0, "No usable validation predictions.")
    target = int(predictions["target_index"].iloc[0])
    y = (predictions["label_true"].to_numpy() == target).astype(int)
    p = predictions["prediction_prob"].to_numpy(dtype=float)
    n_pos = int(y.sum())
    if n_pos < MIN_VAL_POSITIVES:
        return result(None, None, None, n_pos, f"Only {n_pos} held-out positives; too few to tune.")

    score = np.array([_fbeta_at(y, p, t, recall_beta) for t in _GRID])
    midpoint = _plateau_midpoint(score)
    candidate = shrink_threshold(midpoint)
    f1_cur = _f1_at(y, p, current)
    f1_new = _f1_at(y, p, candidate)
    if abs(candidate - float(current)) < MIN_THRESHOLD_MOVE:
        return result(None, f1_cur, f1_new, n_pos, "Current threshold is already at the suggestion.")
    notes = [f"Held-out windows peak around {midpoint:.2f}; shrunk toward {SHRINK_PRIOR:.2f} for the dense trace."]
    ratio = float(np.mean(p >= midpoint)) / float(np.mean(y))
    if ratio > MAX_PREDICTED_OVER_TRUE:
        notes.append(f"Weak model: its best window threshold flags {ratio:.1f}x more windows than are positive.")
    if n_pos < LOW_POSITIVES_WARNING:
        notes.append(f"Only {n_pos} held-out positives; treat as provisional.")
    return result(candidate, f1_cur, f1_new, n_pos, " ".join(notes))


def suggest_onset_thresholds(
    project_root: Path,
    behaviors: list[tuple[str, str]],
    current: dict[str, float],
    recall_beta: float = 1.0,
) -> list[OnsetSuggestion]:
    """One suggestion per ``(behavior_id, name)``, using the newest model per behavior."""
    model_dirs = _model_dirs_by_behavior(Path(project_root))
    out: list[OnsetSuggestion] = []
    for bid, name in behaviors:
        cur = float(current.get(bid, 0.65))
        model_dir = model_dirs.get(bid)
        if model_dir is None:
            out.append(OnsetSuggestion(bid, name, cur, None, None, None, 0, "No trained model with validation predictions."))
            continue
        try:
            preds = pd.read_parquet(model_dir / "validation_predictions.parquet")
        except Exception:
            preds = pd.DataFrame()
        out.append(suggest_for_predictions(bid, name, cur, preds, recall_beta))
    return out
