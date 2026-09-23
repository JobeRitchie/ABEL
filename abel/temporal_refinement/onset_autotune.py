"""Suggest per-behavior onset thresholds from held-out validation predictions.

Only ``validation_predictions.parquet`` is scored: it holds the leak-free
per-window probabilities of the held-out sessions, and it scores labeled
windows only, so sparse labels are not interpolated into bouts. Onset is a
per-window decision, which is why it can be tuned this way while min-bout and
merge-gap cannot.

The F1-versus-threshold curve is usually flat near its peak, so the raw argmax
fits noise. The suggestion is the middle of the plateau around the peak (every
threshold within ``PEAK_TOLERANCE`` of peak F1). Its top edge is the wrong pick:
the dense trace averages overlapping windows, so a short bout peaks lower in the
trace than in any single window, and a threshold at the top of the plateau
misses it (Sniff Anogenital: flat 0.15-0.56, the top edge lost real bouts).

A suggestion is made whenever it moves the threshold, even at equal held-out F1:
on a flat curve the F1 gain is ~0, so gating on it froze every threshold at
wherever it first landed. ``recall_beta`` > 1 optimizes F-beta instead of F1 to
favor fewer missed bouts. In a leak-free COACIECNO test (5-fold by session,
scored on the averaged trace) beta 1.5 cost ~0.01 F1 for +0.08 recall and ~27%
fewer labeled subjects with no detected bout.

The held-out windows are single-window probabilities, but bouts are cut from
the averaged dense trace, where short or weak bouts peak lower. A threshold
tuned on windows can sit above the peaks of many real bouts, and those bouts
are never counted. So the suggestion is capped at the trace peak of the lowest
``1 - PEAK_COVERAGE`` of labeled bouts. In a leak-free COACIECNO test on
out-of-fold traces, capping at the 20-30th percentile of bout peaks kept F1
(0.664 to 0.665-0.669) while catching more labeled bouts (0.80 to 0.81-0.83)
and leaving fewer labeled animals with no bout. Lower caps (5-10th) cost F1.
The deployed trace has seen the labeled windows, so its peaks there run high
and a 10th-percentile cap on it is milder than the same cap on honest peaks.
Extending bouts to half their own peak, prominence peaks, a two-state Viterbi
and a second-stage bout classifier were also tested. None beat a single
threshold by more than 0.011 F1, so bout detection stays a single threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from abel.storage.file_store import read_json, read_yaml
from abel.temporal_refinement.bout_postprocess import smooth_probabilities

PEAK_TOLERANCE = 0.01
# Suggestions closer than this to the current threshold are not worth a change.
MIN_THRESHOLD_MOVE = 0.01
RECALL_BETA = 1.5
MIN_VAL_POSITIVES = 5
LOW_POSITIVES_WARNING = 30
# Suggestions that flag this many times more windows than are truly positive
# are "accept everything" optima from a weak model, not a real operating point.
MAX_PREDICTED_OVER_TRUE = 3.0
_GRID = np.round(np.arange(0.01, 0.96, 0.005), 3)
# A threshold that most frames of the deployed full-session trace already exceed
# would call the whole session positive, whatever the held-out windows say.
MAX_TRACE_FRACTION_ABOVE = 0.5
# At least this share of labeled bouts must peak at or above the threshold on the
# deployed trace. Needs MIN_CAP_BOUTS labeled bouts; positive windows closer than
# BOUT_JOIN_FRAMES count as one bout.
PEAK_COVERAGE = 0.9
MIN_CAP_BOUTS = 10
BOUT_JOIN_FRAMES = 15
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


def _latest_trace_dir(project_root: Path) -> Path | None:
    root = Path(project_root) / "derived" / "temporal_refinement"
    runs = [p for p in root.glob("*/inference_*") if (p / "animal_probability_traces").is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime) / "animal_probability_traces"


def _latest_trace_values(project_root: Path, behavior_id: str, max_files: int = 20) -> np.ndarray | None:
    """Deployed-model probabilities for *behavior_id* from the newest dense inference run."""
    trace_dir = _latest_trace_dir(project_root)
    if trace_dir is None:
        return None
    col = f"prob_{behavior_id}"
    values = []
    for f in sorted(trace_dir.glob("*.parquet"))[:max_files]:
        try:
            values.append(pd.read_parquet(f, columns=[col])[col].to_numpy(dtype=float))
        except Exception:
            continue
    return np.concatenate(values) if values else None


def _load_labels(project_root: Path) -> pd.DataFrame | None:
    path = Path(project_root) / "derived" / "training_sets" / "training_set.parquet"
    try:
        return pd.read_parquet(path, columns=["session_id", "animal_id", "start_frame", "end_frame", "label"])
    except Exception:
        return None


def _labeled_bout_peaks(
    project_root: Path, behavior_id: str, labels: pd.DataFrame | None = None,
) -> np.ndarray | None:
    """Smoothed deployed-trace peak of every labeled bout of *behavior_id*.

    Positive windows (labels may be pipe-joined) closer than BOUT_JOIN_FRAMES
    are joined into one bout per animal.
    """
    trace_dir = _latest_trace_dir(project_root)
    labels = _load_labels(project_root) if labels is None else labels
    if trace_dir is None or labels is None or labels.empty:
        return None
    is_pos = labels["label"].astype(str).str.split("|").apply(lambda parts: behavior_id in parts)
    pos = labels[is_pos.to_numpy(dtype=bool)]
    col = f"prob_{behavior_id}"
    peaks: list[float] = []
    for (sid, aid), grp in pos.groupby(["session_id", "animal_id"], dropna=False):
        candidates = [trace_dir / f"{sid}__{aid}_trace.parquet", trace_dir / f"{sid}_trace.parquet"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        try:
            tr = pd.read_parquet(path, columns=["frame", col]).sort_values("frame")
        except Exception:
            continue
        frames = tr["frame"].to_numpy(dtype=int)
        x = smooth_probabilities(np.nan_to_num(tr[col].to_numpy(dtype=np.float32), nan=0.0))
        bouts: list[list[int]] = []
        for s, e in sorted(zip(grp["start_frame"].astype(int), grp["end_frame"].astype(int))):
            if bouts and s <= bouts[-1][1] + BOUT_JOIN_FRAMES:
                bouts[-1][1] = max(bouts[-1][1], e)
            else:
                bouts.append([s, e])
        for s, e in bouts:
            lo, hi = np.searchsorted(frames, s), np.searchsorted(frames, e, side="right")
            if hi > lo:
                peaks.append(float(x[lo:hi].max()))
    return np.asarray(peaks) if peaks else None


def suggest_for_predictions(
    behavior_id: str,
    behavior_name: str,
    current: float,
    predictions: pd.DataFrame,
    trace_values: np.ndarray | None = None,
    recall_beta: float = 1.0,
    bout_peaks: np.ndarray | None = None,
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
    candidate = _plateau_midpoint(score)
    held_out_best = candidate
    capped = False
    if bout_peaks is not None and len(bout_peaks) >= MIN_CAP_BOUTS:
        cap = float(np.quantile(bout_peaks, 1.0 - PEAK_COVERAGE))
        cap = float(_GRID[max(0, int(np.searchsorted(_GRID, cap, side="right")) - 1)])
        if candidate > cap:
            candidate, capped = cap, True
    f1_cur = _f1_at(y, p, current)
    f1_new = _f1_at(y, p, candidate)
    # Judged at the held-out best: a low cap flags more windows by design, not because the model is weak.
    ratio = float(np.mean(p >= held_out_best)) / float(np.mean(y))
    if ratio > MAX_PREDICTED_OVER_TRUE:
        return result(
            None, f1_cur, f1_new, n_pos,
            f"Best threshold flags {ratio:.1f}x more windows than are positive; model too weak to tune.",
        )
    if trace_values is not None and len(trace_values):
        above = float(np.mean(trace_values >= candidate))
        if above > MAX_TRACE_FRACTION_ABOVE:
            return result(
                None, f1_cur, f1_new, n_pos,
                f"Best held-out threshold {candidate:.3f} is below the deployed model's baseline "
                f"({above:.0%} of session frames exceed it). Retrain so both use the same calibration.",
            )
    if abs(candidate - float(current)) < MIN_THRESHOLD_MOVE:
        return result(None, f1_cur, f1_new, n_pos, "Current threshold is already at the suggestion.")
    notes = []
    if capped:
        missed = float(np.mean(bout_peaks < float(current)))
        notes.append(
            f"Lowered from {held_out_best:.3f} so {PEAK_COVERAGE:.0%} of labeled bouts reach it "
            f"({missed:.0%} of them never reach the current threshold)."
        )
    elif candidate < current and f1_new >= f1_cur - PEAK_TOLERANCE:
        notes.append("About the same held-out F1 at a lower threshold, so fewer short bouts are missed.")
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
    labels = _load_labels(Path(project_root))
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
        out.append(suggest_for_predictions(
            bid, name, cur, preds, _latest_trace_values(project_root, bid), recall_beta,
            _labeled_bout_peaks(project_root, bid, labels),
        ))
    return out
