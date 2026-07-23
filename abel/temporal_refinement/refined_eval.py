"""Post-refinement evaluation of trained behavior models.

The per-segment metrics stored in ``behavior_model_*/metrics.json`` grade the
model on its *rawest* output: a blanket ``prob >= 0.5`` cut with no temporal
refinement. But the product's final bouts are produced by the Temporal Review
stage, which turns per-window probabilities into clean bouts via
``smooth -> threshold -> merge close bouts -> drop short bouts`` using
per-behavior settings tuned in ``config/temporal_review_settings.json``. Those
steps remove isolated false positives and fill dropout gaps, materially changing
TP/FP/FN — so the honest, publication-grade quality number is computed *after*
refinement.

This module is the single source of truth for that computation, shared by:
  * ``abel/benchmark/runner.py``               (benchmark CV evaluation)
  * ``abel/services/validation_service.py``    (Validation tab overview table)
  * ``abel/validation/datamodel.py``           (external validation suite)

It reuses the exact bout-postprocess primitives the real refinement pipeline
uses, so the metrics can never drift from what ships.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abel.storage.file_store import read_json, read_yaml
from abel.temporal_refinement.bout_postprocess import (
    merge_close_bouts,
    remove_short_bouts,
    smooth_probabilities,
    threshold_probabilities,
)

# Baseline used when a project has no temporal_review_settings.json yet. Mirrors
# the product defaults (Temporal Review tab / benchmark runner): the pipeline
# thresholds with a single gate defined by onset threshold, min-bout, and merge-gap.
DEFAULT_TEMPORAL_SETTINGS: dict[str, Any] = {
    "onset_threshold": 0.65,
    "min_bout_duration_frames": 8,
    "merge_gap_frames": 4,
}

# A predicted bout counts as a true positive when its temporal Intersection-over-
# Union with a ground-truth bout reaches this. Deliberately lenient: the review
# tab treats a detection that clearly lands on a real bout as correct even when
# its boundaries are a little loose, so window-level boundary slop doesn't turn a
# good detection into both an FP and an FN.
BOUT_MATCH_IOU = 0.2

# Refined metrics are suppressed once more than this fraction of held-out positives
# sit in observed islands too short to hold a min_bout-length prediction — beyond it
# the score measures label sparsity, not the model.  See refinement_evaluability.
_REFINE_UNSUPPORTED_MAX = 0.10

# Trailing "_<start>_<end>" in a segment id, e.g. seg_MS2_session_a7623464_6_20.
_SEG_FRAME_RE = re.compile(r"_(\d+)_(\d+)$")


def load_temporal_settings(project_root: Path, target_behavior: str) -> dict[str, Any]:
    """Load per-behavior temporal refinement settings from the project.

    Returns a dict with ``onset_threshold``, ``min_bout_duration_frames`` and
    ``merge_gap_frames``. Resolves ``by_behavior`` entries (which may be keyed by
    UUID) against ``behavior_definitions.yaml`` so callers can pass either the
    behavior name or its id. Falls back to the ``__all__`` baseline, then to
    :data:`DEFAULT_TEMPORAL_SETTINGS`.
    """
    settings = dict(DEFAULT_TEMPORAL_SETTINGS)

    review_path = Path(project_root) / "config" / "temporal_review_settings.json"
    if not review_path.exists():
        return settings
    try:
        raw = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return settings

    settings.update(raw.get("__all__", {}) or {})

    by_behavior = raw.get("by_behavior", {}) or {}
    if not by_behavior:
        return settings

    uuid_to_name: dict[str, str] = {}
    defs_path = Path(project_root) / "config" / "behavior_definitions.yaml"
    if defs_path.exists():
        try:
            import yaml  # noqa: PLC0415

            defs = yaml.safe_load(defs_path.read_text(encoding="utf-8")) or {}
            for b in defs.get("behaviors", []):
                uuid_to_name[str(b.get("behavior_id"))] = str(b.get("name", ""))
        except Exception:
            pass

    for uid, cfg in by_behavior.items():
        name = uuid_to_name.get(str(uid), str(uid))
        if name == target_behavior or str(uid) == str(target_behavior):
            settings.update(cfg or {})
            break

    return settings


def _refine_binary_trace(
    sf: np.ndarray,
    ef: np.ndarray,
    clip_probs: np.ndarray,
    onset_threshold: float,
    min_bout_duration_frames: int,
    merge_gap_frames: int,
    smooth_window: int,
) -> tuple[np.ndarray, int] | None:
    """Build one session's refined binary bout trace from sorted segments.

    ``sf``/``ef``/``clip_probs`` must already be sorted by start frame. Assigns
    each segment's target probability to its ``[start, end]`` frame range,
    interpolates inter-segment gaps, then runs the real bout pipeline
    (smooth -> threshold -> merge close -> drop short). Returns the frame-level
    binary trace and its absolute start frame, or ``None`` when the session has
    no usable frame span. The single source of the refinement math for both the
    segment-level remap and the bout-level event matching.
    """
    trace_start = int(sf[0])
    trace_end = int(ef[-1])
    n_frames = trace_end - trace_start + 1
    if n_frames <= 0:
        return None

    frame_trace = np.full(n_frames, np.nan, dtype=np.float32)
    for i in range(len(sf)):
        local_s = int(sf[i]) - trace_start
        local_e = min(int(ef[i]) - trace_start, n_frames - 1)
        frame_trace[local_s : local_e + 1] = clip_probs[i]

    nans = np.isnan(frame_trace)
    if nans.any() and not nans.all():
        known = np.where(~nans)[0]
        frame_trace = np.interp(np.arange(n_frames), known, frame_trace[known]).astype(np.float32)
    elif nans.all():
        return None

    frame_trace = smooth_probabilities(frame_trace, method="moving_average", window=smooth_window)
    binary = threshold_probabilities(frame_trace, onset_threshold)
    binary = merge_close_bouts(binary, merge_gap_frames)
    binary = remove_short_bouts(binary, min_bout_duration_frames)
    return np.asarray(binary), trace_start


def observed_islands(
    sf: np.ndarray, ef: np.ndarray, merge_gap_frames: int
) -> list[tuple[int, int]]:
    """Split start-sorted segments into runs of genuinely observed frames.

    Returns inclusive ``[first_index, last_index]`` slices of the *segment* arrays,
    cut wherever the unobserved gap to the next segment exceeds ``merge_gap_frames``.

    Why this exists: :func:`_refine_binary_trace` needs a probability for every
    frame and interpolates whatever lies between segments.  At **inference** that
    is right and costs nothing — windows tile the video densely (measured on a
    real session: 5513 segments over 22,063 frames, ~375% coverage, *zero* gaps,
    11-frame overlap), so interpolation only fills tiling seams and this function
    returns a single island, leaving behavior unchanged.

    On a **held-out labeled subset** it is wrong: only human-labeled windows are
    present, so coverage collapses (measured: ~34%, 252 gaps, median 36 / max 517
    frames) and interpolation invents two thirds of the trace.  Those invented
    stretches cross the onset threshold and manufacture bouts the model never
    predicted.  Cutting at gaps wider than the merge gap confines refinement to
    frames that were actually scored; seams up to ``merge_gap_frames`` are still
    bridged, which is exactly what that setting is for.
    """
    sf = np.asarray(sf, dtype=int)
    ef = np.asarray(ef, dtype=int)
    n = len(sf)
    if n == 0:
        return []
    gaps = sf[1:] - ef[:-1] - 1
    cuts = np.where(gaps > int(merge_gap_frames))[0]
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [n - 1]))
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def apply_temporal_refinement(
    probs: np.ndarray,
    target_col: int,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    onset_threshold: float,
    min_bout_duration_frames: int,
    merge_gap_frames: int,
    smooth_window: int = 5,
) -> np.ndarray:
    """Apply the real bout-extraction pipeline and map results back to segments.

    For each session, and within it each run of contiguously observed frames
    (see :func:`observed_islands` — one island covering everything at inference):
    1. Build a frame-level probability trace by assigning each segment's
       target-class probability to its ``[start_frame, end_frame]`` range, then
       linearly interpolating gaps between segments.
    2. Apply smoothing -> thresholding (single gate) -> merge close bouts ->
       remove short bouts.
    3. Map the binary bout trace back to segment predictions: a segment is
       predicted positive when the majority of its frames fall inside a bout.

    Returns a new argmax-style prediction vector with refinement applied.
    """
    preds = np.argmax(probs, axis=1).copy()

    for sid in np.unique(session_ids):
        sess_idxs = np.where(session_ids == sid)[0]
        if len(sess_idxs) < 2:
            continue

        sf_all = start_frames[sess_idxs].astype(int)
        ef_all = end_frames[sess_idxs].astype(int)
        order = np.argsort(sf_all)
        sess_idxs = sess_idxs[order]
        sf_all = sf_all[order]
        ef_all = ef_all[order]

        for a, b in observed_islands(sf_all, ef_all, merge_gap_frames):
            idxs = sess_idxs[a : b + 1]
            sf = sf_all[a : b + 1]
            ef = ef_all[a : b + 1]

            clip_probs = probs[idxs, target_col].astype(float)
            refined = _refine_binary_trace(
                sf, ef, clip_probs,
                onset_threshold, min_bout_duration_frames, merge_gap_frames,
                smooth_window,
            )
            if refined is None:
                continue
            binary, trace_start = refined
            n_frames = len(binary)

            for i in range(len(idxs)):
                local_s = int(sf[i]) - trace_start
                local_e = min(int(ef[i]) - trace_start, n_frames - 1)
                clip_span = binary[local_s : local_e + 1]
                if len(clip_span) > 0 and clip_span.mean() >= 0.5:
                    preds[idxs[i]] = target_col
                else:
                    p = probs[idxs[i]].copy()
                    p[target_col] = -1.0
                    preds[idxs[i]] = int(np.argmax(p))

    return preds


def _extract_bouts(binary: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end]`` index runs where ``binary`` is truthy."""
    bouts: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(np.asarray(binary)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bouts.append((start, i - 1))
            start = None
    if start is not None:
        bouts.append((start, len(binary) - 1))
    return bouts


def _bout_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Temporal Intersection-over-Union of two inclusive ``[start, end]`` spans."""
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def _match_bouts(
    pred_bouts: list[tuple[int, int]],
    true_bouts: list[tuple[int, int]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedily match predicted bouts to ground-truth bouts by temporal IoU.

    Each true bout is claimed by at most one predicted bout (its best overlap
    at or above ``iou_threshold``). Returns event-level ``(tp, fp, fn)``: a
    matched prediction is a TP, an unmatched prediction an FP, and an unclaimed
    true bout an FN.
    """
    matched_true: set[int] = set()
    tp = 0
    for pb in pred_bouts:
        best_j, best_iou = -1, iou_threshold
        for j, tb in enumerate(true_bouts):
            if j in matched_true:
                continue
            iou = _bout_iou(pb, tb)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_true.add(best_j)
            tp += 1
    fp = len(pred_bouts) - tp
    fn = len(true_bouts) - len(matched_true)
    return tp, fp, fn


def _refined_bout_counts(
    prob: np.ndarray,
    y_true: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    settings: dict[str, Any],
    smooth_window: int = 5,
) -> tuple[int, int, int]:
    """Event-level TP/FP/FN by matching refined bouts against ground-truth bouts.

    .. warning::
       **Not a valid held-out metric — do not report it.**  It is retained only
       so :mod:`abel.temporal_refinement.auto_settings` can be checked against it,
       and it keeps the original dense-trace behavior deliberately.

       Bouts are not identifiable from a held-out *labeled* subset.  The evaluated
       unit is an isolated ~15-frame window, while a bout needs contiguous
       observation longer than itself, so this systematically reports extreme
       FP and FN even for well-trained models.  Two independent artifacts:
       interpolation across unobserved gaps invents predicted bouts (measured:
       7 of 13 predicted bouts were >50% interpolated frames), and sparse labels
       fragment one real bout into several true bouts while islands shorter than
       ``min_bout_duration_frames`` can hold no surviving prediction (87% of
       islands for a min_bout=30 behavior).  Scoring events properly requires
       dense inference over the full session, plus knowing which spans were
       exhaustively labeled.

    For each session the refined binary trace supplies the predicted bouts; the
    ground-truth ``y_true`` labels, laid onto the same frame axis (with the
    project's merge-gap closing tiling seams), supply the true bouts. Predicted
    and true bouts are matched by temporal IoU (:data:`BOUT_MATCH_IOU`). Counts
    are summed across sessions.
    """
    onset = float(settings["onset_threshold"])
    min_bout = int(settings["min_bout_duration_frames"])
    merge_gap = int(settings["merge_gap_frames"])

    tp = fp = fn = 0
    for sid in np.unique(session_ids):
        idxs = np.where(session_ids == sid)[0]
        if len(idxs) < 2:
            continue
        sf = start_frames[idxs].astype(int)
        ef = end_frames[idxs].astype(int)
        order = np.argsort(sf)
        sf, ef = sf[order], ef[order]
        clip_probs = prob[idxs][order].astype(float)
        yt = y_true[idxs][order].astype(int)

        refined = _refine_binary_trace(
            sf, ef, clip_probs, onset, min_bout, merge_gap, smooth_window,
        )
        if refined is None:
            continue
        pred_binary, trace_start = refined
        n_frames = len(pred_binary)

        # Ground-truth bouts on the same frame axis. Merge-gap closes the 1-frame
        # seams between contiguous positive windows so a single labeled bout does
        # not fragment; no min-bout is applied — real short bouts must still count.
        true_trace = np.zeros(n_frames, dtype=np.uint8)
        for i in range(len(sf)):
            local_s = max(0, int(sf[i]) - trace_start)
            local_e = min(int(ef[i]) - trace_start, n_frames - 1)
            if yt[i] == 1 and local_e >= local_s:
                true_trace[local_s : local_e + 1] = 1
        true_binary = merge_close_bouts(true_trace, merge_gap)

        s_tp, s_fp, s_fn = _match_bouts(
            _extract_bouts(pred_binary), _extract_bouts(true_binary), BOUT_MATCH_IOU
        )
        tp += s_tp
        fp += s_fp
        fn += s_fn
    return tp, fp, fn


def _frames_from_segment_ids(seg_ids: "pd.Series | np.ndarray") -> tuple[np.ndarray, np.ndarray]:
    """Parse trailing ``_<start>_<end>`` frame bounds out of segment ids.

    Segment ids look like ``seg_MS2_session_a7623464_6_20``. Ids that don't match
    yield ``-1`` so callers can drop them.
    """
    starts: list[int] = []
    ends: list[int] = []
    for s in seg_ids:
        m = _SEG_FRAME_RE.search(str(s))
        if m:
            starts.append(int(m.group(1)))
            ends.append(int(m.group(2)))
        else:
            starts.append(-1)
            ends.append(-1)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _macro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Macro precision/recall/F1 — matches the trainer's metrics.json convention."""
    from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: PLC0415

    return (
        float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def _target_encoded_index(model_dir: Path, target_behavior_id: str) -> int | None:
    """Find which encoded label index is the target (positive) class.

    Behavior models are one-vs-rest but the target is NOT always encoded as 1 —
    e.g. an Approach model has ``label_map = {0: <approach id>, 1: no_behavior}``,
    so the positive class is 0. Reading the encoding is essential; assuming
    ``label_true == 1`` is positive silently inverts every metric. Prefers the
    cheap ``model_card.yaml`` labels list (ordered by encoded index); falls back
    to unpickling the model's ``label_map`` only if needed.
    """
    tid = str(target_behavior_id or "").strip()
    if not tid:
        return None
    card = read_yaml(model_dir / "model_card.yaml", {})
    labels = card.get("labels")
    if isinstance(labels, list) and tid in [str(x) for x in labels]:
        return [str(x) for x in labels].index(tid)
    try:
        import pickle  # noqa: PLC0415

        with open(model_dir / "model_state.pkl", "rb") as f:
            label_map = pickle.load(f).get("label_map", {})
        for idx, lab in label_map.items():
            if str(lab) == tid:
                return int(idx)
    except Exception:
        pass
    return None


def refined_holdout_metrics(
    model_dir: Path,
    project_root: Path,
    behavior_name: str,
    *,
    target_behavior_id: str | None = None,
    seg_meta: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Compute raw-vs-refined macro metrics on a model's held-out validation set.

    Reads the honest held-out target probability stored in the model's own
    ``validation_predictions.parquet`` (``prediction_prob``, written at train time
    from the train/val split — NOT the leaky deploy-model ``segment_predictions``)
    and scores the segments twice from the SAME probabilities: once raw
    (``P(target) >= 0.5``) and once after temporal refinement using the project's
    per-behavior settings. Both use macro averaging so the two are directly
    comparable and the delta isolates the effect of refinement.

    The positive class is taken from the stored ``target_index`` when present,
    else resolved from the model's label encoding (see
    :func:`_target_encoded_index`) — never assumed, since behavior models
    frequently encode the target as class 0.

    ``seg_meta`` (segment_id -> start_frame/end_frame) is optional; when omitted,
    frame bounds are parsed from the segment ids. Returns ``None`` when the
    required artifacts are missing or the held-out probability column is absent
    (model predates the change and must be retrained).
    """
    model_dir = Path(model_dir)
    vp_path = model_dir / "validation_predictions.parquet"
    if not vp_path.exists():
        return None

    try:
        df = pd.read_parquet(vp_path)
    except Exception:
        return None
    if df.empty or "label_true" not in df.columns or "session_id" not in df.columns:
        return None
    # The honest held-out probability must be present. Models trained before this
    # column existed cannot be graded leak-free after the fact (the held-out model
    # is not persisted), so we return None and the UI shows "—" until retrained.
    if "prediction_prob" not in df.columns:
        return None

    # Positive-class encoding: prefer the value stamped at train time; otherwise
    # resolve from the model's label map. Never assume positive == 1 (behavior
    # models often encode the target as class 0).
    target_index: int | None = None
    if "target_index" in df.columns and df["target_index"].notna().any():
        try:
            target_index = int(df["target_index"].dropna().iloc[0])
        except (TypeError, ValueError):
            target_index = None
    if target_index is None:
        if target_behavior_id is None:
            rs = read_json(model_dir / "run_settings.json", {})
            target_behavior_id = str(rs.get("target_behavior") or rs.get("target_behavior_id") or "").strip()
        target_index = _target_encoded_index(model_dir, target_behavior_id)
    if target_index is None:
        return None

    if seg_meta is not None and {"start_frame", "end_frame"}.issubset(seg_meta.columns):
        df = df.merge(seg_meta[["segment_id", "start_frame", "end_frame"]], on="segment_id", how="left")
    else:
        s, e = _frames_from_segment_ids(df["segment_id"])
        df["start_frame"] = s
        df["end_frame"] = e

    df = df[np.isfinite(df["prediction_prob"].to_numpy(dtype=float))]
    df = df[df["start_frame"].to_numpy(dtype=float) >= 0]
    if df.empty:
        return None

    p = df["prediction_prob"].to_numpy(dtype=float)
    probs = np.column_stack([1.0 - p, p])
    settings = load_temporal_settings(project_root, behavior_name)
    y_true = (df["label_true"].to_numpy(dtype=int) == int(target_index)).astype(int)

    out = score_raw_and_refined(
        y_true=y_true,
        prob=p,
        session_ids=df["session_id"].astype(str).to_numpy(),
        start_frames=df["start_frame"].to_numpy(dtype=np.int64),
        end_frames=df["end_frame"].to_numpy(dtype=np.int64),
        settings=settings,
    )
    out["n_val"] = int(len(df))
    return out


def refinement_evaluability(
    y_true: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    settings: dict[str, Any],
) -> tuple[bool, float]:
    """Can refinement be scored on this held-out set at all?

    Returns ``(evaluable, unsupported_positive_fraction)``.

    Refinement drops bouts shorter than ``min_bout_duration_frames``.  At
    inference that removes flicker.  On a held-out *labeled* subset the observed
    stretches are isolated windows (measured median island: 15 frames), so when
    ``min_bout`` exceeds the island length **no prediction can survive there no
    matter how good the model is** — every positive in that island is forced to a
    false negative.  Measured on a real project: an 87% unsupported fraction for a
    ``min_bout=30`` behavior, which dragged its refined recall to 0.60 while its
    raw recall was 0.90.

    Reporting a refined score in that regime is as misleading as the interpolated
    bout counts this replaced, just pessimistic instead of optimistic, so callers
    should surface "not evaluable" rather than a number.
    """
    y_true = np.asarray(y_true, dtype=int)
    sess = np.asarray(session_ids)
    sf_all = np.asarray(start_frames, dtype=np.int64)
    ef_all = np.asarray(end_frames, dtype=np.int64)
    min_bout = int(settings["min_bout_duration_frames"])
    merge_gap = int(settings["merge_gap_frames"])

    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or min_bout <= 1:
        return True, 0.0

    unsupported = 0
    for sid in np.unique(sess):
        idxs = np.where(sess == sid)[0]
        if len(idxs) == 0:
            continue
        order = np.argsort(sf_all[idxs])
        idxs = idxs[order]
        sf, ef = sf_all[idxs], ef_all[idxs]
        for a, b in observed_islands(sf, ef, merge_gap):
            span = int(ef[a : b + 1].max()) - int(sf[a]) + 1
            if span < min_bout:
                unsupported += int((y_true[idxs[a : b + 1]] == 1).sum())

    frac = unsupported / float(n_pos)
    return frac <= _REFINE_UNSUPPORTED_MAX, float(frac)


def score_raw_and_refined(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    session_ids: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Score already-oriented held-out predictions raw vs refined.

    All inputs are aligned 1-D arrays with the positive class as ``1`` and
    ``prob`` = P(target). ``settings`` supplies the temporal-refinement knobs.
    Shared by the single-split path and leave-one-subject-out CV so the two
    always agree on the math. Returns macro P/R/F1 and positive-class
    TP/FP/FN/TN for both raw (``prob >= 0.5``) and refined predictions.

    Refined metrics come back as NaN (with ``refined_evaluable`` False) when the
    held-out windows are too sparse to support the project's ``min_bout``; see
    :func:`refinement_evaluability`.
    """
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    probs = np.column_stack([1.0 - p, p])

    refined = apply_temporal_refinement(
        probs,
        target_col=1,
        session_ids=np.asarray(session_ids),
        start_frames=np.asarray(start_frames, dtype=np.int64),
        end_frames=np.asarray(end_frames, dtype=np.int64),
        onset_threshold=float(settings["onset_threshold"]),
        min_bout_duration_frames=int(settings["min_bout_duration_frames"]),
        merge_gap_frames=int(settings["merge_gap_frames"]),
    )

    raw_pred = (p >= 0.5).astype(int)
    ref_pred = (refined == 1).astype(int)

    raw_p, raw_r, raw_f = _macro_prf(y_true, raw_pred)
    ref_p, ref_r, ref_f = _macro_prf(y_true, ref_pred)

    # Suppress refined scores the held-out sampling cannot support, rather than
    # publishing a number that reflects label sparsity instead of the model.
    evaluable, unsupported_frac = refinement_evaluability(
        y_true, session_ids, start_frames, end_frames, settings
    )
    if not evaluable:
        ref_p = ref_r = ref_f = float("nan")

    def _counts(pred: np.ndarray) -> tuple[int, int, int, int]:
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        tn = int(((y_true == 0) & (pred == 0)).sum())
        return tp, fp, fn, tn

    raw_tp, raw_fp, raw_fn, raw_tn = _counts(raw_pred)
    ref_tp, ref_fp, ref_fn, ref_tn = _counts(ref_pred)

    # NOTE: event-level ("bout") TP/FP/FN used to be reported here and has been
    # removed — it was not identifiable from a held-out labeled subset.  Bouts
    # need contiguous observation, but the evaluated unit is an isolated ~15-frame
    # window; scoring them inflated FP (interpolation across unobserved gaps
    # manufactured bouts) and FN (sparse labels fragment one real bout into
    # several, and an island shorter than min_bout can hold no prediction at all —
    # measured 87% of islands for a min_bout=30 behavior).  The counts below are
    # window-level against the reviewer's own accepted labels, which is what the
    # ground truth actually supports.  See `observed_islands`.

    return {
        "raw_precision": raw_p,
        "raw_recall": raw_r,
        "raw_f1": raw_f,
        "refined_precision": ref_p,
        "refined_recall": ref_r,
        "refined_f1": ref_f,
        "raw_tp": raw_tp,
        "raw_fp": raw_fp,
        "raw_fn": raw_fn,
        "raw_tn": raw_tn,
        "refined_tp": ref_tp,
        "refined_fp": ref_fp,
        "refined_fn": ref_fn,
        "refined_tn": ref_tn,
        # False when the held-out windows are too sparse to support min_bout, in
        # which case the refined_* scores above are NaN by design.
        "refined_evaluable": bool(evaluable),
        "refined_unsupported_fraction": float(unsupported_frac),
        "n_val": int(len(y_true)),
        "raw_positive_pred": int(raw_pred.sum()),
        "refined_positive_pred": int(ref_pred.sum()),
        "settings": {
            "onset_threshold": float(settings["onset_threshold"]),
            "min_bout_duration_frames": int(settings["min_bout_duration_frames"]),
            "merge_gap_frames": int(settings["merge_gap_frames"]),
        },
    }
