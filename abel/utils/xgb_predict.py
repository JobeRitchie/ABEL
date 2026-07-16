"""Prediction helpers for XGBoost models trained on the GPU.

ABEL fits boosters with ``device="cuda"`` when a GPU is present.  A booster keeps
that device setting after the fit, so predicting with it on a CPU array (every
`predict_proba` in this codebase passes a NumPy array) makes XGBoost copy the
whole feature matrix host→device for each call.  It warns about this once:

    Falling back to prediction using DMatrix due to mismatched devices...
    XGBoost is running on: cuda:0, while the input data is on: cpu.

Predicting on the GPU is not worth that copy.  Measured on an RTX 4070 with a
300-tree model over 400 features (the shape ABEL's dense inference actually
produces), against the CPU-numpy input we always have:

    rows      GPU booster (copy)   CPU booster
      5,000            23.5 ms        3.4 ms   (7.0x)
     50,000           207.5 ms       32.8 ms   (6.3x)
    300,000          1269.2 ms      155.6 ms   (8.2x)

The transfer dominates; the tree traversal is cheap either way.  Feeding the
DMatrix explicitly — the earlier workaround for this warning — is just the
fallback XGBoost was warning about and is no faster (1304 ms at 300k rows).

So prediction runs on the CPU.  Training still uses the GPU, where the cost is
real and the data is uploaded once.  Probabilities are unchanged (differences are
float32 rounding, ~4e-7).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

# Set on an estimator once its booster has been moved, so repeated prediction
# calls (dense inference runs one per behavior per session) don't re-set the param.
_CPU_MARK = "_abel_booster_on_cpu"


def _iter_xgb_estimators(model: Any) -> Iterator[Any]:
    """Yield every XGBoost sklearn estimator inside ``model``.

    Covers the wrappers ABEL actually builds: a bare ``XGBClassifier``, a
    ``CalibratedClassifierCV`` (fitted, or ``cv="prefit"`` before fitting), and a
    ``Pipeline`` ending in one.
    """
    if model is None:
        return
    if hasattr(model, "get_booster"):
        yield model
        return
    for cc in getattr(model, "calibrated_classifiers_", None) or ():
        est = getattr(cc, "estimator", None)
        if est is not None and hasattr(est, "get_booster"):
            yield est
    est = getattr(model, "estimator", None)          # CalibratedClassifierCV(cv="prefit")
    if est is not None and hasattr(est, "get_booster"):
        yield est
    steps = getattr(model, "steps", None)            # sklearn Pipeline
    if steps:
        final = steps[-1][1]
        if hasattr(final, "get_booster"):
            yield final


def ensure_cpu_prediction(model: Any) -> int:
    """Move every XGBoost booster in ``model`` onto the CPU for prediction.

    Idempotent and safe on non-XGBoost models (returns 0).  Only the *booster's*
    device is changed, not the estimator's ``device`` param, so a later refit of
    the same estimator still trains on the GPU.
    """
    moved = 0
    for est in _iter_xgb_estimators(model):
        if getattr(est, _CPU_MARK, False):
            continue
        try:
            est.get_booster().set_param({"device": "cpu"})
        except Exception:  # noqa: BLE001 — unfitted or not really XGBoost; leave it alone
            continue
        try:
            setattr(est, _CPU_MARK, True)
        except Exception:  # noqa: BLE001 — exotic estimator with __slots__
            pass
        moved += 1
    return moved


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    """``model.predict_proba(x)``, with any GPU booster first moved to the CPU."""
    ensure_cpu_prediction(model)
    return model.predict_proba(x)
