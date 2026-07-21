"""Probability-calibration validation of the production model.

ABEL can calibrate its classifier probabilities (sigmoid / isotonic), and the
ablation shows what that does to F1 — but F1 barely moves under calibration
because it only depends on the argmax, not on whether ``P = 0.8`` really means
"right 80% of the time".  A publication claim that the scores are *usable as
probabilities* (for thresholding, uncertainty triage, or downstream modelling)
needs a calibration figure: a reliability diagram plus the expected calibration
error and Brier score.

This consumes the retained held-out predictions from generalization (the
production, project-configured model — already calibrated if the project turns
calibration on), so it costs no extra training.  See
:func:`abel.validation.metrics.calibration_curve`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from abel.validation import metrics as vmetrics
from abel.validation.analyses.generalization import HoldoutPredictions
from abel.validation.metrics import CalibrationCurve


@dataclass
class CalibrationResult:
    project_id: str
    behavior_id: str
    behavior_name: str
    curve: CalibrationCurve
    n_bins: int = 10

    @property
    def ece(self) -> float:
        return self.curve.ece

    @property
    def brier(self) -> float:
        return self.curve.brier


def run_calibration(
    preds: HoldoutPredictions | None, *, n_bins: int = 10,
) -> CalibrationResult | None:
    """Reliability curve + ECE/MCE/Brier for one (project, behavior).

    Returns ``None`` when no predictions were retained or nothing is scorable.
    """
    if preds is None:
        return None
    curve = vmetrics.calibration_curve(preds.y_true, preds.prob, n_bins=n_bins)
    if curve.n == 0:
        return None
    return CalibrationResult(
        project_id=preds.project_id,
        behavior_id=preds.behavior_id,
        behavior_name=preds.behavior_name,
        curve=curve,
        n_bins=int(n_bins),
    )


def calibration_rows(results: list[CalibrationResult]) -> pd.DataFrame:
    """Tidy per-(project, behavior) calibration summary for CSV / the report."""
    rows = []
    for r in results:
        if r is None:
            continue
        rows.append(
            {
                "project": r.project_id,
                "behavior": r.behavior_name,
                "n_val": r.curve.n,
                "ece": r.curve.ece,
                "mce": r.curve.mce,
                "brier": r.curve.brier,
            }
        )
    return pd.DataFrame(rows)


def reliability_points(results: list[CalibrationResult]) -> pd.DataFrame:
    """Per-bin reliability points (mean confidence, empirical accuracy, count)."""
    rows = []
    for r in results:
        if r is None:
            continue
        c = r.curve
        for conf, acc, cnt in zip(c.bin_confidence, c.bin_accuracy, c.bin_count):
            rows.append(
                {
                    "project": r.project_id,
                    "behavior": r.behavior_name,
                    "mean_confidence": conf,
                    "empirical_accuracy": acc,
                    "count": cnt,
                }
            )
    return pd.DataFrame(rows)
