"""Held-out model performance for the Home tab.

The Home tab used to read ``derived/evaluation/model_metrics.json``.  That
file is rewritten by every training run, so it holds one behavior only (the
last one trained), and it scores the deployed model on the same reviewed
segments it was trained on.  The result read as "Model Performance" but was a
training fit: 1.000 for a project whose held-out F1s were 0.80 to 0.93.

Each model directory keeps its own ``metrics.json`` from the honest train /
validation split.  This module summarizes those across the project's current
behavior models.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(re.sub(r"\bNaN\b", "null", text))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def target_class_metrics(metrics: dict[str, Any]) -> dict[str, float] | None:
    """Target-class precision / recall / F1 from a saved 2x2 confusion matrix.

    ``metrics.json`` stores macro P/R/F1 (target averaged with not-target).  A
    one-vs-rest holdout is mostly not-target, so the macro value sits well
    above the target's own score.  The confusion matrix (rows = true class)
    plus ``n_val_pos`` identify the target row; None when they cannot.
    """
    cm = metrics.get("confusion_matrix")
    n_pos = _finite(metrics.get("n_val_pos"))
    try:
        rows = [[float(v) for v in row] for row in cm]
    except (TypeError, ValueError):
        return None
    if len(rows) != 2 or any(len(r) != 2 for r in rows) or n_pos is None:
        return None
    sums = [sum(r) for r in rows]
    if sums[0] == sums[1]:
        return None  # balanced holdout: the target row cannot be told apart
    matches = [i for i, s in enumerate(sums) if abs(s - n_pos) < 0.5]
    if len(matches) != 1:
        return None
    t = matches[0]
    tp = rows[t][t]
    fn = rows[t][1 - t]
    fp = rows[1 - t][t]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def held_out_summary(
    project_root: Path, behavior_names: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Median held-out metrics across the newest model of each behavior.

    Returns None when no model has held-out metrics.  Keys:

    ``f1`` / ``precision`` / ``recall`` / ``pr_auc``
        Medians across behaviors (None when unavailable).
    ``basis``
        ``"target"`` when every model's target-class scores could be derived,
        otherwise ``"macro"`` (the values ``metrics.json`` stores).
    ``n_models``
        Number of behavior models summarized.
    ``lowest_f1`` / ``lowest_behavior``
        The weakest model, which a median hides.
    """
    from abel.services.workflow_snapshot_service import WorkflowSnapshotService  # noqa: PLC0415

    names = behavior_names or {}
    try:
        models = WorkflowSnapshotService()._auto_resolve_behavior_models(Path(project_root))
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    for bid, version in sorted(models.items()):
        metrics = _read_json(Path(project_root) / "derived" / "models" / version / "metrics.json")
        if not metrics or _finite(metrics.get("f1")) is None:
            continue
        # A model trained without a holdout has nothing honest to report.
        if metrics.get("holdout") is False or metrics.get("evaluated_on") not in (None, "held_out"):
            continue
        rows.append({
            "behavior": names.get(bid, bid),
            "macro": {
                "f1": _finite(metrics.get("f1")),
                "precision": _finite(metrics.get("precision")),
                "recall": _finite(metrics.get("recall")),
            },
            "target": target_class_metrics(metrics),
            "pr_auc": _finite(metrics.get("pr_auc")),
        })
    if not rows:
        return None

    basis = "target" if all(r["target"] is not None for r in rows) else "macro"
    f1s = [r[basis]["f1"] for r in rows]
    weakest = min(
        (r for r in rows if r[basis]["f1"] is not None),
        key=lambda r: r[basis]["f1"], default=None,
    )
    return {
        "f1": _median(f1s),
        "precision": _median([r[basis]["precision"] for r in rows]),
        "recall": _median([r[basis]["recall"] for r in rows]),
        "pr_auc": _median([r["pr_auc"] for r in rows]),
        "basis": basis,
        "n_models": len(rows),
        "lowest_f1": weakest[basis]["f1"] if weakest else None,
        "lowest_behavior": weakest["behavior"] if weakest else "",
    }
