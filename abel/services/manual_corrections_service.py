"""Manual corrections to per-session Analytics values.

A user who has hand-scored a behavior can replace ABEL's whole-session value
(latency, bout count, total duration, mean bout duration) for one session and
one behavior.  The corrections live in ``config/manual_corrections.json`` and
every one keeps the ABEL value it replaced, so the edit is always auditable.

The file is user-entered data that cannot be rebuilt, which is why it sits in
``config`` and not in ``derived``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("abel")

CORRECTIONS_FILE = Path("config") / "manual_corrections.json"

#: Shared look of the project flag on the Home and Validation tabs.
MANUAL_FLAG_STYLE = (
    "color: #FFB74D; font-size: 12px; font-weight: 600; padding: 4px 8px;"
    " border: 1px solid #FFB74D; border-radius: 4px;"
)

#: Correctable metrics, in display order: summary-row key -> column header.
METRICS: dict[str, str] = {
    "n_bouts": "N Bouts",
    "time_spent_s": "Total Duration (s)",
    "mean_bout_s": "Mean Bout Duration (s)",
    "latency_s": "Latency to First (s)",
}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class ManualCorrections:
    """Corrections keyed by behavior id, then session id, then metric."""

    def __init__(self, data: dict[str, dict[str, dict[str, dict[str, Any]]]] | None = None,
                 behavior_names: dict[str, str] | None = None) -> None:
        self._data: dict[str, dict[str, dict[str, dict[str, Any]]]] = data or {}
        # Names as they were when the correction was saved, so the project
        # flag can still name a behavior that was deleted afterwards.
        self.behavior_names: dict[str, str] = dict(behavior_names or {})

    # -- persistence ---------------------------------------------------

    @classmethod
    def load(cls, project_root: Path | None) -> "ManualCorrections":
        if project_root is None:
            return cls()
        path = Path(project_root) / CORRECTIONS_FILE
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Could not read %s", path)
            return cls()
        data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for bid, sessions in (raw.get("corrections") or {}).items():
            if not isinstance(sessions, dict):
                continue
            for sid, metrics in sessions.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, entry in metrics.items():
                    if metric not in METRICS or not isinstance(entry, dict):
                        continue
                    value = _finite(entry.get("value"))
                    if value is None:
                        continue
                    data.setdefault(str(bid), {}).setdefault(str(sid), {})[metric] = {
                        "value": value,
                        "abel_value": _finite(entry.get("abel_value")),
                        "edited_at": str(entry.get("edited_at") or ""),
                    }
        names = raw.get("behavior_names") or {}
        return cls(data, {str(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {})

    def save(self, project_root: Path) -> None:
        """Write the corrections; remove the file when none are left."""
        path = Path(project_root) / CORRECTIONS_FILE
        if not self._data:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "behavior_names": {b: n for b, n in self.behavior_names.items() if b in self._data},
            "corrections": self._data,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def copy(self) -> "ManualCorrections":
        return ManualCorrections(json.loads(json.dumps(self._data)), self.behavior_names)

    # -- access --------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self._data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ManualCorrections) and self._data == other._data

    def get(self, behavior_id: str, session_id: str, metric: str) -> float | None:
        entry = self._data.get(str(behavior_id), {}).get(str(session_id), {}).get(metric)
        return None if entry is None else float(entry["value"])

    def abel_value(self, behavior_id: str, session_id: str, metric: str) -> float | None:
        entry = self._data.get(str(behavior_id), {}).get(str(session_id), {}).get(metric)
        return None if entry is None else entry.get("abel_value")

    def metrics_for(self, behavior_id: str, session_id: str) -> dict[str, float]:
        entries = self._data.get(str(behavior_id), {}).get(str(session_id), {})
        return {m: float(e["value"]) for m, e in entries.items()}

    def set(self, behavior_id: str, session_id: str, metric: str,
            value: float, abel_value: float | None) -> None:
        if metric not in METRICS:
            raise ValueError(f"Not a correctable metric: {metric}")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("A manual value must be a number of 0 or more.")
        self._data.setdefault(str(behavior_id), {}).setdefault(str(session_id), {})[metric] = {
            "value": value,
            "abel_value": _finite(abel_value),
            "edited_at": datetime.now().isoformat(timespec="seconds"),
        }

    def clear(self, behavior_id: str, session_id: str | None = None,
              metric: str | None = None) -> None:
        """Remove one value, one session's values, or a whole behavior's."""
        bid = str(behavior_id)
        if bid not in self._data:
            return
        if session_id is None:
            del self._data[bid]
            return
        sessions = self._data[bid]
        sid = str(session_id)
        if sid in sessions:
            if metric is None:
                del sessions[sid]
            else:
                sessions[sid].pop(metric, None)
                if not sessions[sid]:
                    del sessions[sid]
        if not sessions:
            del self._data[bid]

    def counts(self) -> dict[str, int]:
        """Number of corrected values per behavior id."""
        return {
            bid: sum(len(metrics) for metrics in sessions.values())
            for bid, sessions in self._data.items()
        }

    def latency_by_session(self, behavior_id: str) -> dict[str, float]:
        return {
            sid: float(metrics["latency_s"]["value"])
            for sid, metrics in self._data.get(str(behavior_id), {}).items()
            if "latency_s" in metrics
        }

    # -- applying ------------------------------------------------------

    def apply_to_rows(
        self,
        rows: list[dict[str, Any]],
        behavior_ids: Iterable[str],
        session_templates: dict[str, dict[str, Any]],
        behavior_names: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Return ``rows`` with corrected values swapped in.

        A corrected (session, behavior) pair that has no row yet (ABEL found
        no bouts there) gets one, built from ``session_templates``.  The input
        rows are never mutated.
        """
        wanted = {str(b) for b in behavior_ids} & set(self._data)
        if not wanted:
            return rows
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            bid = str(row.get("behavior_id", ""))
            sid = str(row.get("session_id", ""))
            seen.add((bid, sid))
            manual = self.metrics_for(bid, sid) if bid in wanted else {}
            out.append({**row, **manual} if manual else row)
        for bid in sorted(wanted):
            for sid in self._data[bid]:
                template = session_templates.get(sid)
                if (bid, sid) in seen or template is None:
                    continue
                out.append({
                    **template,
                    "behavior_id": bid,
                    "behavior": behavior_names.get(bid, self.behavior_names.get(bid, bid)),
                    "n_bouts": 0.0,
                    "time_spent_s": 0.0,
                    "mean_bout_s": 0.0,
                    "latency_s": float("nan"),
                    "distance_cm": 0.0,
                    **self.metrics_for(bid, sid),
                })
        return out


def _current_behavior_names(project_root: Path) -> dict[str, str]:
    path = Path(project_root) / "config" / "behavior_definitions.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # noqa: PLC0415

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            str(b.get("behavior_id")): str(b.get("name") or b.get("behavior_id"))
            for b in raw.get("behaviors", [])
            if isinstance(b, dict) and b.get("behavior_id")
        }
    except Exception:
        return {}


def correction_counts_by_name(project_root: Path | None) -> dict[str, int]:
    """Corrected-value count per behavior name; empty when there are none."""
    if project_root is None:
        return {}
    corrections = ManualCorrections.load(project_root)
    counts = corrections.counts()
    if not counts:
        return {}
    names = _current_behavior_names(Path(project_root))
    out: dict[str, int] = {}
    for bid, n in counts.items():
        name = names.get(bid) or corrections.behavior_names.get(bid) or bid
        out[name] = out.get(name, 0) + n
    return dict(sorted(out.items()))


def flag_text(project_root: Path | None) -> str:
    """One-line project flag, or ``""`` when the project has no corrections."""
    counts = correction_counts_by_name(project_root)
    if not counts:
        return ""
    parts = ", ".join(f"{name}: {n}" for name, n in counts.items())
    return (
        "Manual corrections in Analytics (values entered by hand in place of "
        f"ABEL's scoring): {parts}"
    )
