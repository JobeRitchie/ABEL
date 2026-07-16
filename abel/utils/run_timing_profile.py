"""Per-project, cross-run-type timing profile for calibrated ETAs.

Records the wall-clock cost of each broad pipeline *phase* (Preparing, Training,
Scoring, Evaluating, Benchmarking, …) observed in ANY run — single or batch,
retrain / pipeline / run-model — so a later run of any kind can seed its ETA
from real prior measurements instead of starting blind. Phases are the shared
vocabulary across run types, so time learned while *scoring* in a retrain run
informs the scoring estimate of a pipeline run and vice-versa.

**Whole-run totals.** Summing the independent per-phase means does NOT reconstruct
a real run's wall time: each phase mean is measured from a different number of
blocks across a history of runs, so a phase that occurs (say) twice per behavior
is counted once, and a rarely-seen phase drags its mean around. The reconstructed
"one behavior = sum of phase means" therefore drifts badly from reality (measured
on real data: phase-sum ≈ 103 s vs an actual 84 s run). To represent WHOLE-RUN
time accurately we also record the true end-to-end wall time of each completed
run, normalised to seconds-per-behavior and keyed by *run kind* (whole-run totals
are NOT comparable across kinds — a scoring-only run-model pass is far cheaper per
behavior than a full pipeline). This measured total is the primary ETA anchor;
the per-phase means remain for the live within-run shape and as a fallback.

The stored means are weight-capped so they stay adaptive: recent runs keep nudging
them instead of being drowned out by a long history. Pure / Qt-free so it can be
unit-tested headlessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Cap the sample weight so the running mean keeps adapting to recent machine /
# dataset conditions instead of freezing after many runs.
_WEIGHT_CAP = 24.0
_MAX_SANE_SECONDS = 24 * 3600.0
# Phases that are not part of a per-item train/score loop (e.g. a manual, whole-
# run UMAP) and so must not inflate the one-item total used for ETA seeding.
_NON_CORE_PHASES = frozenset({"Embedding"})


class RunTimingProfile:
    """Adaptive per-phase wall-clock means, persisted per project."""

    FILENAME = "run_timing_profile.json"

    def __init__(
        self,
        phases: dict[str, dict[str, float]] | None = None,
        run_totals: dict[str, dict[str, float]] | None = None,
    ) -> None:
        # phase -> {"seconds": running mean, "n": sample count}
        self._phases: dict[str, dict[str, float]] = {}
        for k, v in (phases or {}).items():
            try:
                self._phases[str(k)] = {
                    "seconds": float(v.get("seconds", 0.0)),
                    "n": float(v.get("n", 0.0)),
                }
            except Exception:
                continue
        # run kind -> {"seconds": adaptive mean of end-to-end seconds *per
        # behavior*, "n": sample count}. Kept separate from phases because a
        # whole-run total is only comparable within the same run kind.
        self._run_totals: dict[str, dict[str, float]] = {}
        for k, v in (run_totals or {}).items():
            try:
                self._run_totals[str(k)] = {
                    "seconds": float(v.get("seconds", 0.0)),
                    "n": float(v.get("n", 0.0)),
                }
            except Exception:
                continue

    # ------------------------------------------------------------------
    def record(self, phase: str, seconds: float) -> None:
        """Fold one observed phase duration into its adaptive running mean."""
        if not phase or seconds is None:
            return
        s = float(seconds)
        if s <= 0.0 or s > _MAX_SANE_SECONDS:
            return
        e = self._phases.setdefault(phase, {"seconds": 0.0, "n": 0.0})
        w = min(e["n"], _WEIGHT_CAP)
        e["seconds"] = (e["seconds"] * w + s) / (w + 1.0)
        e["n"] = e["n"] + 1.0

    def phase_seconds(self, phase: str, fallback: float | None = None) -> float:
        e = self._phases.get(phase)
        if e and e["n"] > 0:
            return e["seconds"]
        return fallback if fallback is not None else self.overall_mean()

    def overall_mean(self) -> float:
        """Sample-weighted mean phase duration across all known phases."""
        num = den = 0.0
        for e in self._phases.values():
            num += e["seconds"] * e["n"]
            den += e["n"]
        return (num / den) if den > 0 else 0.0

    def behavior_total(self) -> float:
        """Estimated wall time of one full item = sum of core phase means."""
        return float(
            sum(
                e["seconds"]
                for k, e in self._phases.items()
                if e["n"] > 0 and k not in _NON_CORE_PHASES
            )
        )

    def per_step_seed(self, steps: int, kind: str | None = None) -> float:
        """A per-step seed scaled to a run reporting ``steps`` steps per item.

        The profile is phase-granular (~a handful of phases); an estimator may be
        step-granular (more, finer steps). Dividing one item's total by the step
        count rescales the shared knowledge to that run's granularity.

        When a measured whole-run total exists for ``kind`` it is preferred over
        the summed phase means: the recorded end-to-end wall time is the true
        per-behavior cost, whereas the phase-sum systematically drifts (see the
        module docstring). The phase-sum remains the fallback for a run kind that
        has never completed before.
        """
        steps = max(1, int(steps))
        per_behavior = self.run_total_per_behavior(kind) if kind else 0.0
        if per_behavior <= 0.0:
            per_behavior = self.behavior_total()
        if per_behavior > 0.0:
            return per_behavior / steps
        return self.overall_mean()

    # ------------------------------------------------------------------
    # Whole-run totals (the primary, drift-free ETA anchor)
    # ------------------------------------------------------------------
    def record_run_total(self, kind: str, total_wall_seconds: float, n_units: int) -> None:
        """Fold one completed run's end-to-end wall time into its kind's mean.

        ``total_wall_seconds`` is the whole run's measured duration and
        ``n_units`` the number of behaviors it processed; the stored quantity is
        the per-behavior mean so it scales to future runs of any behavior count.
        """
        if not kind or total_wall_seconds is None:
            return
        n = max(1, int(n_units))
        per = float(total_wall_seconds) / n
        if per <= 0.0 or per > _MAX_SANE_SECONDS:
            return
        e = self._run_totals.setdefault(str(kind), {"seconds": 0.0, "n": 0.0})
        w = min(e["n"], _WEIGHT_CAP)
        e["seconds"] = (e["seconds"] * w + per) / (w + 1.0)
        e["n"] = e["n"] + 1.0

    def has_run_total(self, kind: str | None) -> bool:
        e = self._run_totals.get(str(kind)) if kind else None
        return bool(e and e["n"] > 0)

    def run_total_per_behavior(self, kind: str | None) -> float:
        """Measured mean end-to-end seconds for one behavior of this run kind."""
        e = self._run_totals.get(str(kind)) if kind else None
        return e["seconds"] if e and e["n"] > 0 else 0.0

    def run_total_seconds(self, kind: str | None, n_units: int) -> float:
        """Estimated whole-run wall time = per-behavior mean × behavior count."""
        return self.run_total_per_behavior(kind) * max(1, int(n_units))

    def has_data(self) -> bool:
        return any(e["n"] > 0 for e in self._phases.values()) or any(
            e["n"] > 0 for e in self._run_totals.values()
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "phases": {
                k: {"seconds": round(v["seconds"], 3), "n": v["n"]}
                for k, v in self._phases.items()
            },
            "run_totals": {
                k: {"seconds": round(v["seconds"], 3), "n": v["n"]}
                for k, v in self._run_totals.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RunTimingProfile":
        if not isinstance(d, dict):
            return cls()
        return cls(d.get("phases"), d.get("run_totals"))

    @classmethod
    def load(cls, path: Path) -> "RunTimingProfile":
        try:
            from abel.storage.file_store import read_json  # noqa: PLC0415
            return cls.from_dict(read_json(Path(path), {}))
        except Exception:
            return cls()

    def save(self, path: Path) -> None:
        try:
            from abel.storage.file_store import write_json  # noqa: PLC0415
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            write_json(p, self.to_dict())
        except Exception:
            pass
