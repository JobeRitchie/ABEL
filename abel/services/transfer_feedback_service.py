"""Transfer Feedback — assess how well a Direct Use run transferred.

After applying a trained model to a new project's videos (Direct Use), this
service inspects the results and estimates how trustworthy they look, per
subject and across the population.  It deliberately avoids comparing absolute
metrics to the base project (bout thresholds/scale differ between runs and
would false-flag everyone); instead it relies on:

* within-run population outliers (robust z / MAD across the run's subjects),
* divergence of each subject's *relative* behaviour profile from the base
  project's expected profile (shape, not magnitude),
* threshold-independent confidence-run anomalies from the per-frame
  probability traces (stuck-high or lost-low stretches),
* and absolute "produced essentially nothing" checks (0 or 1 bout).

The output is a per-subject health score + human-readable flags, plus a
population summary, suitable for a deep-dive UI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("abel")

# Tunables (kept explicit so the UI / tests can reason about them).
OUTLIER_Z = 3.5            # robust-z magnitude that counts as a population outlier
STUCK_HIGH_SECONDS = 60.0  # sustained prob>0.9 run length that looks "stuck on"
LOST_LOW_SECONDS = 45.0    # sustained max-prob<0.3 run length that looks "lost"
HIGH_PROB = 0.9
LOW_PROB = 0.3

# Health-score penalty weights per flag type.
_PENALTY = {
    "zero_bout": 18.0,
    "outlier": 12.0,
    "stuck_high": 16.0,
    "lost_low": 16.0,
    "profile": 20.0,
}


@dataclass
class SubjectFeedback:
    subject: str
    sessions: list[str] = field(default_factory=list)
    behavior_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    health_score: float = 100.0
    category: str = "Good"  # Good | Warning | Poor


@dataclass
class TransferFeedbackReport:
    subjects: list[SubjectFeedback] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    population: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    has_traces: bool = False


class TransferFeedbackService:
    """Analyse a Direct Use target project for transfer quality."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        target_root: Path,
        source_root: Path | None = None,
        fps: float = 30.0,
    ) -> TransferFeedbackReport:
        report = TransferFeedbackReport()

        rows = self._load_summary_rows(target_root)
        if not rows:
            report.warnings.append(
                "No analytics found for this project. Open the Analytics tab and "
                "refresh analytics first, then re-run feedback."
            )
            return report

        behaviors = sorted({str(r.get("behavior", "")).strip() for r in rows if r.get("behavior")})
        report.behaviors = behaviors

        # Aggregate per subject × behaviour (a subject may have several sessions).
        per_subject = self._aggregate(rows)

        # Population baselines (robust) for outlier detection.
        pop_stats = self._population_stats(per_subject, behaviors)

        # Base project's expected relative profile (shape over behaviours) —
        # kept for display.  The *flag* uses the within-run reference instead,
        # because a whole Direct Use run often shifts uniformly vs the base
        # (which would otherwise flag every subject).
        base_profile = self._base_profile(source_root, behaviors) if source_root else {}
        run_profile = self._run_profile(per_subject, behaviors)
        profile_dist = {
            sub: self._profile_distance(entry["behaviors"], run_profile, behaviors)
            for sub, entry in per_subject.items()
        }
        profile_stats = self._robust_center_scale(list(profile_dist.values()))

        # Confidence runs from probability traces.
        conf_by_subject = self._confidence_runs(target_root, fps)
        report.has_traces = bool(conf_by_subject)

        subjects: list[SubjectFeedback] = []
        for subject, beh_metrics in sorted(per_subject.items()):
            fb = SubjectFeedback(subject=subject)
            fb.behavior_metrics = beh_metrics["behaviors"]
            fb.sessions = sorted(beh_metrics["sessions"])
            fb.confidence = conf_by_subject.get(subject, {})
            penalties = 0.0

            # ── Flag 1: 0/1 bout for an active behaviour ─────────────
            for beh in behaviors:
                m = fb.behavior_metrics.get(beh)
                if m is None:
                    continue
                if m["n_bouts"] <= 1:
                    fb.flags.append(f"Almost no “{beh}” detected ({int(m['n_bouts'])} bout)")
                    penalties += _PENALTY["zero_bout"]

            # ── Flag 2: population outliers (bouts / time) ───────────
            for beh in behaviors:
                m = fb.behavior_metrics.get(beh)
                if m is None:
                    continue
                for key, label in (("n_bouts", "bout count"), ("time_spent_s", "time")):
                    z = self._robust_z(m[key], pop_stats[beh][key])
                    if abs(z) >= OUTLIER_Z:
                        direction = "high" if z > 0 else "low"
                        fb.flags.append(
                            f"Unusual {label} for “{beh}” ({direction} vs other subjects)"
                        )
                        penalties += _PENALTY["outlier"]

            # ── Flag 3: confidence-run anomalies ─────────────────────
            hi = fb.confidence.get("longest_high_run_s", 0.0)
            lo = fb.confidence.get("longest_low_run_s", 0.0)
            if hi >= STUCK_HIGH_SECONDS:
                fb.flags.append(
                    f"Confidence stuck high for {hi:.0f}s straight (possible over-firing)"
                )
                penalties += _PENALTY["stuck_high"]
            if lo >= LOST_LOW_SECONDS:
                fb.flags.append(
                    f"Confidence stayed very low for {lo:.0f}s straight (model unsure)"
                )
                penalties += _PENALTY["lost_low"]

            # ── Flag 4: behaviour-profile divergence (vs peers) ──────
            dist = profile_dist.get(subject, 0.0)
            fb.confidence["profile_distance"] = dist
            if base_profile:
                fb.confidence["base_profile_distance"] = self._profile_distance(
                    fb.behavior_metrics, base_profile, behaviors
                )
            if abs(self._robust_z(dist, profile_stats)) >= OUTLIER_Z:
                fb.flags.append(
                    "Overall behaviour mix differs sharply from the other subjects "
                    "in this run"
                )
                penalties += _PENALTY["profile"]

            fb.health_score = max(0.0, 100.0 - penalties)
            fb.category = (
                "Good" if fb.health_score >= 80
                else "Warning" if fb.health_score >= 55
                else "Poor"
            )
            subjects.append(fb)

        # Worst first.
        subjects.sort(key=lambda s: (s.health_score, s.subject))
        report.subjects = subjects

        report.population = {
            "n_subjects": len(subjects),
            "n_poor": sum(1 for s in subjects if s.category == "Poor"),
            "n_warning": sum(1 for s in subjects if s.category == "Warning"),
            "n_good": sum(1 for s in subjects if s.category == "Good"),
            "mean_health": float(np.mean([s.health_score for s in subjects])) if subjects else 0.0,
        }
        return report

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_summary_rows(target_root: Path) -> list[dict[str, Any]]:
        path = target_root / "derived" / "analytics_cache" / "analytics_cache.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return list(data.get("summary_rows", []) or [])

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Aggregate summary rows into {subject: {behaviors:{beh:{...}}, sessions:set}}."""
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            sub = str(r.get("subject", "")).strip()
            beh = str(r.get("behavior", "")).strip()
            if not sub or not beh:
                continue
            entry = out.setdefault(sub, {"behaviors": {}, "sessions": set()})
            entry["sessions"].add(str(r.get("session_id", "")))
            bm = entry["behaviors"].setdefault(
                beh, {"n_bouts": 0.0, "time_spent_s": 0.0, "_wdur": 0.0}
            )
            nb = float(r.get("n_bouts", 0) or 0)
            ts = float(r.get("time_spent_s", 0) or 0)
            bm["n_bouts"] += nb
            bm["time_spent_s"] += ts
            bm["_wdur"] += ts  # time forms the weighted-duration numerator
        # finalize mean bout duration
        for entry in out.values():
            for bm in entry["behaviors"].values():
                bm["mean_bout_s"] = (bm["_wdur"] / bm["n_bouts"]) if bm["n_bouts"] > 0 else 0.0
                bm.pop("_wdur", None)
        return out

    @staticmethod
    def _population_stats(
        per_subject: dict[str, dict[str, Any]], behaviors: list[str]
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Per behaviour, robust center+scale (median, MAD) for each metric."""
        stats: dict[str, dict[str, tuple[float, float]]] = {}
        for beh in behaviors:
            stats[beh] = {}
            for key in ("n_bouts", "time_spent_s"):
                vals = np.array([
                    entry["behaviors"].get(beh, {}).get(key, 0.0)
                    for entry in per_subject.values()
                ], dtype=float)
                median = float(np.median(vals)) if vals.size else 0.0
                mad = float(np.median(np.abs(vals - median))) if vals.size else 0.0
                stats[beh][key] = (median, mad)
        return stats

    @staticmethod
    def _robust_z(value: float, center_scale: tuple[float, float]) -> float:
        median, mad = center_scale
        if mad <= 1e-9:
            return 0.0  # no spread → can't call an outlier
        return 0.6745 * (value - median) / mad

    @staticmethod
    def _robust_center_scale(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return median, mad

    def _run_profile(
        self, per_subject: dict[str, dict[str, Any]], behaviors: list[str]
    ) -> dict[str, float]:
        """Median within-run relative time-profile across this run's subjects."""
        vectors: list[np.ndarray] = []
        for entry in per_subject.values():
            vec = np.array([
                entry["behaviors"].get(b, {}).get("time_spent_s", 0.0) for b in behaviors
            ], dtype=float)
            total = vec.sum()
            if total > 0:
                vectors.append(vec / total)
        if not vectors:
            return {}
        med = np.median(np.array(vectors), axis=0)
        s = med.sum()
        if s > 0:
            med = med / s
        return {b: float(med[i]) for i, b in enumerate(behaviors)}

    # ------------------------------------------------------------------
    # Base relative profile
    # ------------------------------------------------------------------

    def _base_profile(self, source_root: Path, behaviors: list[str]) -> dict[str, float]:
        rows = self._load_summary_rows(source_root)
        if not rows:
            return {}
        per_subject = self._aggregate(rows)
        vectors: list[np.ndarray] = []
        for entry in per_subject.values():
            vec = np.array([
                entry["behaviors"].get(b, {}).get("time_spent_s", 0.0) for b in behaviors
            ], dtype=float)
            total = vec.sum()
            if total > 0:
                vectors.append(vec / total)
        if not vectors:
            return {}
        mean_vec = np.mean(vectors, axis=0)
        return {b: float(mean_vec[i]) for i, b in enumerate(behaviors)}

    @staticmethod
    def _profile_distance(
        behavior_metrics: dict[str, dict[str, float]],
        base_profile: dict[str, float],
        behaviors: list[str],
    ) -> float:
        """Cosine distance between a subject's time-profile and the base profile."""
        subj = np.array([
            behavior_metrics.get(b, {}).get("time_spent_s", 0.0) for b in behaviors
        ], dtype=float)
        total = subj.sum()
        if total <= 0:
            return 1.0
        subj = subj / total
        base = np.array([base_profile.get(b, 0.0) for b in behaviors], dtype=float)
        nb = np.linalg.norm(base)
        ns = np.linalg.norm(subj)
        if nb <= 1e-9 or ns <= 1e-9:
            return 1.0
        cos = float(np.dot(subj, base) / (nb * ns))
        return max(0.0, 1.0 - cos)

    # ------------------------------------------------------------------
    # Confidence runs from probability traces
    # ------------------------------------------------------------------

    def _confidence_runs(self, target_root: Path, fps: float) -> dict[str, dict[str, float]]:
        trace_paths = self._trace_paths(target_root)
        if not trace_paths:
            return {}
        sid_to_sub = self._session_subject_map(target_root)
        fps = fps if fps and fps > 0 else 30.0

        import pandas as pd  # noqa: PLC0415
        per_subject: dict[str, dict[str, float]] = {}
        for sid, path in trace_paths.items():
            p = Path(path)
            if not p.exists():
                continue
            try:
                t = pd.read_parquet(p)
            except Exception:
                continue
            prob_cols = [c for c in t.columns if c.startswith("prob_")]
            if not prob_cols:
                continue
            maxp = t[prob_cols].to_numpy(dtype=float).max(axis=1)
            hi_run = self._longest_run(maxp > HIGH_PROB) / fps
            lo_run = self._longest_run(maxp < LOW_PROB) / fps
            mean_conf = float(maxp.mean()) if maxp.size else 0.0
            sub = sid_to_sub.get(str(sid), str(sid))
            agg = per_subject.setdefault(
                sub, {"longest_high_run_s": 0.0, "longest_low_run_s": 0.0,
                      "mean_conf": 0.0, "_n": 0.0}
            )
            agg["longest_high_run_s"] = max(agg["longest_high_run_s"], hi_run)
            agg["longest_low_run_s"] = max(agg["longest_low_run_s"], lo_run)
            agg["mean_conf"] += mean_conf
            agg["_n"] += 1.0
        for agg in per_subject.values():
            n = agg.pop("_n", 1.0) or 1.0
            agg["mean_conf"] = agg["mean_conf"] / n
        return per_subject

    @staticmethod
    def _longest_run(mask: np.ndarray) -> int:
        """Longest run of True in a boolean array."""
        best = run = 0
        for v in mask:
            run = run + 1 if v else 0
            if run > best:
                best = run
        return best

    @staticmethod
    def _trace_paths(target_root: Path) -> dict[str, str]:
        latest = target_root / "derived" / "temporal_refinement" / "target_behavior" / "latest.json"
        if not latest.exists():
            return {}
        try:
            d = json.loads(latest.read_text(encoding="utf-8"))
            inf_dir = str(d.get("inference_dir", "") or "").strip()
            if not inf_dir:
                return {}
            im = Path(inf_dir) / "inference_manifest.json"
            if not im.exists():
                return {}
            m = json.loads(im.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in (m.get("trace_paths", {}) or {}).items()}
        except Exception:
            return {}

    @staticmethod
    def _session_subject_map(target_root: Path) -> dict[str, str]:
        for rel in ("derived/review_tables/import_manifest.json",
                    "derived/import_manifest.json"):
            p = target_root / rel
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                out: dict[str, str] = {}
                for s in d.get("linked_sessions", []) or []:
                    sid = str(s.get("session_id", "")).strip()
                    sub = str(s.get("subject_id", "") or sid).strip()
                    if sid:
                        out[sid] = sub
                if out:
                    return out
        return {}
