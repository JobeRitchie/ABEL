"""Human clip-review effort: what did labeling this project actually cost?

Every other analysis in this suite measures what the *model* did.  This one
measures what the *person* did — the wall-clock labor that produced the labels
every other number rests on.  It is a pure read of files already on disk
(``derived/review_tables/review_decisions.json``); nothing is trained, nothing is
written back to the project, and it costs about a second per project.

How the time is measured
------------------------
ABEL stamps every review decision with the moment it was committed.  It does not
record a start time, so a clip's review duration is measured as the **gap to the
previous decision** by the same person: you look at a clip, you judge it, you
commit — the interval between two commits is the time the second clip took.
Three gap classes fall out of that, and only one of them is review time:

``batch``   gap < :data:`BATCH_SEC` (0.05 s)
    Not a human look.  One UI action — a bulk "assign behavior to selection", a
    held-down keyboard shortcut, a temporal-review interval tiling into windows —
    writes many decisions in one loop.  Counting these as clips reviewed in
    ~0 seconds would drag the mean rate toward zero.

``active``  :data:`BATCH_SEC` <= gap <= :data:`BREAK_SEC` (120 s)
    A real per-clip look.  These, and only these, are summed into review time and
    are the sample behind every seconds-per-clip statistic here.

``break``   gap > :data:`BREAK_SEC`
    The reviewer walked away.  Excluded entirely — charging a lunch break to the
    next clip would turn a 3-hour labeling job into a 3-week one.

What this deliberately under-counts
-----------------------------------
The first decision after every break has no measurable duration (its gap *is* the
break), so it contributes 0 s instead of its true few seconds.  Active hours are
therefore a floor.  :attr:`ReviewEffortResult.active_hours_adjusted` adds the
missing sittings back at the project's own median rate; both are reported, and the
unadjusted figure is the one to quote when a conservative number is wanted.

Who counts as a reviewer
------------------------
Three channels write into the same decisions file and they are not the same work:

* **clip review** — the review queue, one human judgement per clip.  This is the
  only channel timed, and the only one behind the headline rates.
* ``temporal_feedback`` — corrections made by scrubbing a trace in the Temporal
  Review tab.  Real human work, but one action tiles an interval into many
  windows in a single loop, so its decisions are near-simultaneous and would
  corrupt a per-clip rate.  Counted and reported separately, never timed.
* ``imported:<tag>`` — labels copied in from another project by the model
  refinement service.  Not this project's human work at all; excluded everywhere.

Caveat on re-reviews
--------------------
:meth:`ReviewService.upsert_decision` replaces a clip's record in place and
re-stamps it, so a clip reviewed twice keeps only the later timestamp.  The
measured cost is therefore the cost of the *surviving* pass over each clip, not
of every pass ever made — another reason to read these numbers as a floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from abel.validation.datamodel import ProjectRef

#: Gaps below this are one bulk UI action, not a human look at a clip.
BATCH_SEC = 0.05

#: Gaps above this mean the reviewer left; the time is not charged to any clip.
BREAK_SEC = 120.0

#: ``reviewer`` value written by the Temporal Review tab's interval tiling.
REVIEWER_TEMPORAL = "temporal_feedback"

#: ``reviewer`` prefix written for labels imported from another project.
IMPORT_PREFIX = "imported:"

#: Real-time multipliers used for the manual frame-by-frame scoring comparison.
#: Manual ethogram scoring is usually quoted at 1-3x the recording's duration
#: (pause, rewind, re-score), so the comparison is given as a band, not a point.
MANUAL_PASSES = (1.0, 2.0, 3.0)

#: The seconds-per-clip figure ``rare_discovery`` assumes when it converts clip
#: budgets into scoring minutes.  Quoted here purely so this analysis can say how
#: the measured rate compares; nothing in this module changes that constant.
ASSUMED_SEC_PER_CLIP = 4.0


@dataclass
class ReviewEffortResult:
    """One project's human review cost.  ``error`` set means nothing else is valid."""

    project_id: str
    project_name: str = ""

    # ── decision accounting (every row in the file lands in exactly one bucket) ──
    n_decisions_total: int = 0
    n_clip_review: int = 0        # the review queue — the only timed channel
    n_temporal_feedback: int = 0  # trace-scrubbing corrections, counted not timed
    n_imported: int = 0           # copied from another project; not human work here

    # ── gap classification within the clip-review channel ──
    n_timed: int = 0    # gaps counted as one clip's review
    n_batch: int = 0    # bulk UI actions
    n_breaks: int = 0   # reviewer away

    # ── seconds per clip (over the timed gaps) ──
    median_sec: float = float("nan")
    mean_sec: float = float("nan")
    p25_sec: float = float("nan")
    p75_sec: float = float("nan")
    p90_sec: float = float("nan")

    # ── totals ──
    active_hours: float = float("nan")           # sum of timed gaps (a floor)
    active_hours_adjusted: float = float("nan")  # + one median clip per sitting
    clips_per_hour: float = float("nan")         # sustained rate while working

    # ── against the data volume it bought ──
    video_hours: float = float("nan")            # all footage in the project
    footage_reviewed_hours: float = float("nan")  # footage actually looked at
    footage_reviewed_frac: float = float("nan")  # of all footage
    review_hours_per_video_hour: float = float("nan")

    first_decision: str = ""   # ISO, provenance for what window was measured
    last_decision: str = ""
    error: str = ""

    #: The timed gaps themselves, for pooling and the distribution figure.  Not
    #: serialized into the CSV (one row per clip would be ~40k rows per project).
    timed_sec: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def manual_hours(self, passes: float) -> float:
        """Hours a human would spend scoring every frame at ``passes`` x real-time."""
        if not np.isfinite(self.video_hours):
            return float("nan")
        return float(self.video_hours * passes)

    def saving_factor(self, passes: float) -> float:
        """How many times less human time ABEL review took than manual scoring."""
        manual = self.manual_hours(passes)
        if not (np.isfinite(manual) and np.isfinite(self.active_hours)) \
                or self.active_hours <= 0:
            return float("nan")
        return float(manual / self.active_hours)


# ── reading the decisions file ──────────────────────────────────────────────


def decisions_path(project_root: Path) -> Path:
    """Where :class:`ReviewService` persists the decision log."""
    return Path(project_root) / "derived" / "review_tables" / "review_decisions.json"


def _load_decisions(project_root: Path) -> list[dict[str, Any]]:
    """Raw decision rows, tolerating both the wrapped and bare-list shapes."""
    path = decisions_path(project_root)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = raw.get("decisions", []) if isinstance(raw, dict) else raw
    return [r for r in rows if isinstance(r, dict)]


def _channel(row: dict[str, Any]) -> str:
    """Which of the three writers produced this decision (see module docstring)."""
    reviewer = str(row.get("reviewer", ""))
    if reviewer.startswith(IMPORT_PREFIX):
        return "imported"
    if reviewer == REVIEWER_TEMPORAL:
        return "temporal_feedback"
    return "clip_review"


def _timestamps(rows: Iterable[dict[str, Any]]) -> list[datetime]:
    """Parsed decision timestamps, ascending; unparseable rows are dropped.

    Stamped by ``datetime.utcnow()`` and stored naive (see the timestamps note in
    the schema).  Only *differences* are used here, so the missing zone is
    harmless — no local-time conversion is needed or attempted.
    """
    out: list[datetime] = []
    for row in rows:
        value = row.get("timestamp")
        if not value:
            continue
        try:
            out.append(datetime.fromisoformat(str(value)))
        except (TypeError, ValueError):
            continue
    out.sort()
    return out


def classify_gaps(
    stamps: list[datetime], *, break_sec: float = BREAK_SEC,
    batch_sec: float = BATCH_SEC,
) -> tuple[np.ndarray, int, int]:
    """Split consecutive-decision gaps into (timed seconds, n_batch, n_breaks)."""
    timed: list[float] = []
    n_batch = n_breaks = 0
    for previous, current in zip(stamps, stamps[1:]):
        gap = (current - previous).total_seconds()
        if gap > break_sec:
            n_breaks += 1
        elif gap < batch_sec:
            n_batch += 1
        else:
            timed.append(gap)
    return np.asarray(timed, dtype=float), n_batch, n_breaks


# ── the data volume the effort bought ───────────────────────────────────────


def _video_hours(project: ProjectRef) -> float:
    """Total footage in the project, in hours.

    Reuses the rare-discovery whole-video measurement (segment-pool frame extents
    / fps) so the two analyses cannot disagree about how much video exists.
    """
    try:
        from abel.validation.analyses.rare_discovery import (  # noqa: PLC0415
            whole_video_minutes,
        )

        minutes = whole_video_minutes(project)
    except Exception:
        return float("nan")
    return float(minutes / 60.0) if np.isfinite(minutes) else float("nan")


def _footage_reviewed_hours(rows: list[dict[str, Any]], fps: float) -> float:
    """Hours of footage the reviewer actually watched, from the clips' own bounds.

    Measured per clip from ``adjusted_start_frame``/``adjusted_end_frame`` rather
    than assumed from a nominal window length — projects set different clip
    durations, and the decision rows carry the real ones.
    """
    if not np.isfinite(fps) or fps <= 0:
        return float("nan")
    frames = 0
    seen = 0
    for row in rows:
        start, end = row.get("adjusted_start_frame"), row.get("adjusted_end_frame")
        if start is None or end is None:
            continue
        try:
            span = int(end) - int(start) + 1
        except (TypeError, ValueError):
            continue
        if span > 0:
            frames += span
            seen += 1
    if seen == 0:
        return float("nan")
    return float(frames / fps / 3600.0)


# ── per-project measurement ─────────────────────────────────────────────────


def measure_project(
    project: ProjectRef, *, break_sec: float = BREAK_SEC, batch_sec: float = BATCH_SEC,
) -> ReviewEffortResult:
    """Measure one project's human clip-review cost.  Never raises."""
    result = ReviewEffortResult(project_id=project.project_id,
                               project_name=project.name or project.project_id)
    try:
        rows = _load_decisions(project.root)
    except Exception as exc:  # noqa: BLE001 — a missing/odd file must not sink a run
        result.error = f"could not read review decisions: {type(exc).__name__}: {exc}"
        return result
    if not rows:
        result.error = "no review decisions on disk"
        return result

    result.n_decisions_total = len(rows)
    clip_rows = [r for r in rows if _channel(r) == "clip_review"]
    result.n_clip_review = len(clip_rows)
    result.n_temporal_feedback = sum(1 for r in rows if _channel(r) == "temporal_feedback")
    result.n_imported = sum(1 for r in rows if _channel(r) == "imported")

    stamps = _timestamps(clip_rows)
    if len(stamps) < 2:
        result.error = (f"only {len(stamps)} timestamped clip-review decisions "
                        "— need at least 2 to measure a gap")
        return result
    result.first_decision = stamps[0].isoformat(timespec="seconds")
    result.last_decision = stamps[-1].isoformat(timespec="seconds")

    timed, n_batch, n_breaks = classify_gaps(
        stamps, break_sec=break_sec, batch_sec=batch_sec)
    result.timed_sec = timed
    result.n_timed = int(timed.size)
    result.n_batch = int(n_batch)
    result.n_breaks = int(n_breaks)
    if timed.size == 0:
        result.error = ("no gaps fell in the per-clip band — every decision was "
                        "either a bulk action or separated by a break")
        return result

    result.median_sec = float(np.median(timed))
    result.mean_sec = float(timed.mean())
    result.p25_sec = float(np.percentile(timed, 25))
    result.p75_sec = float(np.percentile(timed, 75))
    result.p90_sec = float(np.percentile(timed, 90))

    result.active_hours = float(timed.sum() / 3600.0)
    # Each sitting's first clip has no measurable gap; add it back at the median.
    n_sittings = n_breaks + 1
    result.active_hours_adjusted = float(
        (timed.sum() + n_sittings * result.median_sec) / 3600.0)
    result.clips_per_hour = (float(result.n_timed / result.active_hours)
                             if result.active_hours > 0 else float("nan"))

    result.video_hours = _video_hours(project)
    result.footage_reviewed_hours = _footage_reviewed_hours(clip_rows, float(project.fps))
    if np.isfinite(result.video_hours) and result.video_hours > 0:
        result.review_hours_per_video_hour = float(result.active_hours / result.video_hours)
        if np.isfinite(result.footage_reviewed_hours):
            result.footage_reviewed_frac = float(
                result.footage_reviewed_hours / result.video_hours)
    return result


def run_review_effort(
    projects: list[ProjectRef], *, break_sec: float = BREAK_SEC,
    batch_sec: float = BATCH_SEC, log: Callable[[str], None] | None = None,
) -> list[ReviewEffortResult]:
    """Measure every project's review cost (one cheap pass over the decision logs)."""
    results: list[ReviewEffortResult] = []
    for project in projects:
        if log:
            log(f"[{project.name}] reading review decisions…")
        results.append(measure_project(project, break_sec=break_sec, batch_sec=batch_sec))
    return results


# ── pooling ─────────────────────────────────────────────────────────────────


def _usable(results: Iterable[ReviewEffortResult]) -> list[ReviewEffortResult]:
    return [r for r in results if r is not None and not r.error and r.n_timed > 0]


def pooled_summary(results: list[ReviewEffortResult]) -> dict[str, float]:
    """Totals and pooled per-clip statistics across every measured project.

    The pooled seconds-per-clip statistics are computed over the *concatenated
    gaps*, not by averaging the per-project medians — a project that contributed
    9,000 clips should not weigh the same as one that contributed 900.
    """
    usable = _usable(results)
    if not usable:
        return {}
    gaps = np.concatenate([r.timed_sec for r in usable])
    active_hours = float(gaps.sum() / 3600.0)

    # Every ratio against data volume is pooled over ONLY the projects whose video
    # hours could be measured, numerator included.  Dividing all projects' review
    # hours by some projects' footage would inflate the rate by exactly the share
    # of projects whose segment pool is missing.
    with_video = [r for r in usable if np.isfinite(r.video_hours) and r.video_hours > 0]
    video_hours = float(sum(r.video_hours for r in with_video))
    video_active_hours = float(sum(r.active_hours for r in with_video
                                   if np.isfinite(r.active_hours)))
    reviewed = [r.footage_reviewed_hours for r in with_video
                if np.isfinite(r.footage_reviewed_hours)]
    footage_hours = float(np.sum(reviewed)) if reviewed else float("nan")

    out: dict[str, float] = {
        "n_projects": float(len(usable)),
        "n_projects_with_video": float(len(with_video)),
        "n_clips_timed": float(gaps.size),
        "n_clip_review_decisions": float(sum(r.n_clip_review for r in usable)),
        "median_sec": float(np.median(gaps)),
        "mean_sec": float(gaps.mean()),
        "p25_sec": float(np.percentile(gaps, 25)),
        "p75_sec": float(np.percentile(gaps, 75)),
        "p90_sec": float(np.percentile(gaps, 90)),
        "active_hours": active_hours,
        "active_hours_adjusted": float(
            np.nansum([r.active_hours_adjusted for r in usable])),
        "clips_per_hour": float(gaps.size / active_hours) if active_hours > 0
        else float("nan"),
        "video_hours": video_hours,
        "footage_reviewed_hours": footage_hours,
    }
    if video_hours > 0:
        # ``video_active_hours``, not ``active_hours``: same project subset on both
        # sides of every ratio (see the note above).
        out["video_active_hours"] = video_active_hours
        out["review_hours_per_video_hour"] = video_active_hours / video_hours
        out["review_min_per_video_hour"] = 60.0 * video_active_hours / video_hours
        if np.isfinite(footage_hours):
            out["footage_reviewed_frac"] = footage_hours / video_hours
        for passes in MANUAL_PASSES:
            out[f"manual_hours_{passes:g}x"] = video_hours * passes
            if video_active_hours > 0:
                out[f"saving_factor_{passes:g}x"] = (
                    video_hours * passes / video_active_hours)
    return out


# ── tidy exports ────────────────────────────────────────────────────────────


def results_to_frame(results: list[ReviewEffortResult]) -> pd.DataFrame:
    """One row per project — the paste-ready cost table."""
    rows = []
    for r in results:
        if r is None:
            continue
        row = {
            "project": r.project_id,
            "n_clip_review_decisions": r.n_clip_review,
            "n_clips_timed": r.n_timed,
            "n_bulk_actions": r.n_batch,
            "n_breaks": r.n_breaks,
            "n_temporal_feedback": r.n_temporal_feedback,
            "n_imported_excluded": r.n_imported,
            "median_sec_per_clip": r.median_sec,
            "mean_sec_per_clip": r.mean_sec,
            "p25_sec_per_clip": r.p25_sec,
            "p75_sec_per_clip": r.p75_sec,
            "p90_sec_per_clip": r.p90_sec,
            "active_review_hours": r.active_hours,
            "active_review_hours_adjusted": r.active_hours_adjusted,
            "clips_per_hour": r.clips_per_hour,
            "video_hours": r.video_hours,
            "footage_reviewed_hours": r.footage_reviewed_hours,
            "footage_reviewed_frac": r.footage_reviewed_frac,
            "review_hours_per_video_hour": r.review_hours_per_video_hour,
        }
        for passes in MANUAL_PASSES:
            row[f"manual_hours_{passes:g}x"] = r.manual_hours(passes)
            row[f"saving_vs_manual_{passes:g}x"] = r.saving_factor(passes)
        row["first_decision"] = r.first_decision
        row["last_decision"] = r.last_decision
        row["error"] = r.error
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_to_frame(results: list[ReviewEffortResult]) -> pd.DataFrame:
    """The pooled summary as a one-row frame (same units as the per-project table)."""
    summary = pooled_summary(results)
    if not summary:
        return pd.DataFrame()
    return pd.DataFrame([{"scope": "pooled", **summary}])


def summary_text(results: list[ReviewEffortResult]) -> str:
    """The headline paragraph, for the GUI status line and the run log."""
    summary = pooled_summary(results)
    if not summary:
        failed = [r for r in results if r is not None and r.error]
        if failed:
            return "No review effort could be measured:\n" + "\n".join(
                f"  • {r.project_id}: {r.error}" for r in failed)
        return "No review effort could be measured."

    lines = [
        f"Human clip review: {summary['active_hours']:.1f} h of active labeling "
        f"across {int(summary['n_projects'])} project(s).",
        f"  • {summary['median_sec']:.1f} s per clip (median; IQR "
        f"{summary['p25_sec']:.1f}-{summary['p75_sec']:.1f} s, p90 "
        f"{summary['p90_sec']:.1f} s) over {int(summary['n_clips_timed']):,} timed clips.",
        f"  • {summary['clips_per_hour']:.0f} clips/hour sustained while working.",
    ]
    if "review_min_per_video_hour" in summary:
        lines.append(
            f"  • {summary['review_min_per_video_hour']:.1f} review-minutes per hour "
            f"of video ({summary['video_hours']:.0f} h of footage total).")
    if "footage_reviewed_frac" in summary:
        lines.append(
            f"  • the reviewer watched {summary['footage_reviewed_frac'] * 100:.2f}% "
            "of all footage.")
    for passes in MANUAL_PASSES:
        key = f"saving_factor_{passes:g}x"
        if key in summary:
            lines.append(
                f"  • vs. manual frame-by-frame scoring at {passes:g}x real-time "
                f"({summary[f'manual_hours_{passes:g}x']:.0f} h): "
                f"{summary[key]:.0f}x less human time.")
    lines.append(
        f"  • measured rate is {summary['median_sec']:.1f} s/clip; the rare-discovery "
        f"effort figures assume {ASSUMED_SEC_PER_CLIP:.1f} s/clip.")
    return "\n".join(lines)


# ── figure ──────────────────────────────────────────────────────────────────


def _short(name: str, width: int = 18) -> str:
    return name if len(name) <= width else name[:width - 1] + "…"


def plot_review_effort(results: list[ReviewEffortResult], save_path: Path) -> Path:
    """Three panels: per-clip time, review hours vs footage, and the manual anchor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    usable = _usable(results)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    def _blank(ax, title: str) -> None:
        ax.text(0.5, 0.5, "no measurable review time", ha="center", va="center",
                fontsize=9, color="#777")
        ax.axis("off")
        ax.set_title(title, fontsize=10.5)

    def _clean(ax) -> None:
        ax.grid(axis="y", alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ── panel 1: seconds per clip, distribution per project ──
    ax = axes[0]
    if usable:
        labels = [_short(r.project_id) for r in usable]
        data = [r.timed_sec for r in usable]
        box = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                         whis=(5, 95),
                         medianprops={"color": "#16161d", "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")
        for x, r in enumerate(usable, start=1):
            ax.text(x, r.median_sec, f"{r.median_sec:.1f}s", ha="center", va="bottom",
                    fontsize=8, color="#16161d")
        ax.axhline(ASSUMED_SEC_PER_CLIP, color="#C44E52", linestyle="--", linewidth=1.1,
                   zorder=0)
        ax.text(0.99, ASSUMED_SEC_PER_CLIP, f" assumed {ASSUMED_SEC_PER_CLIP:g}s",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color="#C44E52")
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(usable) + 1))
        ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
        ax.set_ylabel("seconds per clip (log)")
        ax.set_title("Time per clip reviewed\nbox = IQR · whiskers = 5-95%",
                     fontsize=10.5)
        _clean(ax)
    else:
        _blank(ax, "Time per clip reviewed")

    # ── panel 2: review hours against the footage they cover ──
    ax = axes[1]
    if usable:
        x = np.arange(len(usable))
        review = [r.active_hours for r in usable]
        video = [r.video_hours for r in usable]
        # Linear, never log: these are bars, and a log bar chart moves the baseline
        # off zero, which makes a 1.1 h bar and a 4.0 h bar look near-identical.
        ax.bar(x - 0.19, video, 0.38, color="#CCD3E0", edgecolor="white", linewidth=0.4,
               label="video in project")
        ax.bar(x + 0.19, review, 0.38, color="#55A868", edgecolor="white", linewidth=0.4,
               label="human review")
        for xi, v in zip(x, review):
            if np.isfinite(v):
                ax.text(xi + 0.19, v, f"{v:.1f}h", ha="center", va="bottom", fontsize=7.5)
        ax.margins(y=0.14)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(r.project_id) for r in usable], fontsize=8,
                           rotation=35, ha="right")
        ax.set_ylabel("hours")
        ax.set_title("Review effort vs. footage\n(active labeling only)", fontsize=10.5)
        ax.legend(fontsize=8, frameon=False)
        _clean(ax)
    else:
        _blank(ax, "Review effort vs. footage")

    # ── panel 3: the manual-scoring anchor, pooled ──
    ax = axes[2]
    summary = pooled_summary(results)
    manual = [(f"manual {p:g}x real-time", summary.get(f"manual_hours_{p:g}x"))
              for p in MANUAL_PASSES]
    manual = [(lab, v) for lab, v in manual if v is not None and np.isfinite(v)]
    if summary and manual:
        labels = [lab for lab, _ in manual] + ["ABEL clip review"]
        values = [v for _, v in manual] + [summary["active_hours"]]
        colors = ["#CCD3E0"] * len(manual) + ["#55A868"]
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4)
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.0f} h",
                    va="center", fontsize=8.5)
        # Linear for the same reason as panel 2 — these bars are read by length.
        ax.margins(x=0.16)
        ax.invert_yaxis()
        ax.set_xlabel("human hours")
        savings = [summary.get(f"saving_factor_{p:g}x", float("nan"))
                   for p in MANUAL_PASSES]
        savings = [s for s in savings if np.isfinite(s)]
        subtitle = (f"{min(savings):.0f}-{max(savings):.0f}x less human time "
                    "than manual scoring"
                    if savings else "pooled across projects")
        ax.set_title(f"Human cost, pooled\n{subtitle}", fontsize=10.5)
        ax.grid(axis="x", alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    else:
        _blank(ax, "Human cost, pooled")

    fig.suptitle("Human clip-review effort (lower = cheaper)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
