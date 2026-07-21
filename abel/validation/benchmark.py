"""Pipeline throughput benchmark: how fast is ABEL's data processing?

Measures the three stages a user actually waits on, per project, on one
representative session, and normalizes each by the video's real duration so the
numbers compare across datasets and machines:

1. **Feature extraction / session** — pose cleaning + pose features + (optional)
   video/context features + windowed representation.  Timed by running ABEL's real
   ``FeaturePrepService.prepare`` on one session with the cache disabled.  Runs in
   an isolated temp copy of the project's config, so it never overwrites the
   project's own feature caches.
2. **Model training (given features)** — time to fit one behavior classifier once
   the features exist.  Reuses the validation engine's real ``train_and_evaluate``
   fit; nothing is written to the project.
3. **Dense inference / video** — running the trained model(s) over every window of
   a new session via ABEL's real ``TemporalRefinementService`` dense inference.

Every stage is optional and independently selectable, so a machine that can't run
(say) context extraction still benchmarks the others.

Run it on whatever projects you like::

    python -m abel.validation.benchmark \\
        --projects c:/Users/jober/CIE_NSF "g:/.../Dominance" \\
        --stages extract train infer --out ./benchmark_out

Writes ``benchmark.csv`` (tidy, paste-ready) + ``benchmark.png`` (grouped bars).

NOTE: the *inference* stage re-runs dense temporal inference for the chosen
session on the real project (idempotent — it recomputes that session's traces from
the unchanged models/config).  Pass ``--stages extract train`` to skip it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from abel.validation import prism
from abel.validation.datamodel import ProjectRef, _safe_model_name

STAGE_EXTRACT = "extract"
STAGE_TRAIN = "train"
STAGE_INFER = "infer"
ALL_STAGES = [STAGE_EXTRACT, STAGE_TRAIN, STAGE_INFER]


# ── Session picking ─────────────────────────────────────────────────────────


@dataclass
class SessionPick:
    session_id: str
    subject_id: str
    pose_path: Path
    video_path: Path | None
    fps: float
    frame_count: int
    duration_sec: float


def _asset_path(asset) -> Path | None:
    for attr in ("local_path", "source_path"):
        p = getattr(asset, attr, None)
        if p and Path(str(p)).exists():
            return Path(str(p))
    return None


def pick_session(project_root: Path, need_video: bool) -> SessionPick | None:
    """First linked session whose pose (and video, if required) exists on disk."""
    from abel.services.import_service import ImportService

    manifest = ImportService().load_manifest(Path(project_root))
    if manifest is None:
        return None
    videos = {v.asset_id: v for v in getattr(manifest, "videos", [])}
    poses = {p.asset_id: p for p in getattr(manifest, "poses", [])}
    for s in getattr(manifest, "linked_sessions", []) or []:
        pa = poses.get(getattr(s, "pose_asset_id", None))
        va = videos.get(getattr(s, "video_asset_id", None))
        pose_path = _asset_path(pa) if pa else None
        video_path = _asset_path(va) if va else None
        if pose_path is None:
            continue
        if need_video and video_path is None:
            continue
        fps = float(getattr(va, "fps", None) or 30.0) if va else 30.0
        frames = int(getattr(va, "frame_count", 0) or getattr(pa, "frame_count", 0) or 0)
        dur = float(getattr(va, "duration_sec", 0.0) or 0.0) if va else 0.0
        if dur <= 0 and frames and fps:
            dur = frames / fps
        return SessionPick(
            session_id=str(s.session_id),
            subject_id=str(getattr(s, "subject_id", None) or s.session_id),
            pose_path=pose_path, video_path=video_path,
            fps=fps, frame_count=frames, duration_sec=dur)
    return None


def _snapshot(project_root: Path) -> dict[str, Any]:
    p = Path(project_root) / "derived" / "workflow_snapshot.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# ── Result record ───────────────────────────────────────────────────────────


@dataclass
class StageTiming:
    project_id: str
    stage: str                       # "extract" | "train" | "infer"
    detail: str = ""                 # e.g. behavior name, or "pose+context+repr"
    seconds: float = float("nan")    # wall-clock compute time
    video_seconds: float = float("nan")   # duration of the video processed (n/a for train)
    x_realtime: float = float("nan")      # seconds / video_seconds (lower = faster)
    faster_than_realtime: float = float("nan")  # video_seconds / seconds
    units: str = ""                  # extra scalar (n_windows, n_pos, n_features…)
    breakdown: str = ""              # JSON of sub-stage seconds (extraction)
    error: str = ""

    def to_row(self) -> dict:
        """Flat CSV row.

        ``breakdown`` holds a JSON object; a nested document inside one cell of a
        spreadsheet is unusable in Excel/Prism, so it is expanded into real
        ``breakdown_<substage>_sec`` columns.  ``x_realtime`` is dropped — it is
        just ``1 / faster_than_realtime``, and shipping both invites a reader to
        plot the wrong one.
        """
        d = asdict(self)
        raw = d.pop("breakdown", "") or ""
        d.pop("x_realtime", None)
        if raw:
            try:
                for key, val in json.loads(raw).items():
                    d[f"breakdown_{key}_sec"] = float(val)
            except (ValueError, TypeError):
                d["breakdown_raw"] = raw
        return d


def _normalize(seconds: float, video_seconds: float) -> tuple[float, float]:
    if video_seconds and video_seconds > 0 and seconds > 0:
        return seconds / video_seconds, video_seconds / seconds
    return float("nan"), float("nan")


# ── Stage 1: feature extraction ─────────────────────────────────────────────


def time_extraction(
    project_root: str | Path, *, log: Callable[[str], None] | None = None,
) -> StageTiming:
    """Time pose+context+representation extraction for one session (cache off)."""
    from abel.services.feature_prep_service import (
        FeaturePrepService, PrepConfig, SessionJob,
    )

    project_root = Path(project_root)
    project = ProjectRef.load(project_root)
    out = StageTiming(project_id=project.project_id, stage=STAGE_EXTRACT)
    pick = pick_session(project_root, need_video=project.use_video_features)
    if pick is None:
        out.error = "no session with required assets (pose/video) found on disk"
        return out
    out.video_seconds = pick.duration_sec

    snap = _snapshot(project_root)
    win = int(snap.get("segment_window_frames") or 60)
    stride = int(snap.get("segment_stride_frames") or 15)

    # Isolated temp project so we never overwrite the real feature caches.
    tmp_root = project_root / "derived" / "_benchmark_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    (tmp_root / "config").mkdir(parents=True, exist_ok=True)
    for name in ("project.yaml",):
        src = project_root / name
        if src.exists():
            shutil.copy2(src, tmp_root / name)
    src_cfg = project_root / "config"
    if src_cfg.is_dir():
        shutil.copytree(src_cfg, tmp_root / "config", dirs_exist_ok=True)

    job = SessionJob(
        session_id=pick.session_id, subject_id=pick.subject_id,
        pose_path=pick.pose_path, video_path=pick.video_path, fps=pick.fps)
    cfg = PrepConfig(
        use_video_features=project.use_video_features,
        segment_window_frames=win, segment_stride_frames=stride,
        reuse_cached=False)

    if log:
        log(f"[{project.project_id}] extracting features for session "
            f"{pick.session_id} ({pick.duration_sec:.0f}s video)…")
    try:
        t0 = time.perf_counter()
        res = FeaturePrepService().prepare(tmp_root, [job], cfg)
        out.seconds = float(time.perf_counter() - t0)
        out.breakdown = json.dumps({k: round(float(v), 3) for k, v in res.timings.items()})
        out.units = f"frames={pick.frame_count}"
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out.x_realtime, out.faster_than_realtime = _normalize(out.seconds, out.video_seconds)
    return out


# ── Stage 2: training (given features) ──────────────────────────────────────


def time_training(
    project_root: str | Path, behavior_names: list[str] | None = None,
    *, log: Callable[[str], None] | None = None,
) -> list[StageTiming]:
    """Time fitting each requested behavior classifier once, on the full dataset."""
    from abel.services.active_learning_trainer_service import ActiveLearningTrainerService
    from abel.validation import holdout, subsample
    from abel.validation.engine import run_one_config

    project_root = Path(project_root)
    project = ProjectRef.load(project_root)
    if not project.is_valid():
        return [StageTiming(project_id=project.project_id, stage=STAGE_TRAIN,
                            error="no training set")]

    if behavior_names:
        bids = project.behavior_ids_matching(behavior_names)
    else:
        bids = [b for b in project.behavior_names if b != "no_behavior"]

    trainer = ActiveLearningTrainerService()
    try:
        hsplit = holdout.split(project, min_confidence=1.0, test_size=0.25, seed=42)
    except Exception as exc:  # noqa: BLE001
        return [StageTiming(project_id=project.project_id, stage=STAGE_TRAIN,
                            error=f"holdout failed: {exc}")]

    pool = hsplit.train_pool
    out: list[StageTiming] = []
    for bid in bids:
        name = project.behavior_label(bid)
        n_pos = int(subsample.count_positives(pool, bid))
        if n_pos == 0:
            out.append(StageTiming(project_id=project.project_id, stage=STAGE_TRAIN,
                                   detail=name, error="no positives"))
            continue
        if log:
            log(f"[{project.project_id}] training {name} (n_pos={n_pos})…")
        r = run_one_config(
            trainer, project, str(bid), pool, hsplit.holdout, seed=42,
            n_pos_train=n_pos, n_neg_train=int(len(pool) - n_pos))
        out.append(StageTiming(
            project_id=project.project_id, stage=STAGE_TRAIN, detail=name,
            seconds=float(r.elapsed_sec_fit),
            units=f"n_pos={n_pos}, n_features={r.n_features}",
            error=str(r.error or "")))
    return out


# ── Stage 3: dense inference ────────────────────────────────────────────────


def _resolve_selected_models(project_root: Path, project: ProjectRef) -> dict[str, str]:
    """behavior_id -> model-dir name, resolved by scanning derived/models."""
    models_root = Path(project_root) / "derived" / "models"
    if not models_root.is_dir():
        return {}
    existing = {p.name for p in models_root.iterdir() if p.is_dir()}
    selected: dict[str, str] = {}
    for bid, name in project.behavior_names.items():
        if bid == "no_behavior":
            continue
        for token in (name, bid):
            safe = _safe_model_name(token)
            for cand in (f"behavior_model_{safe}", safe):
                if cand in existing and (models_root / cand / "metrics.json").exists():
                    selected[str(bid)] = cand
                    break
            if str(bid) in selected:
                break
    return selected


def time_inference(
    project_root: str | Path, *, log: Callable[[str], None] | None = None,
) -> StageTiming:
    """Time dense temporal inference over one session (all behaviors, all windows)."""
    project_root = Path(project_root)
    project = ProjectRef.load(project_root)
    out = StageTiming(project_id=project.project_id, stage=STAGE_INFER)
    pick = pick_session(project_root, need_video=False)
    if pick is None:
        out.error = "no session found on disk"
        return out
    out.video_seconds = pick.duration_sec

    selected = _resolve_selected_models(project_root, project)
    if not selected:
        out.error = "no trained behavior models found under derived/models"
        return out

    try:
        from abel.temporal_refinement.temporal_refinement_service import (
            TemporalRefinementConfig, TemporalRefinementService,
        )
        svc = TemporalRefinementService()
        svc.set_project(project_root)
        cfg = TemporalRefinementConfig()
        cfg.selected_behavior_models = dict(selected)
        if log:
            log(f"[{project.project_id}] dense inference on session "
                f"{pick.session_id} with {len(selected)} model(s)…")
        t0 = time.perf_counter()
        result = svc.run_temporal_refinement_inference(
            concept_id="target_behavior", sessions=[pick.session_id],
            mode="dense", config=cfg, force=True, max_sessions=1)
        out.seconds = float(time.perf_counter() - t0)
        n_win = ""
        if isinstance(result, dict):
            n_win = str(result.get("n_windows") or result.get("windows") or "")
        out.units = f"n_models={len(selected)}" + (f", n_windows={n_win}" if n_win else "")
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"

    out.x_realtime, out.faster_than_realtime = _normalize(out.seconds, out.video_seconds)
    return out


# ── Orchestration ───────────────────────────────────────────────────────────


def run_benchmark(
    project_roots: list[str], stages: list[str], behavior_names: list[str] | None,
    *, log: Callable[[str], None] | None = None,
) -> list[StageTiming]:
    results: list[StageTiming] = []
    for root in project_roots:
        if STAGE_EXTRACT in stages:
            results.append(time_extraction(root, log=log))
        if STAGE_TRAIN in stages:
            results.extend(time_training(root, behavior_names, log=log))
        if STAGE_INFER in stages:
            results.append(time_inference(root, log=log))
    return results


def results_to_frame(results: list[StageTiming]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def plot_benchmark(results: list[StageTiming], save_path: Path) -> Path:
    """Bar panels: extraction ×real-time, training seconds, inference ×real-time.

    All three panels are keyed on the *project*, so they read across as one row.
    Training is timed per behavior, which used to put ~36 unrotated tick labels in
    the middle panel (illegible, and not comparable to its neighbours); it is now
    summarised per project as a bar of the mean fit time with every behavior's
    individual time overlaid as a dot, so the spread is still visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    ok = [r for r in results if not r.error and np.isfinite(r.seconds)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    def _short(pid: str) -> str:
        return pid if len(pid) <= 18 else pid[:17] + "…"

    def _bars(ax, rows, value_fn, ylabel, title, color, fmt="{:.2f}"):
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.axis("off"); ax.set_title(title); return
        labels = [_short(r.project_id) for r in rows]
        vals = [value_fn(r) for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, vals, 0.68, color=color, edgecolor="white", linewidth=0.4)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        # Rotated + right-anchored: project names are long enough to collide at 0°.
        ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.margins(y=0.16)
        ax.grid(axis="y", alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _train_bars(ax, rows, color):
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.axis("off"); ax.set_title("Training (given features)"); return
        by_proj: dict[str, list[float]] = {}
        for r in rows:
            by_proj.setdefault(str(r.project_id), []).append(float(r.seconds))
        projects = sorted(by_proj)
        x = np.arange(len(projects))
        means = [float(np.mean(by_proj[p])) for p in projects]
        ax.bar(x, means, 0.68, color=color, edgecolor="white", linewidth=0.4, zorder=1)
        rng = np.random.default_rng(0)
        for xi, p in zip(x, projects):
            vals = by_proj[p]
            jitter = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(xi + jitter, vals, s=14, color="#1B5E20", alpha=0.65,
                       edgecolor="white", linewidth=0.4, zorder=3)
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.1f}s", ha="center", va="bottom", fontsize=8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(p) for p in projects], fontsize=8, rotation=35,
                           ha="right")
        ax.set_ylabel("seconds (model fit)")
        n_beh = sum(len(v) for v in by_proj.values())
        ax.set_title(f"Training (given features)\nbar = project mean · "
                     f"dot = one behavior ({n_beh} models)", fontsize=10.5)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ext = [r for r in ok if r.stage == STAGE_EXTRACT]
    trn = [r for r in ok if r.stage == STAGE_TRAIN]
    inf = [r for r in ok if r.stage == STAGE_INFER]
    _bars(axes[0], ext, lambda r: r.faster_than_realtime, "× faster than real-time",
          "Feature extraction / session", "#4C72B0", "{:.1f}×")
    _train_bars(axes[1], trn, "#55A868")
    _bars(axes[2], inf, lambda r: r.faster_than_realtime, "× faster than real-time",
          "Dense inference / video", "#C44E52", "{:.1f}×")
    fig.suptitle("ABEL pipeline throughput (higher = faster)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--projects", nargs="+", required=True,
                    help="One or more ABEL project roots.")
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                    help="Which stages to benchmark (default: all).")
    ap.add_argument("--behaviors", nargs="+", default=None,
                    help="Behavior names to time training for (default: all in project).")
    ap.add_argument("--out", default="./benchmark_out")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(
        args.projects, args.stages, args.behaviors,
        log=lambda m: print(m, flush=True))

    df = results_to_frame(results)
    csv_path = out_dir / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    png_path = plot_benchmark(results, out_dir / "benchmark.png")
    prism.write_all(out_dir, bench_df=df)

    print("\n=== Pipeline throughput ===", flush=True)
    cols = ["project_id", "stage", "detail", "seconds", "video_seconds",
            "faster_than_realtime", "units", "error"]
    show = df[[c for c in cols if c in df.columns]] if not df.empty else df
    print(show.to_string(index=False) if not show.empty else "(no results)", flush=True)
    print(f"\nCSV : {csv_path}\nPlot: {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
