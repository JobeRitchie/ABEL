"""ETA and progress reporting for a "run saved model" pass.

Three defects this locks in, all observed on a real 3-chamber-social run:

* The ETA came from ``elapsed / steps_done``. On the first event that read the
  whole run as 80 ms; the steps are wildly unequal (scoring every segment dwarfs
  candidate selection) so the average never converged either.
* Any stage that emitted several progress messages was mis-measured: the
  estimator refreshed its stage-entry timestamp on every repeat call, so a
  four-minute stage was booked as the gap since its last message.
* Assembling the training set — enrichment plus the R3D backfill, measured at
  148 s for 1292 segments — reported nothing at all, so the UI sat on
  "Built representations for …" with no sign of life.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from abel.models.schemas import ImportManifest, LinkedSession, PoseAsset, VideoAsset
from abel.services.import_service import ImportService
from abel.ui.tabs.active_learning_tab import ActiveLearningTab
from abel.utils.eta_estimator import StageEtaEstimator, blend_whole_run_eta

SESSION = "session_aaa00000"


def _clock():
    t = [0.0]
    return t, (lambda: t[0])


# ── Estimator: long stages must survive their own chatter ────────────────────
def test_repeated_events_in_one_stage_do_not_erase_its_duration():
    # A run-model stage emits many messages ("R3D features: session 7/33…",
    # "[Inference] Fitting ensemble model 2/3…") while it works. Every one of
    # those used to reset the stage's entry timestamp, so a 100 s stage that
    # chattered every second was learned as 1 s and the ETA collapsed.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=1, stages_per_item=3, clock=clock)

    est.update(0, 0)                     # enter stage 0 at t=0
    for tick in range(1, 100):           # 99 sub-messages, still stage 0
        t[0] = float(tick)
        est.update(0, 0)
    t[0] = 100.0
    est.update(0, 1)                     # stage 0 really took 100 s

    assert est.expected_stage_seconds(0) == pytest.approx(100.0), (
        "a stage's measured duration must span the whole stage, not the gap "
        "since its last progress message"
    )


def test_eta_counts_down_inside_a_long_stage():
    # With the current stage's cost drawn down as it elapses, the reported ETA
    # falls during a multi-minute stage instead of freezing until it ends.
    t, clock = _clock()
    est = StageEtaEstimator(n_items=1, stages_per_item=4, seed_stage_seconds=60.0, clock=clock)

    def eta_now(stage: int) -> float:
        remaining = est.update(0, stage)
        return max(0.0, remaining - min(est.seconds_in_stage(), est.expected_stage_seconds(stage)))

    t[0] = 0.0
    first = eta_now(0)
    t[0] = 30.0
    halfway = eta_now(0)

    assert first == pytest.approx(240.0)          # 4 stages × 60 s seed
    assert halfway == pytest.approx(210.0)        # 30 s of the current stage spent
    assert halfway < first

    # Overrunning the stage floors at the remaining stages, never below zero.
    t[0] = 500.0
    assert eta_now(0) == pytest.approx(180.0)


def test_a_fast_first_stage_does_not_stand_in_for_unmeasured_slow_ones():
    # Stage 0 (loading cached representations) is quick; stages 1+ (scoring every
    # segment) are not. Once stage 0 was measured the estimator used ITS time for
    # every stage it had not seen, so the ETA collapsed from minutes to seconds
    # exactly as the slowest part of the run started.
    t, clock = _clock()
    est = StageEtaEstimator(1, 5, seed_stage_seconds=60.0, clock=clock)
    est.update(0, 0)
    t[0] = 2.0
    est.update(0, 1)                     # stage 0 measured at 2 s

    assert est.expected_stage_seconds(0) == pytest.approx(2.0)
    assert est.expected_stage_seconds(3) == pytest.approx(60.0), (
        "an unmeasured stage must fall back to the prior-run seed, not to the "
        "duration of whichever cheap stage happened to run first"
    )


def test_blend_drops_an_anchor_reality_has_disproved():
    # A stale whole-run total (dataset has grown since it was recorded) can be
    # shorter than the time already spent. Blending it in pushed the projected
    # total below the elapsed time, so the UI read "ETA 0 s" while the run kept
    # working. Once elapsed passes the anchor, only the live estimate counts.
    stale = blend_whole_run_eta(
        hist_total=39.0, elapsed=210.0, live_remaining=23.0, frac=0.4,
        live_calibrated=True,
    )
    assert stale == pytest.approx(23.0)


def test_seconds_in_stage_tracks_the_stage_not_the_last_message():
    t, clock = _clock()
    est = StageEtaEstimator(n_items=1, stages_per_item=2, clock=clock)
    est.update(0, 0)
    t[0] = 40.0
    est.update(0, 0)                     # a mid-stage progress message
    assert est.seconds_in_stage() == pytest.approx(40.0)


# ── Training-set assembly must report while it works ─────────────────────────
@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A project with one reviewed window that is absent from the feature table."""
    service = ImportService()
    manifest = ImportManifest(
        videos=[VideoAsset(asset_id="vid_1", source_path=str(tmp_path / "raw" / "v.mp4"))],
        poses=[PoseAsset(asset_id="pose_1", source_path=str(tmp_path / "raw" / "p.csv"), format="csv")],
        linked_sessions=[
            LinkedSession(session_id=SESSION, video_asset_id="vid_1", pose_asset_id="pose_1")
        ],
    )
    service.save_manifest(tmp_path, manifest)

    pose_dir = tmp_path / "derived" / "pose_features" / "sessions"
    pose_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": np.arange(200),
            "session_id": SESSION,
            "animal_id": "ind0",
            "speed": np.linspace(0.0, 4.0, 200),
        }
    ).to_parquet(pose_dir / f"{SESSION}.parquet", index=False)

    labels_path = tmp_path / "derived" / "review_labels" / "reviewer_labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "segment_id": f"bout_abcd_{SESSION}_100_129",
                "review_label": "Rear",
                "reviewer_id": "human",
                "confidence": 1.0,
            }
        ]
    ).to_parquet(labels_path, index=False)
    return tmp_path


def _tab(project: Path) -> ActiveLearningTab:
    tab = ActiveLearningTab.__new__(ActiveLearningTab)
    tab._project_root = project
    tab._imports = ImportService()
    tab._seeds = SimpleNamespace(seeds=[])
    tab._auto_generate_reviewed_windows = SimpleNamespace(isChecked=lambda: True)
    tab._remap_reviewed_windows = SimpleNamespace(isChecked=lambda: True)
    return tab


def _segment_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": f"{SESSION}_0_29",
                "session_id": SESSION,
                "animal_id": "ind0",
                "start_frame": 0,
                "end_frame": 29,
                "speed_mean": 0.1,
                "speed_std": 0.01,
            }
        ]
    )


def test_training_set_assembly_reports_its_slow_phases(project: Path) -> None:
    seen: list[str] = []
    tab = _tab(project)

    train = tab._build_training_set(_segment_df(), "Rear", progress_cb=seen.append)

    assert not train.empty
    assert any("reviewer labels" in m.lower() for m in seen)
    assert any("computing features" in m.lower() for m in seen), (
        "on-the-fly enrichment is the long silent phase of a run-model pass; "
        f"it must report progress. Saw: {seen}"
    )


def test_training_set_assembly_still_works_without_a_callback(project: Path) -> None:
    # The batch/retrain callers pass no callback; assembly must be unchanged.
    tab = _tab(project)
    assert not tab._build_training_set(_segment_df(), "Rear").empty
