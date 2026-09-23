"""Cancel / Stop actually stops extraction, active learning and dense inference.

Regression: the buttons only set a flag that was read between whole sessions
(or behaviors), and broad ``except Exception`` handlers swallowed the cancel,
so a run kept going for minutes to hours after the user pressed Cancel.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abel.utils.cancellation import (
    OperationCancelled,
    cancel_scope,
    cancellable,
    check_cancel,
    checkpoint,
    is_cancel_traceback,
    propagate_scope,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def test_cancel_escapes_broad_exception_handlers():
    flag = [True]
    with pytest.raises(OperationCancelled):
        try:
            check_cancel(flag)
        except Exception:  # an "optional step failed, carry on" handler
            pytest.fail("cancel was swallowed as an ordinary error")


def test_cancellable_progress_is_a_checkpoint():
    seen: list[str] = []
    flag = [False]
    cb = cancellable(seen.append, flag)
    cb("a")
    flag[0] = True
    with pytest.raises(OperationCancelled):
        cb("b")
    assert seen == ["a"]
    # A None callback still checks the flag.
    with pytest.raises(OperationCancelled):
        cancellable(None, flag)("x")


def test_checkpoint_is_noop_outside_scope_and_raises_inside():
    checkpoint()
    flag = [True]
    with cancel_scope(flag):
        with pytest.raises(OperationCancelled):
            checkpoint()
    checkpoint()  # scope is unbound again


def test_scope_reaches_pool_threads_only_when_propagated():
    flag = [True]

    def _work() -> str:
        checkpoint()
        return "ran"

    with cancel_scope(flag), cf.ThreadPoolExecutor(1) as ex:
        assert ex.submit(_work).result() == "ran"
        with pytest.raises(OperationCancelled):
            ex.submit(propagate_scope(_work)).result()


def test_subclass_cancels_are_recognized_in_tracebacks():
    from abel.services.feature_prep_service import PrepCancelledError
    from abel.ui.tabs.active_learning_tab import PipelineCancelledError

    for exc in (OperationCancelled(), PrepCancelledError("PREP_CANCELLED_BY_USER"),
                PipelineCancelledError()):
        assert isinstance(exc, OperationCancelled)
        assert is_cancel_traceback(f"Traceback ...\n{type(exc).__name__}: {exc}")
    assert not is_cancel_traceback("ValueError: bad input")


def test_task_worker_reports_cancel_through_failed():
    from PySide6.QtCore import QCoreApplication

    from abel.workers.task_worker import TaskWorker

    _app = QCoreApplication.instance() or QCoreApplication([])
    got: list[str] = []

    def _job():
        raise OperationCancelled()

    worker = TaskWorker(_job)
    worker.signals.failed.connect(got.append)
    worker.run()
    assert len(got) == 1 and is_cancel_traceback(got[0])


# ── Feature extraction ───────────────────────────────────────────────────


def _noisy_clip(tmp_path: Path, n: int = 200, w: int = 64, h: int = 64) -> Path:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    path = tmp_path / "noise.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for _ in range(n):
        frame = np.clip(128 + rng.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def test_video_chunk_stops_mid_chunk(tmp_path):
    from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService

    n = 200
    clip = _noisy_clip(tmp_path, n=n)
    xs = np.full(n, 32.0)
    zeros = np.zeros(n)
    calls = [0]

    def _check():
        calls[0] += 1
        if calls[0] >= 2:
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        ContextFeatureService._process_video_chunk(
            video_path=clip, frame_start=0, frame_end=n,
            body_x=xs, body_y=xs, paw_l_x=zeros, paw_l_y=zeros, paw_r_x=zeros, paw_r_y=zeros,
            nose_x=xs, nose_y=xs, target_roi={}, has_target=False, local_radius=12,
            config=ContextFeatureConfig(downsample_factor=1, prefer_gpu=False),
            cancel_check=_check,
        )
    # Checked every 64 frames: the second check is frame 64 of 200.
    assert calls[0] == 2


def test_prep_cancel_stops_running_session_and_skips_the_rest(tmp_path, monkeypatch):
    from abel.services import feature_prep_service as fps

    flag = [False]
    started: list[str] = []

    class _FakeContext:
        def compute_frame_context(self, *, session_id, cancel_check=None, **_kw):
            started.append(session_id)
            for chunk in range(100):
                if chunk == 3:
                    flag[0] = True  # user presses Cancel mid-session
                cancel_check()
            pytest.fail("context extraction ignored the cancel")

    monkeypatch.setattr(fps, "ContextFeatureService", _FakeContext)
    monkeypatch.setattr(
        fps, "plan_session_workers",
        lambda n, **_kw: fps.WorkerPlan(max_workers=1, intra_session_workers=1, source="test"),
    )
    jobs = [
        fps.SessionJob(f"s{i}", "M1", tmp_path / "p.csv", tmp_path / "v.mp4", 30.0)
        for i in range(3)
    ]

    def _check():
        if flag[0]:
            raise fps.PrepCancelledError("PREP_CANCELLED_BY_USER")

    class _Obs:
        def stage_start(self, *a): ...
        def stage_done(self, *a): ...
        def stage_skip(self, *a): ...
        def stage_advance(self, *a): ...
        def log(self, *a): ...

    result = fps.PrepResult()
    with pytest.raises(fps.PrepCancelledError):
        fps.FeaturePrepService()._extract_sessions(
            tmp_path, jobs, fps.PrepConfig(use_video_features=True), {}, _Obs(),
            result, _check, cached_pose={"s0", "s1", "s2"},
        )
    assert started == ["s0"]
    assert result.n_sessions_processed == 0


def test_canceled_kinematics_write_no_partial_npz(tmp_path, monkeypatch):
    from abel.models.schemas import PoseFeaturePreset
    from abel.services.pose_features_service import PoseFeatureConfig, PoseFeaturesService
    from abel.services.pose_processing_service import PoseData

    n = 300
    parts = ["nose", "tail_base"]
    xs = pd.DataFrame({"nose": np.linspace(0, 50, n), "tail_base": np.linspace(0, 40, n)})
    lk = pd.DataFrame({"nose": np.ones(n), "tail_base": np.ones(n)})
    pose = PoseData(parts, xs, xs.copy(), lk, xs["nose"].to_numpy(), xs["nose"].to_numpy(), n)

    svc = PoseFeaturesService()
    svc.set_project(tmp_path)
    monkeypatch.setattr(svc._pose_service, "load", lambda *a, **k: pose)
    monkeypatch.setattr(svc._pose_service, "clean_pose", lambda p, **k: p)
    monkeypatch.setattr(svc, "_pixels_per_mm_for_session", lambda sid: None)

    flag = [False]

    def _progress(done, total):
        if done == 3:
            flag[0] = True

    cfg = PoseFeatureConfig("s1", tmp_path / "p.csv", PoseFeaturePreset(preset_id="p", name="p"))
    result = svc.extract_features(cfg, progress_callback=_progress, cancel_flag=flag)

    assert not result.success
    assert "Canceled by user." in result.warnings
    assert not (tmp_path / "derived" / "pose_features" / "s1.npz").exists()


# ── Active learning ──────────────────────────────────────────────────────


def test_active_learning_progress_relay_raises_once_stopped():
    from abel.ui.tabs.active_learning_tab import ActiveLearningTab, PipelineCancelledError

    emitted: list[tuple] = []

    class _Sig:
        def emit(self, *a):
            emitted.append(a)

    class _Tab:
        _cancel_flag = [False]
        _pipeline_progress_updated = _Sig()

    tab = _Tab()
    ActiveLearningTab._emit_pipeline_progress(tab, 1, 10, "log", "status")
    tab._cancel_flag[0] = True
    with pytest.raises(PipelineCancelledError):
        ActiveLearningTab._emit_pipeline_progress(tab, 2, 10, "log", "status")
    assert emitted == [(1, 10, "log", "status")]

    ran: list[bool] = []

    def _task():
        ran.append(True)
        checkpoint()

    with pytest.raises(PipelineCancelledError):
        ActiveLearningTab._cancel_scoped(tab, _task)()
    assert ran == [True]


# ── Dense inference ──────────────────────────────────────────────────────


def test_dense_inference_task_is_cancellable():
    from abel.ui.tabs.temporal_refinement_tab import TemporalRefinementTab

    reached: list[str] = []

    class _Manager:
        def run_temporal_refinement_inference(self, *, progress_cb, **_kw):
            progress_cb("Inference session 1/3: s1")
            reached.append("s1")
            checkpoint()  # e.g. between model scoring batches
            reached.append("after-checkpoint")
            return {"status": "ok"}

    class _Tab:
        _cancel_flag = [False]

        def _require_manager(self):
            return _Manager()

    tab = _Tab()
    out = TemporalRefinementTab._infer_task(tab, progress_cb=lambda m: None, concept_id="c", config=object())
    assert out == {"status": "ok"}

    reached.clear()
    tab._cancel_flag[0] = True
    with pytest.raises(OperationCancelled):
        TemporalRefinementTab._infer_task(tab, progress_cb=lambda m: None, concept_id="c", config=object())
    assert reached == []
