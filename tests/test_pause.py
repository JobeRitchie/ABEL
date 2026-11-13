"""Pause / Resume for feature extraction and active learning.

A pause blocks at the same stop points Cancel / Stop uses, so a paused job
sits at a safe place.  Resuming must give the same result as never pausing,
and stopping a paused job must leave the project ready for a fresh run.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from abel.utils.cancellation import (
    JobFlag,
    OperationCancelled,
    cancel_scope,
    cancellable,
    check_cancel,
    checkpoint,
)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _run_in_thread(fn):
    out: dict = {}

    def _target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            out["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, out


# ── The flag ─────────────────────────────────────────────────────────────


def test_job_flag_is_still_a_plain_cancel_flag():
    flag = JobFlag()
    assert flag == [False] and not flag[0]
    check_cancel(flag)  # not paused, not cancelled: returns at once
    flag[0] = True
    with pytest.raises(OperationCancelled):
        check_cancel(flag)


@pytest.mark.parametrize("kind", ["check_cancel", "cancellable", "checkpoint"])
def test_every_stop_point_blocks_while_paused_and_continues_on_resume(kind):
    flag = JobFlag()
    seen: list[str] = []

    def _job():
        if kind == "check_cancel":
            check_cancel(flag)
        elif kind == "cancellable":
            cancellable(seen.append, flag)("progress")
        else:
            with cancel_scope(flag):
                checkpoint()
        return "done"

    flag.pause()
    thread, out = _run_in_thread(_job)
    assert _wait_for(lambda: flag.waiting == 1)
    time.sleep(0.25)
    assert thread.is_alive() and not out and not seen  # really paused

    flag.resume()
    thread.join(5)
    assert out == {"value": "done"}
    assert flag.waiting == 0
    if kind == "cancellable":
        assert seen == ["progress"]


def test_stop_while_paused_unwinds_quickly_with_the_callers_cancel_type():
    class _MyCancel(OperationCancelled):
        pass

    flag = JobFlag()
    flag.pause()
    thread, out = _run_in_thread(lambda: check_cancel(flag, _MyCancel))
    assert _wait_for(lambda: flag.waiting == 1)

    t0 = time.monotonic()
    flag[0] = True  # Stop pressed; nobody presses Resume
    thread.join(5)
    assert time.monotonic() - t0 < 1.0
    assert isinstance(out.get("error"), _MyCancel)
    assert flag.waiting == 0


def test_clock_leaves_paused_time_out():
    flag = JobFlag()
    t0 = flag.clock()
    flag.pause()
    frozen = flag.clock()
    time.sleep(0.3)
    assert flag.clock() == frozen
    flag.resume()
    assert flag.clock() - t0 < 0.2


def test_pause_listeners_follow_state_and_get_the_current_state_on_add():
    flag = JobFlag()
    states: list[bool] = []
    flag.pause()
    flag.add_pause_listener(states.append)
    flag.resume()
    flag.pause()
    flag.remove_pause_listener(states.append)
    flag.resume()
    assert states == [True, False, True]


# ── Feature extraction ───────────────────────────────────────────────────


def _kinematics_service(tmp_path, monkeypatch):
    from abel.services.pose_features_service import PoseFeaturesService
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
    return svc


def _npz_arrays(path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def test_paused_then_resumed_kinematics_match_an_unpaused_run(tmp_path, monkeypatch):
    from abel.models.schemas import PoseFeaturePreset
    from abel.services.pose_features_service import PoseFeatureConfig

    preset = PoseFeaturePreset(preset_id="p", name="p")
    plain_root, paused_root = tmp_path / "plain", tmp_path / "paused"
    plain_root.mkdir()
    paused_root.mkdir()

    plain = _kinematics_service(plain_root, monkeypatch)
    assert plain.extract_features(PoseFeatureConfig("s1", plain_root / "p.csv", preset)).success

    flag = JobFlag()
    svc = _kinematics_service(paused_root, monkeypatch)

    def _progress(done, total):
        if done == 3:
            flag.pause()  # user presses Pause mid-session

    thread, out = _run_in_thread(lambda: svc.extract_features(
        PoseFeatureConfig("s1", paused_root / "p.csv", preset),
        progress_callback=_progress, cancel_flag=flag,
    ))
    assert _wait_for(lambda: flag.waiting == 1)
    npz = paused_root / "derived" / "pose_features" / "s1.npz"
    assert not npz.exists()  # nothing half-written while paused

    flag.resume()
    thread.join(10)
    assert out["value"].success

    a = _npz_arrays(plain_root / "derived" / "pose_features" / "s1.npz")
    b = _npz_arrays(npz)
    assert a.keys() == b.keys()
    for key in a:
        if a[key].dtype.kind in "fiub":
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_stop_after_pause_leaves_no_partial_file_and_a_new_run_completes(tmp_path, monkeypatch):
    from abel.models.schemas import PoseFeaturePreset
    from abel.services.pose_features_service import PoseFeatureConfig

    svc = _kinematics_service(tmp_path, monkeypatch)
    cfg = PoseFeatureConfig("s1", tmp_path / "p.csv", PoseFeaturePreset(preset_id="p", name="p"))
    flag = JobFlag()

    def _progress(done, total):
        if done == 3:
            flag.pause()

    thread, out = _run_in_thread(
        lambda: svc.extract_features(cfg, progress_callback=_progress, cancel_flag=flag)
    )
    assert _wait_for(lambda: flag.waiting == 1)
    flag[0] = True  # Cancel pressed while paused
    thread.join(10)

    npz = tmp_path / "derived" / "pose_features" / "s1.npz"
    assert not out["value"].success
    assert "Canceled by user." in out["value"].warnings
    assert not npz.exists()

    # What the tab does when the next run starts.
    flag.resume()
    flag[0] = False
    assert svc.extract_features(cfg, cancel_flag=flag).success
    assert npz.exists()


def test_prep_stop_point_pauses_and_then_stops(tmp_path, monkeypatch):
    """The prep stage's own cancel check is a pause point, and Stop wins over it."""
    from abel.services import feature_prep_service as fps

    flag = JobFlag()
    reached: list[int] = []

    class _FakeContext:
        def compute_frame_context(self, *, session_id, cancel_check=None, **_kw):
            assert cancel_check.flag is flag  # how chunk processes find the pause
            for chunk in range(100):
                if chunk == 3:
                    flag.pause()
                cancel_check()
                reached.append(chunk)

    monkeypatch.setattr(fps, "ContextFeatureService", _FakeContext)
    monkeypatch.setattr(fps.FeaturePrepService, "_preflight_inputs", lambda *a, **k: None)
    monkeypatch.setattr(fps.FeaturePrepService, "keypoint_aliases", lambda *a, **k: {})
    monkeypatch.setattr(fps.FeaturePrepService, "_pose_changed", lambda *a, **k: False)
    monkeypatch.setattr(fps.FeaturePrepService, "_context_changed", lambda *a, **k: False)
    monkeypatch.setattr(fps.FeaturePrepService, "cached_pose_sessions", lambda *a, **k: {"s0"})
    monkeypatch.setattr(fps.FeaturePrepService, "cached_context_sessions", lambda *a, **k: set())
    monkeypatch.setattr(fps.FeaturePrepService, "_worth_chunk_processes", lambda *a, **k: False)
    monkeypatch.setattr(
        fps, "plan_session_workers",
        lambda n, **_kw: fps.WorkerPlan(max_workers=1, intra_session_workers=1, source="test"),
    )
    jobs = [fps.SessionJob("s0", "M1", tmp_path / "p.csv", tmp_path / "v.mp4", 30.0)]

    thread, out = _run_in_thread(lambda: fps.FeaturePrepService().prepare(
        tmp_path, jobs, fps.PrepConfig(use_video_features=True), cancel_flag=flag,
    ))
    assert _wait_for(lambda: flag.waiting >= 1, timeout=60), out  # GPU probe can be slow
    time.sleep(0.25)
    assert reached == [0, 1, 2]  # stopped advancing at the pause

    flag[0] = True
    thread.join(10)
    assert isinstance(out.get("error"), fps.PrepCancelledError)
    assert reached == [0, 1, 2]


def test_chunk_worker_waits_while_paused_and_a_cancel_ends_the_wait(monkeypatch):
    from abel.services import context_chunk_pool as pool

    pause, cancel = threading.Event(), threading.Event()
    monkeypatch.setattr(pool, "_worker_pause", pause)
    monkeypatch.setattr(pool, "_worker_cancel", cancel)

    pool._worker_cancel_check()  # neither set: returns

    pause.set()
    thread, out = _run_in_thread(pool._worker_cancel_check)
    time.sleep(0.3)
    assert thread.is_alive()
    pause.clear()
    thread.join(5)
    assert out == {"value": None}

    pause.set()
    thread, out = _run_in_thread(pool._worker_cancel_check)
    time.sleep(0.2)
    cancel.set()
    thread.join(5)
    assert isinstance(out.get("error"), OperationCancelled)


def test_run_chunks_passes_the_pause_on_to_the_worker_processes(monkeypatch):
    from abel.services import context_chunk_pool as pool

    flag = JobFlag()
    pause_event = threading.Event()
    seen: list[bool] = []

    def _fake_run(*_a, **_k):
        seen.append(pause_event.is_set())
        flag.pause()
        seen.append(pause_event.is_set())
        flag.resume()
        seen.append(pause_event.is_set())
        return []

    monkeypatch.setattr(pool, "_get_pool", lambda use_gpu: (object(), threading.Event(), pause_event))
    monkeypatch.setattr(pool, "_run_chunks_in_pool", _fake_run)

    def _check() -> None: ...
    _check.flag = flag

    assert pool.run_chunks("cpu", {}, [], cancel_check=_check) == []
    assert seen == [False, True, False]
    flag.pause()  # the listener is gone once the session's chunks are done
    assert not pause_event.is_set()
    flag.resume()


# ── Active learning ──────────────────────────────────────────────────────


def test_active_learning_progress_relay_pauses_then_stops():
    from abel.ui.tabs.active_learning_tab import ActiveLearningTab, PipelineCancelledError

    emitted: list[tuple] = []

    class _Sig:
        def emit(self, *a):
            emitted.append(a)

    class _Tab:
        _cancel_flag = JobFlag()
        _pipeline_progress_updated = _Sig()

    tab = _Tab()
    tab._cancel_flag.pause()
    thread, out = _run_in_thread(
        lambda: ActiveLearningTab._emit_pipeline_progress(tab, 1, 10, "log", "status")
    )
    assert _wait_for(lambda: tab._cancel_flag.waiting == 1)
    assert emitted == []
    tab._cancel_flag.resume()
    thread.join(5)
    assert emitted == [(1, 10, "log", "status")]

    tab._cancel_flag.pause()
    thread, out = _run_in_thread(
        lambda: ActiveLearningTab._emit_pipeline_progress(tab, 2, 10, "log", "status")
    )
    assert _wait_for(lambda: tab._cancel_flag.waiting == 1)
    tab._cancel_flag[0] = True
    thread.join(5)
    assert isinstance(out.get("error"), PipelineCancelledError)
    assert len(emitted) == 1


# ── The button ───────────────────────────────────────────────────────────


def test_pause_button_toggles_the_flag_and_resets_between_jobs():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from abel.ui.widgets.pause_button import PauseButton

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    flag = JobFlag()
    button = PauseButton(flag)
    events: list[str] = []
    button.pause_requested.connect(lambda: events.append("requested"))
    button.pause_reached.connect(lambda: events.append("reached"))
    button.resumed.connect(lambda: events.append("resumed"))

    assert not button.isEnabled()
    button.set_job_running(True)
    button.click()
    assert flag.paused and "Resume" in button.text()

    thread, _out = _run_in_thread(lambda: check_cancel(flag))
    assert _wait_for(lambda: (app.processEvents(), "reached" in events)[1])

    button.click()
    thread.join(5)
    assert not flag.paused and "Pause" in button.text()
    assert events == ["requested", "reached", "resumed"]

    button.click()
    button.set_job_running(False)  # job ended (or was stopped) while paused
    assert not flag.paused and not button.isEnabled() and "Pause" in button.text()
