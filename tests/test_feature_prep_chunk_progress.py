"""Feature prep reports each finished video chunk, not just finished sessions.

Regression: the Features tab ran context extraction without a progress
callback and dropped every stage message, so a long video session showed
nothing between "Preprocessing N session(s)" and the end of the stage.
"""

from __future__ import annotations

from abel.services import feature_prep_service as fps
from abel.services.feature_prep_service import (
    STAGE_PREPROCESS,
    FeaturePrepService,
    PrepConfig,
    PrepResult,
    SessionJob,
)


class _Recorder:
    def __init__(self) -> None:
        self.advances: list[tuple[str, float, str]] = []
        self.logs: list[str] = []

    def stage_start(self, key, label, total_units): ...
    def stage_done(self, key): ...
    def stage_skip(self, key, message): ...

    def stage_advance(self, key, done_units, message):
        self.advances.append((key, done_units, message))

    def log(self, message):
        self.logs.append(message)


class _FakeContext:
    def compute_frame_context(self, *, progress_cb=None, **_kw):
        progress_cb(0, 1, "Detected 1920x1080 — downsampling 2x")
        for i in range(4):
            progress_cb(0, 4, f"chunk {i + 1}/4: starting frames …")
            progress_cb(i + 1, 4, f"chunk {i + 1}/4 done (frames …)")


def test_chunk_completions_reach_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(fps, "ContextFeatureService", _FakeContext)
    monkeypatch.setattr(
        fps, "plan_session_workers",
        lambda n, **_kw: fps.WorkerPlan(max_workers=1, intra_session_workers=1, source="test"),
    )
    svc = FeaturePrepService()
    job = SessionJob("session_a", "M1", tmp_path / "p.csv", tmp_path / "v.mp4", 30.0)
    obs = _Recorder()

    svc._extract_sessions(
        tmp_path, [job], PrepConfig(use_video_features=True), {}, obs,
        PrepResult(), lambda: None, cached_pose={"session_a"},
    )

    chunk_msgs = [m for _k, _d, m in obs.advances if "chunk" in m]
    assert len(chunk_msgs) == 4 and all("done" in m for m in chunk_msgs)
    # Sub-session progress is fractional and never exceeds one session.
    fracs = [d for _k, d, m in obs.advances if "chunk" in m]
    assert fracs == [0.25, 0.5, 0.75, 1.0]
    assert obs.advances[-1][:2] == (STAGE_PREPROCESS, 1.0)
    assert "Processed session 1/1" in obs.advances[-1][2]
    # "starting" chatter is dropped; one-off setup notes are logged.
    assert not any("starting" in m for _k, _d, m in obs.advances)
    assert any("downsampling" in m for m in obs.logs)
