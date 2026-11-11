"""The faster extraction paths must give the same numbers as the paths they replace.

* R3D clips are streamed to the model while the video is still decoding
  (``_stream_clips`` / ``_decode_and_embed``) instead of being held in memory.
* Context frame chunks can run in worker processes (``context_chunk_pool``)
  instead of threads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from abel.services import context_chunk_pool
from abel.services import r3d_feature_service as r3d
from abel.services.context_feature_service import ContextFeatureConfig, ContextFeatureService
from abel.services.r3d_feature_service import N_SAMPLE_FRAMES, R3DFeatureService
from abel.utils.cancellation import OperationCancelled

W, H, N_FRAMES = 96, 72, 150


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vid") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        frame = rng.integers(0, 60, size=(H, W, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + i % 50, 20 + (i * 3) % 30), 8, (255, 200, 40), -1)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


def _tracks(n_animals: int = 1):
    out = []
    for a in range(n_animals):
        cx = 30.0 + 10 * a + (np.arange(N_FRAMES) % 40)
        cy = 25.0 + 5 * a + (np.arange(N_FRAMES) % 20)
        out.append((cx, cy, 48))
    return out


def _reference_clips(video, starts, ends, tracks, animal_row):
    """The original algorithm: one buffer for every segment, filled in a pass."""
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sides = [int(np.clip(s, r3d._CROP_MIN_PX, max(1, min(width, height)))) for _, _, s in tracks]
    n_pose = min(int(min(len(x), len(y))) for x, y, _ in tracks)
    limit = max(0, min(n_video, n_pose) - 1)
    n = len(starts)
    idx = np.empty((n, N_SAMPLE_FRAMES), dtype=np.int64)
    for i in range(n):
        total = max(1, int(ends[i]) - int(starts[i]) + 1)
        idx[i] = int(starts[i]) + np.linspace(0, total - 1, num=N_SAMPLE_FRAMES, dtype=int)
    idx = np.clip(idx, 0, limit)
    clips = np.zeros((n, N_SAMPLE_FRAMES, 112, 112, 3), dtype=np.uint8)
    filled = np.zeros(n, dtype=int)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    for i in range(n):
        for s in range(N_SAMPLE_FRAMES):
            f = int(idx[i, s])
            if f >= len(frames):
                continue
            cx, cy, _ = tracks[int(animal_row[i])]
            side = sides[int(animal_row[i])]
            x0 = int(np.clip(cx[f] - side / 2.0, 0, max(0, width - side)))
            y0 = int(np.clip(cy[f] - side / 2.0, 0, max(0, height - side)))
            crop = frames[f][y0 : y0 + side, x0 : x0 + side]
            clips[i, s] = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA
            )
            filled[i] += 1
    kept = [i for i in range(n) if filled[i] >= N_SAMPLE_FRAMES // 2]
    return clips[kept], kept


@pytest.mark.parametrize(
    "starts, window, n_animals",
    [
        (np.arange(0, 135, 15), 15, 1),          # back-to-back windows
        (np.arange(0, 130, 4), 15, 1),           # heavily overlapping windows
        (np.array([90, 0, 45, 30, 120]), 15, 1),  # not in start order
        (np.tile(np.arange(0, 120, 15), 2), 15, 2),  # animal A's windows, then animal B's
    ],
)
def test_streamed_clips_match_the_all_in_memory_decode(video, starts, window, n_animals):
    ends = starts + window - 1
    tracks = _tracks(n_animals)
    animal_row = (np.arange(len(starts)) >= len(starts) // 2).astype(int) if n_animals == 2 \
        else np.zeros(len(starts), dtype=int)
    want, want_kept = _reference_clips(video, starts, ends, tracks, animal_row)

    got = list(R3DFeatureService._stream_clips(video, starts, ends, tracks, animal_row))
    assert [i for i, _ in got] == want_kept == sorted(want_kept)
    assert np.array_equal(np.stack([c for _, c in got]), want)

    clips, kept = R3DFeatureService._decode_clips(video, starts, ends, tracks, animal_row)
    assert kept == want_kept and np.array_equal(clips, want)


def test_streaming_handles_no_segments(video):
    empty = np.zeros(0, dtype=int)
    assert list(R3DFeatureService._stream_clips(video, empty, empty, _tracks(), empty)) == []
    clips, kept = R3DFeatureService._decode_clips(video, empty, empty, _tracks(), empty)
    assert clips.shape == (0, N_SAMPLE_FRAMES, 112, 112, 3) and kept == []


def test_decode_and_embed_feeds_the_model_the_same_batches(video, monkeypatch):
    monkeypatch.setattr(r3d, "_BATCH", 4)
    seen: list[int] = []

    def fake_embed(clips: np.ndarray) -> np.ndarray:
        seen.append(len(clips))
        return clips.reshape(len(clips), -1).astype(np.float32)[:, : r3d.R3D_DIMS]

    monkeypatch.setattr(R3DFeatureService, "_embed", staticmethod(fake_embed))
    starts = np.arange(0, 132, 12)          # 11 segments: batches of 4, 4, 3
    ends = starts + 14
    row = np.zeros(len(starts), dtype=int)
    emb, kept = R3DFeatureService()._decode_and_embed(video, starts, ends, _tracks(), row)

    want, want_kept = _reference_clips(video, starts, ends, _tracks(), row)
    assert kept == want_kept
    assert seen == [4, 4, 3]
    assert np.array_equal(emb, fake_embed(want))


def test_decode_and_embed_surfaces_a_model_failure(video, monkeypatch):
    monkeypatch.setattr(r3d, "_BATCH", 2)

    def boom(clips):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(R3DFeatureService, "_embed", staticmethod(boom))
    starts = np.arange(0, 120, 10)
    with pytest.raises(RuntimeError, match="forward failed"):
        R3DFeatureService()._decode_and_embed(
            video, starts, starts + 14, _tracks(), np.zeros(len(starts), dtype=int)
        )


# ── Context chunks: worker processes vs in-process ───────────────────────────


def _chunk_kwargs(video: Path) -> dict:
    t = np.arange(N_FRAMES, dtype=float)
    x = 30.0 + (t % 40)
    y = 25.0 + (t % 20)
    return dict(
        video_path=video, body_x=x, body_y=y,
        paw_l_x=x - 4, paw_l_y=y + 3, paw_r_x=x + 4, paw_r_y=y + 3,
        nose_x=x + 8, nose_y=y - 2,
        target_roi={}, has_target=False, local_radius=16,
        config=ContextFeatureConfig(flow_temporal_stride=5, downsample_factor=1),
        extra_rois=[],
    )


BOUNDARIES = [(0, 50), (50, 100), (100, N_FRAMES)]


@pytest.fixture
def two_workers(monkeypatch):
    monkeypatch.setenv("ABEL_CONTEXT_PROCESSES", "2")
    monkeypatch.setattr(context_chunk_pool, "_disabled_reason", None)
    yield
    context_chunk_pool.shutdown(cancel=True)


def test_setting_zero_processes_keeps_the_thread_path(video, monkeypatch):
    monkeypatch.setenv("ABEL_CONTEXT_PROCESSES", "0")
    assert context_chunk_pool.planned_workers(use_gpu=False) == 0
    assert context_chunk_pool.run_chunks("cpu", _chunk_kwargs(video), BOUNDARIES) is None


def test_worker_processes_return_the_same_chunks_in_order(video, two_workers):
    kwargs = _chunk_kwargs(video)
    want = [
        ContextFeatureService._process_video_chunk(frame_start=s, frame_end=e, **kwargs)
        for s, e in BOUNDARIES
    ]
    progress: list[tuple[int, int]] = []
    got = context_chunk_pool.run_chunks(
        "cpu", kwargs, BOUNDARIES, progress_cb=lambda d, n, m: progress.append((d, n)),
    )
    assert got is not None and len(got) == len(want)
    for a, b in zip(got, want):
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(np.asarray(a[key], float), np.asarray(b[key], float), equal_nan=True), key
    assert sorted(progress) == [(1, 3), (2, 3), (3, 3)]


def test_a_cancel_reaches_the_caller_with_its_own_type(video, two_workers):
    class StopNow(OperationCancelled):
        pass

    def cancel_check() -> None:
        raise StopNow("PREP_CANCELLED_BY_USER")

    with pytest.raises(StopNow):
        context_chunk_pool.run_chunks("cpu", _chunk_kwargs(video), BOUNDARIES, cancel_check=cancel_check)
    assert context_chunk_pool._pool is None  # workers were told to stop


def test_a_chunk_error_propagates_and_the_pool_survives(video, two_workers, tmp_path):
    bad = dict(_chunk_kwargs(video), video_path=tmp_path / "missing.avi")
    with pytest.raises(ValueError, match="Unable to open video"):
        context_chunk_pool.run_chunks("cpu", bad, BOUNDARIES)
    assert context_chunk_pool._pool is not None
    assert context_chunk_pool.run_chunks("cpu", _chunk_kwargs(video), BOUNDARIES) is not None


# ── Consolidation is skipped when nothing changed ─────────────────────────────
def _fake_consolidate(calls):
    def _run(project_root):
        calls.append(1)
        mono = project_root / "derived" / "pose_features" / "frame_pose.parquet"
        parts = sorted((mono.parent / "sessions").glob("*.parquet"))
        mono.write_bytes(b"".join(p.read_bytes() for p in parts))
        return mono
    return _run


def test_consolidation_runs_once_then_only_when_a_file_changes(tmp_path):
    import os

    from abel.services.feature_prep_service import FeaturePrepService as F

    sessions = tmp_path / "derived" / "pose_features" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "a.parquet").write_bytes(b"aaaa")
    (sessions / "b.parquet").write_bytes(b"bb")
    mono = sessions.parent / "frame_pose.parquet"
    calls: list[int] = []
    run = _fake_consolidate(calls)

    assert F._consolidate_if_changed(mono, run, tmp_path) is True
    before = (mono.read_bytes(), mono.stat().st_mtime_ns)
    assert F._consolidate_if_changed(mono, run, tmp_path) is False
    assert (mono.read_bytes(), mono.stat().st_mtime_ns) == before
    assert len(calls) == 1

    # A re-extracted session (new content, new timestamp) forces a merge.
    (sessions / "b.parquet").write_bytes(b"BBB")
    assert F._consolidate_if_changed(mono, run, tmp_path) is True
    assert mono.read_bytes() == b"aaaaBBB"

    # A new session forces a merge.
    (sessions / "c.parquet").write_bytes(b"c")
    assert F._consolidate_if_changed(mono, run, tmp_path) is True

    # So does anything else rewriting the merged file (e.g. a session removal).
    st = mono.stat()
    os.utime(mono, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert F._consolidate_if_changed(mono, run, tmp_path) is True

    # A deleted session file forces a merge too.
    (sessions / "a.parquet").unlink()
    assert F._consolidate_if_changed(mono, run, tmp_path) is True
    assert F._consolidate_if_changed(mono, run, tmp_path) is False
    assert len(calls) == 5


def test_consolidation_runs_when_the_merged_file_is_missing(tmp_path):
    from abel.services.feature_prep_service import FeaturePrepService as F

    sessions = tmp_path / "derived" / "pose_features" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "a.parquet").write_bytes(b"a")
    mono = sessions.parent / "frame_pose.parquet"
    calls: list[int] = []
    run = _fake_consolidate(calls)
    F._consolidate_if_changed(mono, run, tmp_path)
    mono.unlink()
    assert F._consolidate_if_changed(mono, run, tmp_path) is True
    assert mono.exists()
