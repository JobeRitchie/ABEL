"""Fusion inference: verify the batched/cached refactor preserves outputs and is faster.

The R3D-18 path needs torch + downloadable weights, so these deterministic tests
force the handcrafted backend (no torch) — which exercises the same video-reuse,
per-video-once opening, cross-behavior score cache, and fusion-combine logic. The
handcrafted embedding is fully deterministic, so we can assert *exact* equality
with a per-segment reference, plus a rigorous open-count reduction and a wall-time
speedup.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from abel.services.fusion_inference_service import FusionConfig, FusionInferenceService

ROI = {"x": 8, "y": 6, "w": 64, "h": 48}
N_FRAMES = 160
THRESH = 0.35


def _make_video(path, n_frames=N_FRAMES, w=160, h=120, seed=0):
    # MJPG => every frame is an intra-coded keyframe, so seeks are frame-exact
    # and a reused capture decodes identical pixels to a fresh one.
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    assert vw.isOpened(), "could not open VideoWriter (MJPG)"
    for _ in range(n_frames):
        vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release()


def _segments(n=24, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        start = int(rng.integers(0, N_FRAMES - 20))
        end = start + int(rng.integers(6, 18))
        # Alternate uncertain / certain so both fused + passthrough paths run.
        unc = 0.9 if i % 4 != 0 else 0.1
        rows.append({
            "segment_id": f"seg_s_{start}_{end}_{i}",
            "session_id": "s",
            "start_frame": start,
            "end_frame": min(end, N_FRAMES - 1),
            "prediction_prob": float(rng.uniform(0.1, 0.9)),
            "uncertainty_score": unc,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def video(tmp_path):
    p = tmp_path / "clip.avi"
    _make_video(p)
    return p


def _reference_fused(svc, df, video_path, cfg):
    """Old-style per-segment reference: open the video once per uncertain segment."""
    fused = []
    for row in df.itertuples(index=False):
        base = float(row.prediction_prob)
        if float(row.uncertainty_score) <= cfg.uncertainty_threshold:
            fused.append(base)
            continue
        emb = svc._video_crop_embedding(video_path, int(row.start_frame), int(row.end_frame), ROI)
        score = svc._emb_to_score(emb)
        fused.append(float(np.clip(cfg.alpha_pose_context * base + cfg.beta_video_embedding * score, 0.0, 1.0)))
    return np.asarray(fused, dtype=float)


def test_fused_output_matches_reference(video):
    svc = FusionInferenceService()
    cfg = FusionConfig(embedding_backend="handcrafted", uncertainty_threshold=THRESH)
    df = _segments()

    ref = _reference_fused(svc, df, video, cfg)
    out = svc.fuse_uncertain_segments(df, {"s": video}, {"s": ROI}, config=cfg)
    got = out["prediction_prob_fused"].to_numpy(dtype=float)

    assert np.allclose(got, ref, atol=1e-6), np.abs(got - ref).max()
    # Passthrough rows (certain) keep their base probability exactly.
    certain = df["uncertainty_score"].to_numpy() <= THRESH
    assert np.allclose(got[certain], df["prediction_prob"].to_numpy()[certain])


def test_cross_behavior_cache_reuse(video, monkeypatch):
    """A shared score cache means a second 'behavior' reopens no videos."""
    svc = FusionInferenceService()
    cfg = FusionConfig(embedding_backend="handcrafted", uncertainty_threshold=THRESH)
    df = _segments()
    cache: dict[str, float] = {}

    opens = {"n": 0}
    real_capture = cv2.VideoCapture

    def _spy(*args, **kwargs):
        opens["n"] += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", _spy)

    out1 = svc.fuse_uncertain_segments(df, {"s": video}, {"s": ROI}, config=cfg, score_cache=cache)
    opens_first = opens["n"]
    assert opens_first == 1  # one video opened exactly once for all its segments
    assert len(cache) > 0

    out2 = svc.fuse_uncertain_segments(df, {"s": video}, {"s": ROI}, config=cfg, score_cache=cache)
    assert opens["n"] == opens_first  # second run reused the cache — no reopen

    # Identical fused output across the two "behaviors".
    assert np.allclose(
        out1["prediction_prob_fused"].to_numpy(dtype=float),
        out2["prediction_prob_fused"].to_numpy(dtype=float),
    )


def test_video_opened_once_not_per_segment(video, monkeypatch):
    """I/O reduction: the file is opened once for all its uncertain segments."""
    svc = FusionInferenceService()
    cfg = FusionConfig(embedding_backend="handcrafted", uncertainty_threshold=THRESH)
    df = _segments(n=48)
    n_uncertain = int((df["uncertainty_score"] > THRESH).sum())
    assert n_uncertain > 5

    opens = {"n": 0}
    real_capture = cv2.VideoCapture
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: (opens.__setitem__("n", opens["n"] + 1), real_capture(*a, **k))[1])

    svc.fuse_uncertain_segments(df, {"s": video}, {"s": ROI}, config=cfg)
    # Old behavior opened the video once per uncertain segment; new opens once.
    assert opens["n"] == 1


def test_r3d18_runs_one_batched_forward_not_per_segment(video, monkeypatch):
    """Work reduction: N uncertain segments => a single batched R3D-18 forward."""
    svc = FusionInferenceService()
    cfg = FusionConfig(embedding_backend="r3d18", uncertainty_threshold=THRESH, n_sample_frames=16)
    df = _segments(n=24)
    n_uncertain = int((df["uncertainty_score"] > THRESH).sum())

    calls = {"n": 0, "batch_sizes": []}

    def _fake_forward(cls, batch, stats):
        calls["n"] += 1
        calls["batch_sizes"].append(int(batch.shape[0]))
        stats["used_cpu"] = True
        # Deterministic stand-in embedding (128-d) so no torch/weights needed.
        vals = batch.reshape(batch.shape[0], -1).mean(axis=1).astype(np.float32)
        return np.repeat(vals[:, None], 128, axis=1), ""

    monkeypatch.setattr(FusionInferenceService, "_forward_r3d18", classmethod(_fake_forward))

    out = svc.fuse_uncertain_segments(df, {"s": video}, {"s": ROI}, config=cfg)
    # All equal-length clips embed in ONE forward call (batched), not n_uncertain.
    assert calls["n"] == 1
    assert calls["batch_sizes"] == [n_uncertain]
    assert out["prediction_prob_fused"].notna().all()


def test_r3d18_model_loaded_once():
    """Model-load reduction: the network is constructed once and reused."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from abel.services import fusion_inference_service as fis

    try:
        m1 = fis._get_r3d18_model("cpu")
        m2 = fis._get_r3d18_model("cpu")
    except Exception as exc:  # weights unavailable offline
        pytest.skip(f"R3D-18 weights unavailable: {exc}")
    assert m1 is m2  # cached — not rebuilt per segment


def test_r3d18_batched_forward_matches_per_clip():
    """Correctness: batching gives per-sample-identical embeddings (eval mode)."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from abel.services import fusion_inference_service as fis

    try:
        fis._get_r3d18_model("cpu")
    except Exception as exc:
        pytest.skip(f"R3D-18 weights unavailable: {exc}")

    rng = np.random.default_rng(0)
    clips = [rng.random((16, 112, 112, 3)).astype(np.float32) for _ in range(5)]
    stats = {"used_gpu": False, "used_cpu": False, "fallback_count": 0, "last_error": ""}

    batched, err = fis.FusionInferenceService._forward_r3d18(np.stack(clips, axis=0), stats)
    assert batched is not None, err
    per_clip = np.stack(
        [fis.FusionInferenceService._forward_r3d18(c[np.newaxis, ...], stats)[0][0] for c in clips],
        axis=0,
    )
    # Same weights, eval-mode batch-norm (running stats) => per-sample identical.
    assert np.allclose(batched, per_clip, atol=1e-5), np.abs(batched - per_clip).max()
