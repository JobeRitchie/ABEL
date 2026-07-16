"""Selective keypoint-video fusion for uncertain segments.

Performance design (all behaviour-preserving):

* The R3D-18 network is loaded **once** per device and reused for every clip
  (previously it was reconstructed and re-loaded from disk for every segment).
* Clip embeddings are computed in **batches** through a single forward pass
  (previously one forward per clip). In ``eval`` mode batch-norm uses running
  statistics, so batching is numerically per-sample identical.
* Each video is opened **once** and its uncertain segments are read in ascending
  frame order (previously the file was re-opened per segment).
* Embedding *scores* are **cached by segment** (``score_cache``) and reused
  across behaviours — the crop embedding depends only on (video, frames, ROI),
  not on the behaviour being trained, so a multi-behaviour "retrain all" run
  embeds each segment once rather than once per behaviour.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")

# Serialise GPU R3D18 forward passes so concurrent worker threads don't cause
# CUDA contention or access violations.  Same pattern as the optical-flow GPU
# lock.  Held only around the batched forward now — not model construction or
# video decode — so contention is far lower than the old per-segment locking.
_r3d18_gpu_lock = threading.Lock()
_R3D18_GPU_LOCK_TIMEOUT: float = 120.0

# One R3D-18 instance per device, loaded lazily and reused for all clips.
_r3d18_model_lock = threading.Lock()
_R3D18_MODELS: dict[str, Any] = {}

# Cap the per-forward batch so a large uncertain set can't exhaust GPU memory.
_R3D18_BATCH_MAX = 32


def _get_r3d18_model(device_name: str):
    """Return a cached, eval-mode R3D-18 for ``device_name`` (loads once)."""
    with _r3d18_model_lock:
        model = _R3D18_MODELS.get(device_name)
        if model is None:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18

            model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(torch.device(device_name)).eval()
            _R3D18_MODELS[device_name] = model
        return model


@dataclass
class FusionConfig:
    uncertainty_threshold: float = 0.35
    alpha_pose_context: float = 0.7
    beta_video_embedding: float = 0.3
    embedding_backend: str = "auto"  # auto | r3d18 | handcrafted
    n_sample_frames: int = 16


class FusionInferenceService:
    """Augment uncertain predictions with local video crop embeddings."""

    # ── ROI + score helpers ──────────────────────────────────────────────
    @staticmethod
    def _roi_xywh(roi: dict[str, Any]) -> tuple[int, int, int, int]:
        x = int(roi.get("x", 0) or 0)
        y = int(roi.get("y", 0) or 0)
        w = max(1, int(roi.get("w", 64) or 64))
        h = max(1, int(roi.get("h", 64) or 64))
        return x, y, w, h

    @staticmethod
    def _emb_to_score(emb: np.ndarray) -> float:
        """Normalise a crop embedding into a [0, 1] fusion score (unchanged)."""
        if emb.shape[0] >= 5:
            return float(1.0 / (1.0 + np.exp(-((float(np.mean(emb[:4])) + float(np.std(emb))) - 0.5) / 0.25)))
        return float(1.0 / (1.0 + np.exp(-((emb[2] + emb[4]) - 10.0) / 5.0)))

    # ── Clip decode (shared by single + batched paths) ───────────────────
    @classmethod
    def _decode_clip_r3d18(
        cls, cap, start_frame: int, end_frame: int, roi: dict[str, Any], n_sample_frames: int
    ) -> np.ndarray | None:
        """Decode+preprocess sampled crop frames for R3D-18: ``(T,112,112,3)``.

        Identical sampling/preprocessing to the original per-segment path; takes
        an already-open ``VideoCapture`` so a video is opened once per file.
        """
        import cv2

        x, y, w, h = cls._roi_xywh(roi)
        n_total = max(1, end_frame - start_frame + 1)
        idxs = np.linspace(0, n_total - 1, num=max(4, int(n_sample_frames)), dtype=int)
        frames: list[np.ndarray] = []
        for rel in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame + int(rel)))
            ok, frame = cap.read()
            if not ok:
                continue
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
            frames.append(rgb.astype(np.float32) / 255.0)
        if len(frames) < 4:
            return None
        return np.stack(frames, axis=0)

    @classmethod
    def _handcrafted_from_cap(cls, cap, start_frame: int, end_frame: int, roi: dict[str, Any]) -> np.ndarray:
        """Handcrafted gradient/motion embedding from an open capture (unchanged maths)."""
        import cv2

        x, y, w, h = cls._roi_xywh(roi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        samples: list[np.ndarray] = []
        last_gray: np.ndarray | None = None
        for _ in range(max(0, end_frame - start_frame + 1)):
            ok, frame = cap.read()
            if not ok:
                break
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            gray = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), dtype=np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx * gx + gy * gy)
            if last_gray is None:
                motion = np.zeros_like(gray, dtype=np.float32)
            else:
                motion = cv2.absdiff(gray, last_gray).astype(np.float32)
            last_gray = gray
            samples.append(
                np.array(
                    [
                        float(np.mean(gray)),
                        float(np.std(gray)),
                        float(np.mean(grad_mag)),
                        float(np.std(grad_mag)),
                        float(np.mean(motion)),
                        float(np.std(motion)),
                    ],
                    dtype=np.float32,
                )
            )
        if not samples:
            return np.zeros(6, dtype=np.float32)
        return np.mean(np.stack(samples, axis=0), axis=0)

    # ── Batched R3D-18 forward (model loaded once) ───────────────────────
    @classmethod
    def _forward_r3d18(cls, batch: np.ndarray, stats: dict[str, Any]) -> tuple[np.ndarray | None, str]:
        """Embed a batch of clips ``(B,T,112,112,3)`` → ``(B,128)``.

        Uses a cached model; tries CUDA (serialised by the GPU lock) then CPU.
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch always present in app env
            return None, f"torch import failed: {exc}"

        device_order = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        t = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).contiguous()
        last_error = ""
        for device_name in device_order:
            acquired = False
            if device_name == "cuda":
                acquired = _r3d18_gpu_lock.acquire(timeout=_R3D18_GPU_LOCK_TIMEOUT)
                if not acquired:
                    logger.warning("R3D18 GPU lock timed out after %.0fs; using CPU.", _R3D18_GPU_LOCK_TIMEOUT)
                    continue
            try:
                model = _get_r3d18_model(device_name)
                device = torch.device(device_name)
                x = t.to(device=device, dtype=torch.float32)
                mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
                std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
                x = (x - mean) / std
                with torch.no_grad():
                    feats = model.stem(x)
                    feats = model.layer1(feats)
                    feats = model.layer2(feats)
                    pooled = feats.mean(dim=(2, 3, 4)).detach().cpu().numpy()
                if device_name == "cuda":
                    stats["used_gpu"] = True
                    torch.cuda.empty_cache()
                else:
                    stats["used_cpu"] = True
                return pooled.astype(np.float32), ""
            except Exception as exc:
                last_error = str(exc).splitlines()[0]
                if device_name == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue
            finally:
                if acquired:
                    _r3d18_gpu_lock.release()
        return None, last_error or "r3d18 batch inference failed"

    @classmethod
    def _embed_decoded_grouped(
        cls, decoded: dict[int, np.ndarray], stats: dict[str, Any]
    ) -> tuple[dict[int, np.ndarray], list[int]]:
        """Batch-embed decoded clips (grouped by frame count) → per-index embeddings.

        Returns ``(embeddings, failed_indices)``; failed indices should fall back
        to the handcrafted embedding.
        """
        embeddings: dict[int, np.ndarray] = {}
        failed: list[int] = []
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, clip in decoded.items():
            by_len[clip.shape[0]].append(i)
        for _t_len, idxs in by_len.items():
            for chunk_start in range(0, len(idxs), _R3D18_BATCH_MAX):
                chunk = idxs[chunk_start : chunk_start + _R3D18_BATCH_MAX]
                batch = np.stack([decoded[i] for i in chunk], axis=0)
                emb_batch, err = cls._forward_r3d18(batch, stats)
                if emb_batch is None:
                    if err:
                        stats["last_error"] = err
                    failed.extend(chunk)
                    continue
                for j, i in enumerate(chunk):
                    embeddings[i] = emb_batch[j]
        return embeddings, failed

    # ── Per-video score computation (opens each video once) ───────────────
    @classmethod
    def _scores_for_video(
        cls,
        video_path: Path,
        idxs: list[int],
        sessions: list[str],
        starts: np.ndarray,
        ends: np.ndarray,
        roi_lookup: dict[str, dict[str, int]],
        backend: str,
        n_sample_frames: int,
        key_fn,
        cache: dict[str, float],
        stats: dict[str, Any],
    ) -> None:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats["last_error"] = f"failed to open video {video_path}"
            return
        try:
            use_r3d18 = backend in {"auto", "r3d18"}
            decoded: dict[int, np.ndarray] = {}
            handcraft_idxs: list[int] = []
            if use_r3d18:
                for i in idxs:
                    roi = roi_lookup.get(sessions[i], {"x": 0, "y": 0, "w": 64, "h": 64})
                    clip = cls._decode_clip_r3d18(cap, int(starts[i]), int(ends[i]), roi, n_sample_frames)
                    if clip is not None:
                        decoded[i] = clip
                    else:
                        handcraft_idxs.append(i)
            else:
                handcraft_idxs = list(idxs)

            if decoded:
                embeddings, failed = cls._embed_decoded_grouped(decoded, stats)
                for i, emb in embeddings.items():
                    cache[key_fn(i)] = cls._emb_to_score(emb)
                handcraft_idxs.extend(failed)

            # Handcrafted fallback (also the sole path for backend="handcrafted").
            for i in handcraft_idxs:
                roi = roi_lookup.get(sessions[i], {"x": 0, "y": 0, "w": 64, "h": 64})
                emb = cls._handcrafted_from_cap(cap, int(starts[i]), int(ends[i]), roi)
                cache[key_fn(i)] = cls._emb_to_score(emb)
                stats["used_cpu"] = True
                stats["fallback_count"] += 1
        finally:
            cap.release()

    # ── Backward-compatible single-clip embedding (used by tests) ─────────
    @classmethod
    def _video_crop_embedding_r3d18(
        cls,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        roi: dict[str, Any],
        n_sample_frames: int,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        """Single-clip R3D-18 embedding (reference path; shares the batched forward)."""
        try:
            import cv2  # noqa: F401
        except Exception:
            return None, None, "cv2 import failed"
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, f"failed to open video {video_path}"
        try:
            clip = cls._decode_clip_r3d18(cap, start_frame, end_frame, roi, n_sample_frames)
        finally:
            cap.release()
        if clip is None:
            return None, None, "insufficient sampled frames for r3d18"
        stats: dict[str, Any] = {"used_gpu": False, "used_cpu": False, "fallback_count": 0, "last_error": ""}
        emb, err = cls._forward_r3d18(clip[np.newaxis, ...], stats)
        if emb is None:
            return None, None, err
        device_name = "cuda" if stats["used_gpu"] else "cpu"
        return emb[0].astype(np.float32), device_name, ""

    @classmethod
    def _video_crop_embedding(cls, video_path: Path, start_frame: int, end_frame: int, roi: dict[str, Any]) -> np.ndarray:
        """Single-clip handcrafted embedding (reference path)."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        try:
            return cls._handcrafted_from_cap(cap, start_frame, end_frame, roi)
        finally:
            cap.release()

    # ── Public API ───────────────────────────────────────────────────────
    def fuse_uncertain_segments(
        self,
        segments: pd.DataFrame,
        video_lookup: dict[str, Path],
        roi_lookup: dict[str, dict[str, int]],
        config: FusionConfig | None = None,
        diagnostics: dict[str, Any] | None = None,
        score_cache: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Fuse video-crop embedding scores into uncertain segments' probabilities.

        ``score_cache`` (segment → score) is read and populated in place; pass the
        same dict across behaviours in a multi-behaviour run to embed each segment
        only once (the embedding is behaviour-independent).
        """
        cfg = config or FusionConfig()
        out = segments.copy()
        cache = score_cache if score_cache is not None else {}
        backend = (cfg.embedding_backend or "auto").lower().strip()
        n_frames = max(4, int(cfg.n_sample_frames))

        try:
            import torch

            gpu_available = bool(torch.cuda.is_available())
        except Exception:
            gpu_available = False

        n = len(out)
        seg_ids = out["segment_id"].astype(str).tolist() if "segment_id" in out.columns else [str(i) for i in range(n)]
        base_probs = out["prediction_prob"].astype(float).to_numpy()
        uncs = out["uncertainty_score"].astype(float).to_numpy()
        sessions = out["session_id"].astype(str).tolist()
        starts = out["start_frame"].to_numpy(dtype=int)
        ends = out["end_frame"].to_numpy(dtype=int)

        def _key(i: int) -> str:
            roi = roi_lookup.get(sessions[i]) or {}
            x, y, w, h = self._roi_xywh(roi)
            return f"{seg_ids[i]}::{backend}::{n_frames}::{x}_{y}_{w}_{h}"

        # Rows that are uncertain, have a video, and aren't already cached.
        need_idx = [
            i
            for i in range(n)
            if uncs[i] > cfg.uncertainty_threshold
            and video_lookup.get(sessions[i])
            and _key(i) not in cache
        ]

        stats: dict[str, Any] = {"used_gpu": False, "used_cpu": False, "fallback_count": 0, "last_error": ""}
        if need_idx:
            by_video: dict[Path, list[int]] = defaultdict(list)
            for i in need_idx:
                by_video[video_lookup[sessions[i]]].append(i)
            for video_path, idxs in by_video.items():
                idxs.sort(key=lambda i: int(starts[i]))  # ascending frame order per file
                self._scores_for_video(
                    video_path, idxs, sessions, starts, ends, roi_lookup,
                    backend, n_frames, _key, cache, stats,
                )

        # Combine: uncertain rows with a (cached) score get fused; others keep base.
        fused_probs = base_probs.astype(float).copy()
        for i in range(n):
            if uncs[i] <= cfg.uncertainty_threshold:
                continue
            score = cache.get(_key(i))
            if score is None:
                continue
            fused_probs[i] = float(np.clip(cfg.alpha_pose_context * base_probs[i] + cfg.beta_video_embedding * score, 0.0, 1.0))

        out["prediction_prob_fused"] = fused_probs
        if diagnostics is not None:
            diagnostics.update(
                {
                    "fusion_gpu_available": bool(gpu_available),
                    "fusion_device_used": "gpu" if stats["used_gpu"] else "cpu",
                    "fusion_used_cpu_fallback": bool(stats["fallback_count"] > 0),
                    "fusion_fallback_count": int(stats["fallback_count"]),
                    "fusion_fallback_reason": stats["last_error"],
                }
            )
        return out
