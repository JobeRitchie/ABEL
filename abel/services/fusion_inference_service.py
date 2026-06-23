"""Selective keypoint-video fusion for uncertain segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("abel")

# Serialise GPU R3D18 calls so concurrent worker threads don't cause CUDA
# contention or access violations.  Same pattern as the optical-flow GPU lock.
_r3d18_gpu_lock = threading.Lock()
_R3D18_GPU_LOCK_TIMEOUT: float = 120.0


@dataclass
class FusionConfig:
    uncertainty_threshold: float = 0.35
    alpha_pose_context: float = 0.7
    beta_video_embedding: float = 0.3
    embedding_backend: str = "auto"  # auto | r3d18 | handcrafted
    n_sample_frames: int = 16


class FusionInferenceService:
    """Augment uncertain predictions with local video crop embeddings."""

    @staticmethod
    def _video_crop_embedding_r3d18(
        video_path: Path,
        start_frame: int,
        end_frame: int,
        roi: dict[str, Any],
        n_sample_frames: int,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        try:
            import cv2
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except Exception:
            return None, None, "torch/torchvision/cv2 import failed"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, f"failed to open video {video_path}"

        x = int(roi.get("x", 0) or 0)
        y = int(roi.get("y", 0) or 0)
        w = max(1, int(roi.get("w", 64) or 64))
        h = max(1, int(roi.get("h", 64) or 64))

        n_total = max(1, end_frame - start_frame + 1)
        idxs = np.linspace(0, n_total - 1, num=max(4, int(n_sample_frames)), dtype=int)
        frames: list[np.ndarray] = []
        try:
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
        finally:
            cap.release()

        if len(frames) < 4:
            return None, None, "insufficient sampled frames for r3d18"

        x_tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(3, 0, 1, 2).unsqueeze(0)
        # Prefer CUDA where available; always retry on CPU to avoid hard failure.
        device_order = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        last_error = ""
        # Serialise GPU access across worker threads to prevent CUDA contention
        # and the intermittent access violations that prompted the earlier
        # blanket worker-thread block.  The lock lets R3D18 run on GPU safely
        # from any thread by ensuring only one call uses the device at a time.
        lock_acquired = _r3d18_gpu_lock.acquire(timeout=_R3D18_GPU_LOCK_TIMEOUT)
        if not lock_acquired:
            logger.warning("R3D18 GPU lock timed out after %.0fs; falling back to CPU.", _R3D18_GPU_LOCK_TIMEOUT)
            device_order = ["cpu"]
        try:
            for device_name in device_order:
                try:
                    device = torch.device(device_name)
                    x_dev = x_tensor.to(device=device, dtype=torch.float32)
                    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
                    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
                    x_dev = (x_dev - mean) / std

                    model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(device)
                    model.eval()
                    with torch.no_grad():
                        feats = model.stem(x_dev)
                        feats = model.layer1(feats)
                        feats = model.layer2(feats)
                        pooled = feats.mean(dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy()
                    del model
                    if device_name == "cuda":
                        torch.cuda.empty_cache()
                    return pooled.astype(np.float32), device_name, ""
                except Exception as exc:
                    last_error = str(exc).splitlines()[0]
                    if device_name == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
        finally:
            if lock_acquired:
                _r3d18_gpu_lock.release()
        return None, None, last_error or "r3d18 inference failed"

    @staticmethod
    def _video_crop_embedding(video_path: Path, start_frame: int, end_frame: int, roi: dict[str, Any]) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:
            raise ImportError("opencv-python is required for fusion inference") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        x = int(roi.get("x", 0) or 0)
        y = int(roi.get("y", 0) or 0)
        w = max(1, int(roi.get("w", 64) or 64))
        h = max(1, int(roi.get("h", 64) or 64))

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

            sample = np.array(
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
            samples.append(sample)

        cap.release()
        if not samples:
            return np.zeros(6, dtype=np.float32)
        return np.mean(np.stack(samples, axis=0), axis=0)

    def fuse_uncertain_segments(
        self,
        segments: pd.DataFrame,
        video_lookup: dict[str, Path],
        roi_lookup: dict[str, dict[str, int]],
        config: FusionConfig | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        cfg = config or FusionConfig()
        out = segments.copy()
        fused_probs: list[float] = []
        used_gpu = False
        used_cpu = False
        fallback_count = 0
        gpu_available = False
        last_error = ""

        try:
            import torch

            gpu_available = bool(torch.cuda.is_available())
        except Exception:
            gpu_available = False

        for row in out.itertuples(index=False):
            base_prob = float(getattr(row, "prediction_prob"))
            unc = float(getattr(row, "uncertainty_score"))
            if unc <= cfg.uncertainty_threshold:
                fused_probs.append(base_prob)
                continue

            session_id = str(getattr(row, "session_id"))
            video_path = video_lookup.get(session_id)
            roi = roi_lookup.get(session_id, {"x": 0, "y": 0, "w": 64, "h": 64})
            if not video_path:
                fused_probs.append(base_prob)
                continue

            emb: np.ndarray | None = None
            backend = (cfg.embedding_backend or "auto").lower().strip()
            if backend in {"auto", "r3d18"}:
                emb, device_name, err = self._video_crop_embedding_r3d18(
                    video_path=video_path,
                    start_frame=int(getattr(row, "start_frame")),
                    end_frame=int(getattr(row, "end_frame")),
                    roi=roi,
                    n_sample_frames=max(4, int(cfg.n_sample_frames)),
                )
                if device_name == "cuda":
                    used_gpu = True
                if device_name == "cpu":
                    used_cpu = True
                if err:
                    last_error = err
            if emb is None:
                fallback_count += 1
                emb = self._video_crop_embedding(
                    video_path=video_path,
                    start_frame=int(getattr(row, "start_frame")),
                    end_frame=int(getattr(row, "end_frame")),
                    roi=roi,
                )
                used_cpu = True

            # Normalize embedding score into [0, 1] for fusion.
            if emb.shape[0] >= 5:
                emb_score = float(1.0 / (1.0 + np.exp(-((float(np.mean(emb[:4])) + float(np.std(emb))) - 0.5) / 0.25)))
            else:
                emb_score = float(1.0 / (1.0 + np.exp(-((emb[2] + emb[4]) - 10.0) / 5.0)))
            fused = cfg.alpha_pose_context * base_prob + cfg.beta_video_embedding * emb_score
            fused_probs.append(float(np.clip(fused, 0.0, 1.0)))

        out["prediction_prob_fused"] = np.asarray(fused_probs)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "fusion_gpu_available": bool(gpu_available),
                    "fusion_device_used": "gpu" if used_gpu else "cpu",
                    "fusion_used_cpu_fallback": bool(fallback_count > 0),
                    "fusion_fallback_count": int(fallback_count),
                    "fusion_fallback_reason": last_error,
                }
            )
        return out
