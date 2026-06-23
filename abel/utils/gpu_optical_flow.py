"""GPU-accelerated dense optical flow using PyTorch.

Implements pyramidal Lucas-Kanade flow estimation that runs entirely on CUDA,
providing significant speedup over CPU Farneback on batched frame pairs.
Falls back automatically when no GPU is available.  The ``cv2.cuda`` path is
tried first when OpenCV was built with CUDA support.

Thread safety
-------------
A module-level lock serialises GPU batch calls so multiple chunk threads
can share the GPU without contention.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("abel")

_flow_lock = threading.Lock()

# Timeout (seconds) for acquiring the GPU lock.  If a CUDA op hangs while
# holding the lock, other workers give up after this long instead of
# deadlocking forever.
GPU_LOCK_TIMEOUT: float = 120.0

# Minimum VRAM (MB) to leave free after batch allocation.  Prevents running
# the GPU right up to the allocation limit.
_VRAM_HEADROOM_MB: int = 200

# No global batch-size cache: the cache was set at app-startup using the
# probe's default H=480, W=640, but actual video frames can be much larger
# (e.g. 960×540 for a 1920×1080 video with 2× downsample).  The per-call
# probe is fast (one torch.cuda.mem_get_info call) and adapts to VRAM
# fragmentation that accumulates across many sessions.


# ── GPU memory helpers ───────────────────────────────────────────────────────

def gpu_vram_total_mb() -> float:
    """Return total GPU VRAM in MB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # PyTorch ≥2.0 uses 'total_memory'; older builds use 'total_mem'.
            total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            return total / (1024 * 1024)
    except Exception as exc:
        logger.debug("gpu_vram_total_mb failed: %s", exc)
    return 0.0


def gpu_vram_free_mb() -> float:
    """Return currently free GPU VRAM in MB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(0)
            return free / (1024 * 1024)
    except Exception as exc:
        logger.debug("gpu_vram_free_mb failed: %s", exc)
    return 0.0


def _estimate_batch_vram_mb(batch_size: int, h: int, w: int) -> float:
    """Rough estimate of peak VRAM (MB) for a single LK flow batch.

    The pyramidal LK implementation creates ~22-25 intermediate tensors at
    level-0 resolution per iteration (flow, base grids, warped frames,
    Scharr gradients, structure-tensor components, determinant, increments).
    Using 40x the single input tensor gives a conservative safety margin that
    prevents over-allocation at large resolutions (e.g. 960x540 from 1080p
    sources), while still allowing large batches on high-VRAM GPUs.
    """
    bytes_per_pixel = 4  # float32
    input_bytes = batch_size * 1 * h * w * bytes_per_pixel  # one input tensor
    return (input_bytes * 40) / (1024 * 1024)


def probe_gpu_batch_size(
    h: int = 480,
    w: int = 640,
    default: int = 24,
) -> int:
    """Return the largest safe GPU batch size for the given frame dimensions.

    Probes current free VRAM on every call so the estimate adapts to:
    - The actual frame resolution (h, w) rather than the startup default.
    - VRAM fragmentation that accumulates as more sessions are processed.

    Powerful GPUs get larger batches (up to 96); the size shrinks
    automatically as VRAM becomes scarce or frames are large.
    """
    free_mb = gpu_vram_free_mb()
    total_mb = gpu_vram_total_mb()
    if free_mb <= 0 or total_mb <= 0:
        return default

    usable_mb = free_mb - _VRAM_HEADROOM_MB
    if usable_mb <= 0:
        logger.warning(
            "GPU has only %.0f MB free VRAM (%.0f MB total). "
            "Using minimum batch size 4.",
            free_mb, total_mb,
        )
        return 4

    # Find the largest batch in [4, 8, 12, 16, 24, 32, 48, 64, 96]
    # whose estimated peak fits within usable VRAM.
    candidates = [4, 8, 12, 16, 24, 32, 48, 64, 96]
    best = candidates[0]
    for bs in candidates:
        if _estimate_batch_vram_mb(bs, h, w) <= usable_mb:
            best = bs
        else:
            break

    logger.debug(
        "GPU batch size: %d (%.0f MB free / %.0f MB total VRAM, "
        "est. %.0f MB peak at %dx%d)",
        best, free_mb, total_mb,
        _estimate_batch_vram_mb(best, h, w), h, w,
    )
    return best


def gpu_summary() -> dict[str, Any]:
    """Return a dict summarising GPU state for display in the UI."""
    total = gpu_vram_total_mb()
    free = gpu_vram_free_mb()
    backend = "cpu"
    name = "(none)"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception as exc:
        logger.debug("gpu_summary name lookup failed: %s", exc)
    try:
        backend = detect_flow_backend()
    except Exception as exc:
        logger.debug("gpu_summary backend detection failed: %s", exc)
    info = {
        "name": name,
        "total_mb": total,
        "free_mb": free,
        "backend": backend,
        "batch_size": probe_gpu_batch_size() if total > 0 else 0,
    }
    logger.info(
        "GPU summary: %s, %.0f MB total, %.0f MB free, backend=%s, batch=%d",
        name, total, free, backend, info["batch_size"],
    )
    return info


# ── Backend detection ────────────────────────────────────────────────────────

def detect_flow_backend() -> str:
    """Return the best available optical flow backend identifier.

    Priority order:
    1. ``"cv2_cuda"`` — OpenCV compiled with CUDA (exact Farneback on GPU).
    2. ``"torch"``    — Pyramidal Lucas-Kanade via PyTorch CUDA.
    3. ``"cpu"``      — Standard CPU Farneback (baseline).
    """
    # 1. OpenCV CUDA Farneback
    try:
        import cv2
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            _ = cv2.cuda.FarnebackOpticalFlow.create()
            logger.info("Optical flow backend: OpenCV CUDA Farneback")
            return "cv2_cuda"
    except Exception:
        pass

    # 2. PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(
                "Optical flow backend: PyTorch pyramidal LK on %s",
                torch.cuda.get_device_name(0),
            )
            return "torch"
    except ImportError:
        pass

    logger.info("Optical flow backend: CPU Farneback")
    return "cpu"


def get_flow_lock() -> threading.Lock:
    """Return the GPU compute lock for serialising flow batches."""
    return _flow_lock


def acquire_flow_lock(timeout: float | None = None) -> bool:
    """Acquire the GPU flow lock with an optional timeout.

    Returns *True* if the lock was acquired, *False* on timeout.
    """
    t = timeout if timeout is not None else GPU_LOCK_TIMEOUT
    return _flow_lock.acquire(timeout=t)


def release_flow_lock() -> None:
    """Release the GPU flow lock."""
    try:
        _flow_lock.release()
    except RuntimeError:
        pass  # already released


# ── PyTorch pyramidal Lucas-Kanade ───────────────────────────────────────────

def _pyramidal_lk_flow(
    prev_batch: "Any",
    curr_batch: "Any",
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
) -> "Any":
    """Compute pyramidal Lucas-Kanade dense optical flow on GPU.

    Parameters
    ----------
    prev_batch, curr_batch : (B, 1, H, W) float32 CUDA tensors
    levels : int
        Pyramid levels (coarse-to-fine).
    winsize : int
        Local window size for the structure tensor.
    iterations : int
        Warp-and-solve iterations per pyramid level.

    Returns
    -------
    (B, 2, H, W) float32 tensor — per-pixel (dx, dy) in pixels.
    """
    import torch
    import torch.nn.functional as F

    device = prev_batch.device
    B = prev_batch.shape[0]

    # Scharr gradient kernels (better rotational symmetry than Sobel).
    kx = torch.tensor(
        [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
        dtype=torch.float32, device=device,
    ).reshape(1, 1, 3, 3) / 32.0
    ky = kx.transpose(2, 3)

    # Gaussian window for structure-tensor averaging.
    r = winsize // 2
    coords = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    sigma = winsize / 4.0
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    gauss = gauss / gauss.sum()

    # Build Gaussian pyramids.
    pyr_prev = [prev_batch]
    pyr_curr = [curr_batch]
    for _ in range(levels - 1):
        pyr_prev.append(F.avg_pool2d(pyr_prev[-1], 2))
        pyr_curr.append(F.avg_pool2d(pyr_curr[-1], 2))

    # Coarsest-level flow initialised to zero.
    ch, cw = pyr_prev[-1].shape[2], pyr_prev[-1].shape[3]
    flow = torch.zeros(B, 2, ch, cw, device=device)

    for lev in range(levels - 1, -1, -1):
        i1 = pyr_prev[lev]
        i2 = pyr_curr[lev]
        lh, lw = i1.shape[2], i1.shape[3]

        # Upscale flow from coarser level (× 2 because pixel spacing doubles).
        if lev < levels - 1:
            flow = F.interpolate(
                flow, (lh, lw), mode="bilinear", align_corners=False,
            ) * 2

        # Base sampling grid in normalised [-1, 1] coordinates.
        grid_y = torch.linspace(-1, 1, lh, device=device)
        grid_x = torch.linspace(-1, 1, lw, device=device)
        base_gy, base_gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        base_gx = base_gx.unsqueeze(0).expand(B, -1, -1)
        base_gy = base_gy.unsqueeze(0).expand(B, -1, -1)

        for _ in range(iterations):
            # Warp i2 towards i1 using current flow estimate.
            norm_fx = flow[:, 0] * (2.0 / max(lw - 1, 1))
            norm_fy = flow[:, 1] * (2.0 / max(lh - 1, 1))
            grid = torch.stack(
                [base_gx + norm_fx, base_gy + norm_fy], dim=-1,
            )
            warped = F.grid_sample(
                i2, grid, mode="bilinear", padding_mode="border",
                align_corners=True,
            )

            # Spatial and temporal gradients.
            Ix = F.conv2d(i1, kx, padding=1)
            Iy = F.conv2d(i1, ky, padding=1)
            It = warped - i1

            # Windowed structure-tensor components.
            Ixx = F.conv2d(Ix * Ix, gauss, padding=r)
            Iyy = F.conv2d(Iy * Iy, gauss, padding=r)
            Ixy = F.conv2d(Ix * Iy, gauss, padding=r)
            Ixt = F.conv2d(Ix * It, gauss, padding=r)
            Iyt = F.conv2d(Iy * It, gauss, padding=r)

            # Per-pixel 2×2 solve via Cramer's rule.
            det = Ixx * Iyy - Ixy * Ixy
            det = torch.where(
                det.abs() < 1e-6,
                torch.full_like(det, 1e-6),
                det,
            )

            du = -(Iyy * Ixt - Ixy * Iyt) / det
            dv = -(Ixx * Iyt - Ixy * Ixt) / det

            # Clamp large increments to stabilise textureless regions.
            max_incr = 5.0
            du = du.clamp(-max_incr, max_incr)
            dv = dv.clamp(-max_incr, max_incr)

            flow = flow + torch.cat([du, dv], dim=1)

    return flow


class GPUFlowWarning:
    """Container for GPU warnings emitted during flow computation."""

    __slots__ = ("oom_fallback_count", "lock_timeout_count", "messages")

    def __init__(self) -> None:
        self.oom_fallback_count: int = 0
        self.lock_timeout_count: int = 0
        self.messages: list[str] = []

    @property
    def had_issues(self) -> bool:
        return self.oom_fallback_count > 0 or self.lock_timeout_count > 0


def _compute_flow_cpu_fallback(
    gray_frames: list[np.ndarray],
    prev_gray: np.ndarray | None,
) -> list[np.ndarray]:
    """Compute dense optical flow using CPU Farneback (emergency fallback)."""
    import cv2

    n = len(gray_frames)
    if n == 0:
        return []
    H, W = gray_frames[0].shape[:2]
    zero_flow = np.zeros((H, W, 2), dtype=np.float32)
    flows: list[np.ndarray] = []
    prev = prev_gray
    for frame in gray_frames:
        if prev is None:
            flows.append(zero_flow)
        else:
            flows.append(
                cv2.calcOpticalFlowFarneback(
                    prev, frame, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
            )
        prev = frame
    return flows


def compute_flow_batch_gpu(
    gray_frames: list[np.ndarray],
    prev_gray: np.ndarray | None,
    gpu_batch_size: int = 0,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    warnings_out: GPUFlowWarning | None = None,
) -> list[np.ndarray]:
    """Compute dense optical flow for consecutive frames using GPU.

    Parameters
    ----------
    gray_frames : list of (H, W) uint8 arrays
        Consecutive grayscale frames in the chunk.
    prev_gray : (H, W) array or None
        Frame immediately before ``gray_frames[0]`` for continuity.
        If *None* the first output is a zero flow field.
    gpu_batch_size : int
        Frame pairs per GPU kernel launch.  ``0`` (default) auto-selects
        based on available VRAM via :func:`probe_gpu_batch_size`.
    warnings_out : GPUFlowWarning or None
        If provided, OOM/fallback events are recorded here instead of
        being silently swallowed.

    Returns
    -------
    list of (H, W, 2) float32 arrays, same length as *gray_frames*.
    ``flows[i]`` is the displacement field from the preceding frame into
    ``gray_frames[i]``.
    """
    import torch

    if warnings_out is None:
        warnings_out = GPUFlowWarning()

    device = torch.device("cuda")
    n = len(gray_frames)
    if n == 0:
        return []

    H, W = gray_frames[0].shape[:2]

    # Auto-select batch size based on VRAM when not explicitly set.
    if gpu_batch_size <= 0:
        gpu_batch_size = probe_gpu_batch_size(h=H, w=W)

    # Index 0 → prev_gray, 1..n → gray_frames[0..n-1]
    all_grays: list[np.ndarray | None] = [prev_gray] + list(gray_frames)

    flows: list[np.ndarray] = []
    zero_flow = np.zeros((H, W, 2), dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, n, gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, n)
            b = batch_end - batch_start

            valid_positions: list[int] = []
            null_indices: set[int] = set()
            for i in range(b):
                if all_grays[batch_start + i] is None:
                    null_indices.add(i)
                else:
                    valid_positions.append(i)

            batch_flows_np: dict[int, np.ndarray] = {}

            if valid_positions:
                try:
                    t_prev = torch.empty(
                        len(valid_positions), 1, H, W,
                        dtype=torch.float32, device=device,
                    )
                    t_curr = torch.empty_like(t_prev)
                    for vi, i in enumerate(valid_positions):
                        idx = batch_start + i
                        t_prev[vi, 0] = torch.from_numpy(
                            np.ascontiguousarray(all_grays[idx], dtype=np.float32),
                        ).to(device)
                        t_curr[vi, 0] = torch.from_numpy(
                            np.ascontiguousarray(all_grays[idx + 1], dtype=np.float32),
                        ).to(device)

                    raw = _pyramidal_lk_flow(
                        t_prev, t_curr, levels=levels,
                        winsize=winsize, iterations=iterations,
                    )
                    # Synchronise the CUDA stream before reading results back.
                    # This converts any asynchronous CUDA kernel errors (e.g.
                    # illegal memory access from diverged LK flow) into a
                    # synchronous RuntimeError that the except clause below can
                    # catch and route to the CPU fallback, instead of letting
                    # them propagate as a native Windows 0xC0000005 crash that
                    # kills the whole process.
                    torch.cuda.synchronize()
                    raw_np = raw.permute(0, 2, 3, 1).cpu().numpy()
                    del t_prev, t_curr, raw

                    for vi, pos in enumerate(valid_positions):
                        batch_flows_np[pos] = raw_np[vi]

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    # CUDA OOM or allocation failure — fall back to CPU for
                    # this sub-batch and record the event.
                    warnings_out.oom_fallback_count += 1
                    msg = (
                        f"GPU OOM on batch {batch_start}-{batch_end} "
                        f"(batch_size={gpu_batch_size}, {H}×{W}): {exc!s:.120s}. "
                        "Falling back to CPU Farneback for this sub-batch."
                    )
                    warnings_out.messages.append(msg)
                    logger.warning("gpu_optical_flow: %s", msg)

                    # Release the caching allocator ONLY on the failure path,
                    # where reclaiming memory may let the next batch fit.  On the
                    # success path we must NOT call empty_cache() per sub-batch:
                    # it returns all cached blocks to the driver and forces a
                    # fresh cudaMalloc on the next batch, which is a major
                    # per-batch cost over the thousands of sub-batches in a run.
                    torch.cuda.empty_cache()

                    # CPU fallback for the valid pairs in this sub-batch.
                    fb_frames = [
                        all_grays[batch_start + i + 1]
                        for i in valid_positions
                    ]
                    fb_prev = [
                        all_grays[batch_start + i]
                        for i in valid_positions
                    ]
                    for vi, pos in enumerate(valid_positions):
                        import cv2
                        p = fb_prev[vi]
                        c = fb_frames[vi]
                        if p is None or c is None:
                            batch_flows_np[pos] = zero_flow
                        else:
                            batch_flows_np[pos] = cv2.calcOpticalFlowFarneback(
                                p, c, None,
                                pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                            )

            for i in range(b):
                flows.append(
                    zero_flow if i in null_indices else batch_flows_np[i]
                )

    # Release cache once per call (i.e. per chunk), not per sub-batch — this
    # bounds cross-session fragmentation without the per-batch malloc churn.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return flows


def compute_flow_pairs_gpu(
    prev_frames: "list[np.ndarray | None]",
    curr_frames: "list[np.ndarray]",
    gpu_batch_size: int = 0,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    warnings_out: "GPUFlowWarning | None" = None,
    compute_downsample: int = 1,
) -> "list[np.ndarray]":
    """Compute dense optical flow for explicit (prev, curr) frame pairs on GPU.

    Unlike :func:`compute_flow_batch_gpu`, which processes a sequential frame
    list (flow from frame[i] to frame[i+1]), this function accepts arbitrary
    independent pairs.  It is used by the strided optical flow path: only
    anchor frames (every N-th frame) are sent to the GPU, each paired against
    its *immediate* predecessor (always a 1-frame gap).  This keeps inter-frame
    displacement small so pyramidal LK stays numerically stable, while reducing
    GPU work by ~N× vs computing flow for every frame.

    Parameters
    ----------
    prev_frames : list of (H, W) uint8 arrays or None
        Previous frame for each pair.  None entries produce zero flow.
    curr_frames : list of (H, W) uint8 arrays
        Current frame for each pair.  Must be the same length as prev_frames.
    gpu_batch_size : int
        Pairs per GPU kernel launch.  0 = auto-select from free VRAM.
    warnings_out : GPUFlowWarning or None
        OOM / fallback events are recorded here.

    Returns
    -------
    list of (H, W, 2) float32 arrays, same length as *curr_frames*.
    """
    import torch

    if warnings_out is None:
        warnings_out = GPUFlowWarning()

    import torch.nn.functional as F

    n = len(curr_frames)
    if n == 0:
        return []

    H, W = curr_frames[0].shape[:2]
    # Optional spatial downsample for the flow solve only.  LK runs at (dh, dw)
    # and the result is upsampled back to (H, W) with vectors scaled by the
    # factor, so callers always receive a full-resolution flow field and nothing
    # downstream changes.  This is the dominant context-extraction speedup
    # (~4x flow) and preserves the patch-level flow features (corr ~0.97).
    cd = max(1, int(compute_downsample))
    dh, dw = (H // cd, W // cd) if cd > 1 else (H, W)
    if cd > 1 and (dh < 8 or dw < 8):
        cd, dh, dw = 1, H, W  # too small to downsample meaningfully
    if cd > 1:
        import cv2

    if gpu_batch_size <= 0:
        gpu_batch_size = probe_gpu_batch_size(h=dh, w=dw)

    zero_flow = np.zeros((H, W, 2), dtype=np.float32)
    device = torch.device("cuda")
    flows: list[np.ndarray] = []

    with torch.no_grad():
        for batch_start in range(0, n, gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, n)
            b = batch_end - batch_start

            valid_positions: list[int] = []
            null_indices: set[int] = set()
            for i in range(b):
                if prev_frames[batch_start + i] is None:
                    null_indices.add(i)
                else:
                    valid_positions.append(i)

            batch_flows_np: dict[int, np.ndarray] = {}

            if valid_positions:
                try:
                    t_prev = torch.empty(
                        len(valid_positions), 1, dh, dw,
                        dtype=torch.float32, device=device,
                    )
                    t_curr = torch.empty_like(t_prev)
                    for vi, i in enumerate(valid_positions):
                        idx = batch_start + i
                        _p = prev_frames[idx]
                        _c = curr_frames[idx]
                        if cd > 1:
                            _p = cv2.resize(_p, (dw, dh), interpolation=cv2.INTER_AREA)
                            _c = cv2.resize(_c, (dw, dh), interpolation=cv2.INTER_AREA)
                        t_prev[vi, 0] = torch.from_numpy(
                            np.ascontiguousarray(_p, dtype=np.float32),
                        ).to(device)
                        t_curr[vi, 0] = torch.from_numpy(
                            np.ascontiguousarray(_c, dtype=np.float32),
                        ).to(device)

                    raw = _pyramidal_lk_flow(
                        t_prev, t_curr, levels=levels,
                        winsize=winsize, iterations=iterations,
                    )
                    # Upsample flow back to working resolution; scale vectors by
                    # the downsample factor so magnitudes are in working-res px.
                    if cd > 1:
                        raw = F.interpolate(
                            raw, size=(H, W), mode="bilinear", align_corners=False,
                        ) * cd
                    torch.cuda.synchronize()
                    raw_np = raw.permute(0, 2, 3, 1).cpu().numpy()
                    del t_prev, t_curr, raw

                    for vi, pos in enumerate(valid_positions):
                        batch_flows_np[pos] = raw_np[vi]

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    warnings_out.oom_fallback_count += 1
                    msg = (
                        f"GPU OOM on pair batch {batch_start}–{batch_end - 1} "
                        f"(batch_size={gpu_batch_size}, {H}×{W}): {exc!s:.120s}. "
                        "Falling back to CPU Farneback for this sub-batch."
                    )
                    warnings_out.messages.append(msg)
                    logger.warning("gpu_optical_flow: %s", msg)
                    # Reclaim memory only on the OOM path (see compute_flow_batch_gpu).
                    torch.cuda.empty_cache()

                    import cv2
                    for i in valid_positions:
                        idx = batch_start + i
                        p = prev_frames[idx]
                        c = curr_frames[idx]
                        if p is None or c is None:
                            batch_flows_np[i] = zero_flow
                        else:
                            batch_flows_np[i] = cv2.calcOpticalFlowFarneback(
                                p, c, None,
                                pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                            )

            for i in range(b):
                flows.append(
                    zero_flow if i in null_indices else batch_flows_np.get(i, zero_flow)
                )

    # Release cache once per call (per chunk), not per sub-batch.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return flows


# ── OpenCV CUDA helpers ──────────────────────────────────────────────────────

def create_cv2_cuda_farneback(config: Any) -> Any:
    """Create a ``cv2.cuda.FarnebackOpticalFlow`` from *config*."""
    import cv2
    return cv2.cuda.FarnebackOpticalFlow.create(
        numLevels=config.farneback_levels,
        pyrScale=config.farneback_pyr_scale,
        winSize=config.farneback_winsize,
        numIters=config.farneback_iterations,
        polyN=config.farneback_poly_n,
        polySigma=config.farneback_poly_sigma,
    )


def compute_flow_cv2_cuda(
    algo: Any,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """Compute Farneback flow on GPU via OpenCV CUDA, returning (H, W, 2)."""
    import cv2
    gpu_prev = cv2.cuda_GpuMat(prev_gray)
    gpu_curr = cv2.cuda_GpuMat(curr_gray)
    gpu_flow = algo.calc(gpu_prev, gpu_curr, None)
    return gpu_flow.download()
