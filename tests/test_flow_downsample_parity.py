"""P3 prototype → impl: flow spatial-downsample preserves information.

Optical flow dominates context-feature extraction (~73%).  Because the flow
features are mean magnitude/direction over small patches (low-frequency),
solving flow at reduced resolution and upsampling the field back is a large
speedup that preserves the *information* the model uses.  We assert:

1. the returned flow field is still full working-resolution (downstream code
   that crops patches is therefore unchanged), and
2. the per-patch flow-magnitude signal is highly correlated with the full-res
   baseline (information preserved, even though exact values differ), and
3. magnitudes stay on the same scale (upsample factor applied correctly).

A wall-clock speedup assertion runs only when CUDA is present.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from abel.utils.gpu_optical_flow import compute_flow_pairs_gpu


def _moving_frames(n: int, h: int = 256, w: int = 256, seed: int = 0):
    """Textured static background with a smoothly moving bright blob."""
    rng = np.random.default_rng(seed)
    bg = (rng.random((h, w)) * 40 + 60).astype(np.uint8)
    yy, xx = np.ogrid[:h, :w]
    frames = []
    cx, cy = w // 2, h // 2
    for i in range(n):
        cx = int(np.clip(cx + rng.standard_normal() * 5, 40, w - 40))
        cy = int(np.clip(cy + rng.standard_normal() * 4, 40, h - 40))
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 25.0 ** 2))) * 150
        frames.append(np.clip(bg.astype(np.float32) + blob, 0, 255).astype(np.uint8))
    return frames


def _patch_mag(flows, h, w, r=24):
    out = []
    for f in flows:
        s = f[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
        out.append(float(np.mean(np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2))))
    return np.array(out)


def test_downsampled_flow_is_full_res_and_correlated():
    frames = _moving_frames(60)
    H, W = frames[0].shape
    prev = [None] + frames[:-1]
    curr = frames

    base = compute_flow_pairs_gpu(prev, curr, compute_downsample=1, iterations=3)
    fast = compute_flow_pairs_gpu(prev, curr, compute_downsample=2, iterations=2)

    # 1. Full working-resolution output (downstream patch extraction unchanged).
    assert base[5].shape == (H, W, 2)
    assert fast[5].shape == (H, W, 2)

    base_sig = _patch_mag(base, H, W)
    fast_sig = _patch_mag(fast, H, W)

    # 2. Information preserved: strong correlation despite different values.
    corr = np.corrcoef(base_sig, fast_sig)[0, 1]
    assert corr >= 0.9, f"flow-magnitude correlation too low: {corr:.3f}"

    # 3. Same scale (upsample * factor applied): mean ratio near 1.
    ratio = fast_sig.mean() / max(base_sig.mean(), 1e-6)
    assert 0.5 <= ratio <= 2.0, f"magnitude scale drifted: ratio={ratio:.2f}"


def test_downsample_is_faster():
    import time

    frames = _moving_frames(120, h=480, w=640)
    prev = [None] + frames[:-1]
    curr = frames

    compute_flow_pairs_gpu(prev[:8], curr[:8], compute_downsample=2)
    torch.cuda.synchronize()

    def bench(cd, it):
        t0 = time.perf_counter()
        compute_flow_pairs_gpu(prev, curr, compute_downsample=cd, iterations=it)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    base = min(bench(1, 3) for _ in range(2))
    fast = min(bench(2, 2) for _ in range(2))
    assert fast < base, f"downsampled flow not faster: base={base:.3f}s fast={fast:.3f}s"
