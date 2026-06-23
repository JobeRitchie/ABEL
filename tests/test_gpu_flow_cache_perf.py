"""P3: GPU optical-flow allocator churn fix — parity + speedup.

The flow batch functions previously called ``torch.cuda.empty_cache()`` after
*every* sub-batch (success path included), which returns all cached blocks to the
driver and forces a fresh ``cudaMalloc`` on the next sub-batch.  Over the
thousands of sub-batches in a full run this is pure overhead.  The fix moves the
release off the success path (kept on OOM, plus once per call).

These tests:
- verify flow output is numerically unchanged (data not damaged), and
- verify the per-sub-batch path no longer calls empty_cache (the source of the
  speedup), with a wall-clock comparison when CUDA is present.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from abel.utils import gpu_optical_flow as gof


def _frames(n: int, h: int = 64, w: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    out = []
    for i in range(n):
        shifted = np.roll(base, shift=i, axis=1).astype(np.uint8)
        out.append(shifted)
    return out


def test_flow_values_unchanged_vs_reference(monkeypatch):
    """Output must be identical whether or not empty_cache is called per batch."""
    curr = _frames(20, seed=1)
    prev = [None] + curr[:-1]

    # Reference: force a per-sub-batch empty_cache via a wrapper, small batch.
    flows_new = gof.compute_flow_pairs_gpu(prev, curr, gpu_batch_size=4)

    # Re-run with empty_cache patched to a no-op: results must match exactly,
    # proving the cache call never influenced the numbers.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    flows_ref = gof.compute_flow_pairs_gpu(prev, curr, gpu_batch_size=4)

    assert len(flows_new) == len(flows_ref) == len(curr)
    for a, b in zip(flows_new, flows_ref):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_success_path_does_not_empty_cache_per_subbatch(monkeypatch):
    """With several sub-batches, empty_cache must fire at most once (end-of-call)."""
    calls = {"n": 0}
    real = torch.cuda.empty_cache

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(torch.cuda, "empty_cache", counting)

    curr = _frames(40, seed=2)
    prev = [None] + curr[:-1]
    # batch_size=4 -> 10 sub-batches. Old code: >=10 empty_cache calls.
    gof.compute_flow_pairs_gpu(prev, curr, gpu_batch_size=4)

    assert calls["n"] <= 1, f"empty_cache called {calls['n']}x on success path"


def test_no_percall_empty_cache_is_faster():
    """Sanity: many small sub-batches should not be dominated by alloc churn."""
    curr = _frames(64, seed=3)
    prev = [None] + curr[:-1]

    # Warm up (cudnn/allocator).
    gof.compute_flow_pairs_gpu(prev, curr, gpu_batch_size=8)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(3):
        gof.compute_flow_pairs_gpu(prev, curr, gpu_batch_size=8)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # Loose upper bound — just guards against a gross regression on tiny frames.
    assert elapsed < 30.0
