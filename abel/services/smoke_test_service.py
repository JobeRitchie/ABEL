"""Smoke-test runner for CPU and GPU compute pipelines.

Generates synthetic data, pushes it through the same code paths used by the
real pipeline, and reports timing + correctness for each subsystem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SmokeTestResult:
    """Outcome of a single smoke-test probe."""

    name: str
    passed: bool
    detail: str
    elapsed_ms: float = 0.0


@dataclass
class SmokeTestReport:
    """Aggregated results from all smoke-test probes."""

    results: list[SmokeTestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary_text(self) -> str:
        lines: list[str] = []
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append(f"{'=' * 60}")
        lines.append(f"  SMOKE TEST REPORT — {passed}/{total} passed")
        lines.append(f"{'=' * 60}")
        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            ms = f"{r.elapsed_ms:.1f} ms" if r.elapsed_ms else ""
            lines.append(f"  [{icon}]  {r.name}  {ms}")
            if r.detail:
                for d in r.detail.split("\n"):
                    lines.append(f"         {d}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


class SmokeTestService:
    """Run synthetic-data smoke tests across CPU and GPU subsystems."""

    def run_all(
        self,
        on_line: Callable[[str], None] | None = None,
    ) -> SmokeTestReport:
        """Execute every registered probe and return a report."""

        report = SmokeTestReport()

        probes = [
            ("NumPy basics", self._probe_numpy),
            ("Pandas basics", self._probe_pandas),
            ("OpenCV video decode", self._probe_opencv),
            ("CPU Farneback optical flow", self._probe_cpu_flow),
            ("GPU availability", self._probe_gpu_available),
            ("GPU windowed feature aggregation", self._probe_gpu_features),
            ("GPU pyramidal LK optical flow", self._probe_gpu_flow),
        ]

        for name, fn in probes:
            if on_line:
                on_line(f"Running: {name} ...")
            t0 = time.perf_counter()
            try:
                result = fn()
            except Exception as exc:
                result = SmokeTestResult(
                    name=name, passed=False, detail=f"Exception: {exc}"
                )
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            result.name = name
            report.results.append(result)
            if on_line:
                icon = "PASS" if result.passed else "FAIL"
                on_line(f"  [{icon}]  {name}  ({result.elapsed_ms:.1f} ms)")

        if on_line:
            on_line("")
            on_line(report.summary_text())

        return report

    # ── Individual probes ────────────────────────────────────────────────────

    @staticmethod
    def _probe_numpy() -> SmokeTestResult:
        import numpy as np

        a = np.random.randn(1000, 20).astype(np.float32)
        mean = a.mean(axis=0)
        std = a.std(axis=0)
        ok = bool(mean.shape == (20,) and std.shape == (20,) and np.all(np.isfinite(mean)))
        ver = np.version.version
        return SmokeTestResult(
            name="", passed=ok,
            detail=f"numpy {ver} — mean/std on (1000, 20) array",
        )

    @staticmethod
    def _probe_pandas() -> SmokeTestResult:
        import pandas as pd

        df = pd.DataFrame({"a": range(500), "b": range(500)})
        ok = len(df) == 500 and "a" in df.columns
        return SmokeTestResult(
            name="", passed=ok,
            detail=f"pandas {pd.__version__} — created 500-row DataFrame",
        )

    @staticmethod
    def _probe_opencv() -> SmokeTestResult:
        try:
            import cv2
        except ImportError:
            return SmokeTestResult(
                name="", passed=False,
                detail="opencv-python-headless not installed",
            )
        import numpy as np

        # Synthesize a tiny 3-frame 64×64 video in memory
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        ok = gray.shape == (64, 64) and gray.dtype == np.uint8
        return SmokeTestResult(
            name="", passed=ok,
            detail=f"OpenCV {cv2.__version__} — BGR→gray on 64×64 synthetic frame",
        )

    @staticmethod
    def _probe_cpu_flow() -> SmokeTestResult:
        try:
            import cv2
        except ImportError:
            return SmokeTestResult(
                name="", passed=False,
                detail="opencv-python-headless not installed",
            )
        import numpy as np

        h, w = 128, 128
        prev = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        curr = np.roll(prev, 3, axis=1)  # shift right by 3 pixels
        flow_out = np.zeros((h, w, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, flow_out, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(mag))
        ok = bool(flow.shape == (h, w, 2) and mean_mag > 0)
        return SmokeTestResult(
            name="", passed=ok,
            detail=f"CPU Farneback {h}×{w} — mean magnitude {mean_mag:.3f}",
        )

    @staticmethod
    def _probe_gpu_available() -> SmokeTestResult:
        try:
            import torch
        except ImportError:
            return SmokeTestResult(
                name="", passed=False,
                detail="PyTorch not installed",
            )
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            detail = f"torch {torch.__version__} — {name} ({mem:.1f} GB)"
        else:
            detail = f"torch {torch.__version__} — CUDA not available (CPU only)"
        return SmokeTestResult(name="", passed=has_cuda, detail=detail)

    @staticmethod
    def _probe_gpu_features() -> SmokeTestResult:
        try:
            from abel.utils.gpu_feature_ops import (
                gpu_available,
                windowed_feature_summary,
            )
        except ImportError:
            return SmokeTestResult(
                name="", passed=False,
                detail="gpu_feature_ops module not found",
            )
        import numpy as np

        if not gpu_available():
            return SmokeTestResult(
                name="", passed=False,
                detail="GPU not available — skipping windowed feature test",
            )

        n_frames = 5000
        n_features = 10
        window = 51
        stride = 1
        data = np.random.randn(n_frames, n_features).astype(np.float32)

        t0 = time.perf_counter()
        result = windowed_feature_summary(data, window, stride)
        gpu_ms = (time.perf_counter() - t0) * 1000

        # result is a dict of stat_name → (n_windows, n_features)
        first_key = next(iter(result))
        arr = result[first_key]
        ok = bool(arr.shape[0] > 0 and arr.shape[1] == n_features)
        n_stats = len(result)
        return SmokeTestResult(
            name="", passed=ok,
            detail=(
                f"GPU windowed stats: {n_frames} frames × {n_features} feats "
                f"→ {n_stats} stats × ({arr.shape[0]}, {arr.shape[1]}) in {gpu_ms:.1f} ms"
            ),
        )

    @staticmethod
    def _probe_gpu_flow() -> SmokeTestResult:
        try:
            from abel.utils.gpu_optical_flow import (
                compute_flow_batch_gpu,
                detect_flow_backend,
            )
        except ImportError:
            return SmokeTestResult(
                name="", passed=False,
                detail="gpu_optical_flow module not found",
            )
        import numpy as np

        backend = detect_flow_backend()
        if backend not in ("torch", "cv2_cuda"):
            return SmokeTestResult(
                name="", passed=False,
                detail=f"Flow backend is '{backend}' — no GPU path available",
            )

        h, w = 240, 320
        n = 50
        frames = []
        for i in range(n):
            f = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            frames.append(np.roll(f, i % 5, axis=1))  # synthetic motion
        prev = np.random.randint(0, 255, (h, w), dtype=np.uint8)

        t0 = time.perf_counter()
        flows = compute_flow_batch_gpu(frames, prev, gpu_batch_size=24)
        gpu_ms = (time.perf_counter() - t0) * 1000

        ok = len(flows) == n and flows[0].shape == (h, w, 2)
        per_frame = gpu_ms / n
        mag = float(np.mean(np.sqrt(flows[-1][..., 0] ** 2 + flows[-1][..., 1] ** 2)))
        return SmokeTestResult(
            name="", passed=ok,
            detail=(
                f"GPU pyramidal LK: {n} frames at {h}×{w} "
                f"in {gpu_ms:.0f} ms ({per_frame:.1f} ms/frame), "
                f"backend={backend}, mean_mag={mag:.3f}"
            ),
        )
