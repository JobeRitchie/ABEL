"""Decide whether the installed PyTorch can actually run kernels on this GPU.

``torch.cuda.is_available()`` only says a driver and a device exist.  It stays
True when the PyTorch build has no kernels for the GPU's architecture, e.g. a
cu126 wheel on an RTX 50-series (Blackwell, sm_120) card.  The first real op
then fails with "no kernel image is available for execution on the device".

:func:`cuda_usable` is the check every GPU path in ABEL should use.  It
compares the device's compute capability against the build's arch list and
then runs a tiny kernel, so an incompatible build falls back to CPU up front
instead of failing mid-pipeline.

Run ``python -m abel.utils.torch_cuda`` to print the status as JSON.  The GPU
installer uses that to verify a freshly installed build in a clean process.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("abel")

_ARCH_RE = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")


@dataclass
class CudaStatus:
    torch_installed: bool = False
    torch_version: str = ""
    cuda_build: str = ""          # CUDA version the wheel was built for, "" for CPU wheels
    device_present: bool = False  # driver + device visible to torch
    device_name: str = ""
    capability: tuple[int, int] | None = None
    arch_list: list[str] = field(default_factory=list)
    usable: bool = False
    reason: str = ""


def parse_arch(entry: str) -> tuple[str, int, int, str] | None:
    """``"sm_120"`` -> ``("sm", 12, 0, "")``; ``"compute_90a"`` -> ``("compute", 9, 0, "a")``."""
    m = _ARCH_RE.match(entry.strip().lower())
    if not m or len(m.group(2)) < 2:
        return None
    digits = m.group(2)
    return m.group(1), int(digits[:-1]), int(digits[-1]), m.group(3)


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """True if a build compiled for ``arch_list`` has code that runs on ``capability``.

    CUDA compatibility rules: a cubin (``sm_XY``) runs on devices of the same
    major version with minor >= Y; PTX (``compute_XY``) is JIT-compiled by the
    driver for any device >= X.Y.  Arch-specific targets (``sm_90a``) only run
    on that exact architecture.  An empty list means the build did not report
    one, so this check cannot rule it out.
    """
    if not arch_list:
        return True
    major, minor = capability
    for entry in arch_list:
        parsed = parse_arch(entry)
        if parsed is None:
            continue
        kind, a_major, a_minor, suffix = parsed
        if suffix:
            if (a_major, a_minor) == (major, minor):
                return True
        elif kind == "sm":
            if a_major == major and a_minor <= minor:
                return True
        elif (a_major, a_minor) <= (major, minor):
            return True
    return False


def _probe() -> CudaStatus:
    status = CudaStatus()
    try:
        import torch
    except Exception as exc:
        status.reason = f"PyTorch not importable: {exc}"
        return status
    status.torch_installed = True
    status.torch_version = str(getattr(torch, "__version__", ""))
    status.cuda_build = str(getattr(torch.version, "cuda", None) or "")
    if not status.cuda_build:
        status.reason = "PyTorch is a CPU-only build"
        return status
    try:
        if not torch.cuda.is_available():
            status.reason = "no CUDA device visible (driver missing or no NVIDIA GPU)"
            return status
        status.device_present = True
        status.device_name = torch.cuda.get_device_name(0)
        status.capability = tuple(torch.cuda.get_device_capability(0))
        status.arch_list = list(torch.cuda.get_arch_list())
    except Exception as exc:
        status.reason = f"CUDA initialisation failed: {exc}"
        return status

    cap = status.capability
    if not arch_supported(cap, status.arch_list):
        status.reason = (
            f"PyTorch {status.torch_version} has no kernels for {status.device_name} "
            f"(compute capability {cap[0]}.{cap[1]}); build targets {', '.join(status.arch_list)}"
        )
        return status
    try:
        x = torch.arange(8, device="cuda", dtype=torch.float32)
        total = float((x * 2.0).sum().item())
        m = torch.ones(16, 16, device="cuda")
        mm = float((m @ m)[0, 0].item())
        torch.cuda.synchronize()
        if total != 56.0 or mm != 16.0:
            status.reason = "CUDA test kernel returned wrong results"
            return status
    except Exception as exc:
        status.reason = f"CUDA test kernel failed: {str(exc).splitlines()[0]}"
        return status
    status.usable = True
    return status


_lock = threading.Lock()
_cached: CudaStatus | None = None


def cuda_status(refresh: bool = False) -> CudaStatus:
    """Probe once per process (thread-safe) and log the outcome."""
    global _cached
    with _lock:
        if _cached is None or refresh:
            _cached = _probe()
            s = _cached
            if s.usable:
                logger.info(
                    "CUDA usable: %s (cc %d.%d), torch %s built for CUDA %s",
                    s.device_name, s.capability[0], s.capability[1],
                    s.torch_version, s.cuda_build,
                )
            elif s.device_present:
                logger.warning(
                    "GPU present but unusable by PyTorch, using CPU instead: %s. "
                    "Relaunch ABEL with run_abel.bat to install a matching PyTorch build, "
                    "or use Dependencies > Install GPU PyTorch.",
                    s.reason,
                )
            else:
                logger.info("CUDA not used: %s", s.reason)
        return _cached


def cuda_usable() -> bool:
    """True only if PyTorch can run kernels on CUDA device 0."""
    return cuda_status().usable


def main() -> int:
    s = _probe()
    data = asdict(s)
    data["capability"] = list(s.capability) if s.capability else None
    sys.stdout.write(json.dumps(data) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
