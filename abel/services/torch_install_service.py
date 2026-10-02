"""Install the PyTorch build that matches this machine's NVIDIA GPU and driver.

PyTorch publishes one wheel per CUDA toolkit (cu126, cu128, cu130, ...), and
each toolkit compiles kernels for a different range of GPU architectures:

* toolkits before CUDA 12.8 stop at Hopper (cc 9.x), so Blackwell cards
  (RTX 50-series, B200; cc 10.x/12.x) get "no kernel image" errors;
* newer toolkits drop old cards (PyTorch's 12.8/12.9 builds start at cc 7.0,
  CUDA 13 builds at cc 7.5).

No single index works for every GPU, and the set of live indexes changes with
each PyTorch release, so nothing here is pinned.  The installer:

1. reads each GPU's compute capability and the driver's max CUDA version from
   ``nvidia-smi``;
2. lists the CUDA indexes on download.pytorch.org that have a wheel for this
   Python and platform (a hard-coded fallback list is used when offline);
3. drops toolkits newer than the driver or known not to cover the GPU, and
   ranks the rest newest-first;
4. installs the top candidate and checks it in a fresh process by running a
   CUDA kernel (:mod:`abel.utils.torch_cuda`). If that fails it moves to the
   next candidate (at most three), then to the nightly build of the best one,
   which is where support for brand-new architectures usually appears first.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

INDEX_ROOT = "https://download.pytorch.org/whl"
_HTTP_TIMEOUT_S = 15

# Used only when the live index listing cannot be fetched.
_FALLBACK_TAGS = ["cu132", "cu130", "cu128", "cu126", "cu118"]
_MAX_STABLE_ATTEMPTS = 3

LineCallback = Callable[[str], None]


@dataclass
class GpuInfo:
    name: str
    capability: tuple[int, int] | None
    driver_version: str


@dataclass
class GpuEnvironment:
    gpus: list[GpuInfo] = field(default_factory=list)
    driver_cuda: tuple[int, int] | None = None  # max CUDA the driver supports

    @property
    def has_nvidia(self) -> bool:
        return bool(self.gpus)

    def fingerprint(self) -> str:
        parts = [f"{g.name}|{g.capability}|{g.driver_version}" for g in self.gpus]
        return ";".join(parts) or "no-nvidia-gpu"


@dataclass
class Candidate:
    tag: str                          # "cu128", or "cpu"
    cuda: tuple[int, int] | None
    torch_version: tuple[int, ...] | None = None
    nightly: bool = False

    @property
    def index_url(self) -> str:
        if self.nightly:
            return f"{INDEX_ROOT}/nightly/{self.tag}"
        return f"{INDEX_ROOT}/{self.tag}"

    def label(self) -> str:
        ver = ".".join(map(str, self.torch_version)) if self.torch_version else "latest"
        return f"{self.tag}{' nightly' if self.nightly else ''} (torch {ver})"


@dataclass
class TorchSetupResult:
    success: bool
    command: list[str]
    output: str
    gpu_usable: bool = False
    candidate: str = ""


# ── Detection ───────────────────────────────────────────────────────────────

def _nvidia_smi_path() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if platform.system() == "Windows":
        for base in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI",
        ):
            exe = base / "nvidia-smi.exe"
            if exe.exists():
                return str(exe)
    return None


def _run_quiet(cmd: list[str], timeout: float = 15) -> str:
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, **kwargs,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _parse_version_pair(text: str) -> tuple[int, int] | None:
    m = re.match(r"\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_gpu_query(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3 or not cols[0]:
            continue
        gpus.append(GpuInfo(cols[0], _parse_version_pair(cols[1]), cols[2]))
    return gpus


def parse_driver_cuda(smi_header: str) -> tuple[int, int] | None:
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_header or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def detect_gpu_environment() -> GpuEnvironment:
    exe = _nvidia_smi_path()
    if exe is None:
        return GpuEnvironment()
    try:
        query = _run_quiet(
            [exe, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
        )
        if not query.strip():
            # Drivers older than ~510 do not know compute_cap.
            query = _run_quiet([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"])
            query = "\n".join(
                f"{p[0]},,{p[-1]}" for p in (l.split(",") for l in query.splitlines()) if len(p) >= 2
            )
        header = _run_quiet([exe])
    except Exception:
        return GpuEnvironment()
    return GpuEnvironment(gpus=parse_gpu_query(query), driver_cuda=parse_driver_cuda(header))


# ── Candidate indexes ───────────────────────────────────────────────────────

def tag_cuda_version(tag: str) -> tuple[int, int] | None:
    """``"cu128"`` -> ``(12, 8)``; ``"cu130"`` -> ``(13, 0)``; ``"cu92"`` -> ``(9, 2)``."""
    m = re.fullmatch(r"cu(\d+)", tag)
    if not m:
        return None
    digits = m.group(1)
    return int(digits[:-1]), int(digits[-1])


def toolkit_arch_range(cuda: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Compute-capability range PyTorch's wheels for a CUDA toolkit cover.

    Returns ``(floor, ceiling)``; ``ceiling`` is ``None`` when unknown (toolkits
    newer than this table, which may already cover GPUs released after it).
    The ceiling is inclusive at the major-version level: a cc 9.0 cubin also
    runs on any later 9.x device.
    """
    if cuda < (12, 8):
        return (5, 0), (9, 9)
    if cuda < (13, 0):
        return (7, 0), (12, 9)
    return (7, 5), None


def _wheel_matches_platform(name: str) -> bool:
    """True if a wheel filename fits this interpreter (cpXY ABI) and OS/CPU."""
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if f"-{py}-{py}-" not in name:
        return False
    machine = platform.machine().lower()
    system = platform.system()
    if system == "Windows":
        return name.endswith("win_arm64.whl" if "arm" in machine else "win_amd64.whl")
    if system == "Linux":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return "linux" in name and name.endswith(f"_{arch}.whl")
    return True


def newest_wheel_version(index_html: str, package: str = "torch") -> tuple[int, ...] | None:
    """Newest stable ``package`` wheel in a PEP 503 page for this Python/platform."""
    best = None
    for href in re.findall(r'href="([^"]+)"', index_html):
        name = urllib.request.unquote(href.split("/")[-1].split("#")[0])
        if not name.startswith(f"{package}-") or not _wheel_matches_platform(name):
            continue
        m = re.match(rf"{package}-(\d+(?:\.\d+)*)(\+[^-]+)?-", name)
        if not m:
            continue
        ver = tuple(int(p) for p in m.group(1).split("."))
        if best is None or ver > best:
            best = ver
    return best


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "abel-torch-setup"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_live_candidates(log: LineCallback | None = None) -> list[Candidate] | None:
    """CUDA indexes that currently ship torch + torchvision for this Python/platform."""
    try:
        root = _fetch(f"{INDEX_ROOT}/")
    except Exception as exc:
        if log:
            log(f"Could not list PyTorch indexes ({exc}); using built-in list.")
        return None
    tags = sorted(
        set(re.findall(r'href="(cu\d+)/?"', root)),
        key=lambda t: tag_cuda_version(t) or (0, 0),
        reverse=True,
    )
    out = []
    for tag in tags:
        cuda = tag_cuda_version(tag)
        if cuda is None or cuda < (11, 8):
            continue
        try:
            ver = newest_wheel_version(_fetch(f"{INDEX_ROOT}/{tag}/torch/"))
            if ver is None:
                continue
            if newest_wheel_version(_fetch(f"{INDEX_ROOT}/{tag}/torchvision/"), "torchvision") is None:
                continue
        except Exception:
            continue
        out.append(Candidate(tag, cuda, ver))
    return out


def rank_candidates(env: GpuEnvironment, available: list[Candidate]) -> list[Candidate]:
    """Order CUDA indexes from most to least likely to work on ``env``'s GPUs."""
    caps = [g.capability for g in env.gpus if g.capability]
    lo = min(caps) if caps else None
    hi = max(caps) if caps else None

    def fits_driver(c: Candidate) -> bool:
        return env.driver_cuda is None or c.cuda <= env.driver_cuda

    def too_old(c: Candidate) -> bool:
        # A toolkit cannot emit code for an architecture released after it.
        ceiling = toolkit_arch_range(c.cuda)[1]
        return hi is not None and ceiling is not None and hi[0] > ceiling[0]

    def covers(c: Candidate) -> bool:
        # The floor is PyTorch's build choice, not a hard limit, so a build
        # below it stays as a last resort rather than being dropped.
        return lo is None or lo >= toolkit_arch_range(c.cuda)[0]

    pool = [c for c in available if fits_driver(c)]
    if not pool and env.driver_cuda is not None:
        # CUDA minor-version compatibility: a 12.x runtime can run on any 12.x driver.
        pool = [c for c in available if c.cuda[0] == env.driver_cuda[0]]
    pool = [c for c in pool if not too_old(c)]
    good = [c for c in pool if covers(c)]
    rest = [c for c in pool if not covers(c)]
    key = lambda c: (c.torch_version or (0,), c.cuda)
    return sorted(good, key=key, reverse=True) + sorted(rest, key=key, reverse=True)


# ── Installation ────────────────────────────────────────────────────────────

def installed_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def probe_in_subprocess(python: str = sys.executable, timeout: float = 300) -> dict:
    """Run :mod:`abel.utils.torch_cuda` in a clean interpreter and return its JSON."""
    try:
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            [python, "-m", "abel.utils.torch_cuda"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(Path(__file__).resolve().parents[2]), **kwargs,
        )
    except Exception as exc:
        return {"torch_installed": False, "usable": False, "reason": f"probe failed: {exc}"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
    return {"torch_installed": False, "usable": False, "reason": f"probe crashed: {tail[0]}"}


class TorchInstaller:
    def __init__(self, python: str = sys.executable) -> None:
        self.python = python
        self._commands: list[list[str]] = []
        self._lines: list[str] = []

    def _log(self, on_line: LineCallback | None, text: str) -> None:
        self._lines.append(text)
        if on_line is not None:
            on_line(text)

    def _pip(self, args: list[str], on_line: LineCallback | None) -> bool:
        cmd = [self.python, "-m", "pip", *args]
        self._commands.append(cmd)
        self._log(on_line, "$ " + " ".join(args))
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1, **kwargs,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                self._log(on_line, line)
        proc.wait()
        return proc.returncode == 0

    def _install(self, cand: Candidate, on_line: LineCallback | None) -> bool:
        pkgs = ["torch", "torchvision"]
        if installed_version("torchaudio"):
            pkgs.append("torchaudio")  # keep it ABI-matched with the new torch
        index = ["--index-url", cand.index_url]
        pre = ["--pre"] if cand.nightly else []
        # Step 1 replaces only the torch packages from the CUDA index (pip
        # downloads before uninstalling, so a failed download leaves the old
        # build in place). Step 2 adds any dependency the new build needs from
        # PyPI: the PyTorch index mirrors only some packages and has broken
        # metadata for others. Torch is already satisfied by then, so pip does
        # not swap in PyPI's CPU-only wheel.
        if not self._pip(["install", "--force-reinstall", "--no-deps", *pre, *pkgs, *index], on_line):
            return False
        return self._pip(["install", *pre, *pkgs], on_line)

    def plan(self, env: GpuEnvironment, on_line: LineCallback | None = None) -> list[Candidate]:
        if not env.has_nvidia:
            if platform.system() == "Darwin":
                return [Candidate("pypi", None)]
            return [Candidate("cpu", None)]
        live = list_live_candidates(lambda t: self._log(on_line, t))
        if live is None:
            live = [Candidate(t, tag_cuda_version(t)) for t in _FALLBACK_TAGS]
        ranked = rank_candidates(env, live)
        if not ranked:
            return []
        # Each attempt is a 2-3 GB download, so try only the best few.
        plan = ranked[:_MAX_STABLE_ATTEMPTS]
        plan.append(Candidate(ranked[0].tag, ranked[0].cuda, None, nightly=True))
        return plan

    def ensure(
        self,
        force: bool = False,
        on_line: LineCallback | None = None,
        env: GpuEnvironment | None = None,
    ) -> TorchSetupResult:
        """Install or repair torch/torchvision so they can use this machine's GPU."""
        env = env or detect_gpu_environment()
        if env.has_nvidia:
            for g in env.gpus:
                cap = f"{g.capability[0]}.{g.capability[1]}" if g.capability else "unknown"
                self._log(on_line, f"GPU: {g.name} (compute capability {cap}), driver {g.driver_version}")
            if env.driver_cuda:
                self._log(on_line, f"Driver supports CUDA up to {env.driver_cuda[0]}.{env.driver_cuda[1]}")
        else:
            self._log(on_line, "No NVIDIA GPU detected; installing the CPU build of PyTorch.")

        if not force and installed_version("torch") and installed_version("torchvision"):
            status = probe_in_subprocess(self.python)
            if status.get("usable") or (status.get("torch_installed") and not env.has_nvidia):
                self._log(on_line, f"PyTorch {status.get('torch_version')} already works here; nothing to do.")
                return self._result(True, status.get("usable", False), "existing")
            self._log(on_line, f"Current PyTorch cannot use the GPU: {status.get('reason')}")

        candidates = self.plan(env, on_line)
        if not candidates:
            self._log(on_line, "No PyTorch build on download.pytorch.org fits this driver. Update the NVIDIA driver and retry.")
            return self._result(False, False, "")

        for cand in candidates:
            self._log(on_line, f"Trying PyTorch {cand.label()} from {cand.index_url}")
            if cand.tag == "pypi":
                ok = self._pip(["install", "--upgrade", "torch", "torchvision"], on_line)
            else:
                ok = self._install(cand, on_line)
            if not ok:
                self._log(on_line, f"Install from {cand.tag} failed; trying the next option.")
                continue
            status = probe_in_subprocess(self.python)
            if not env.has_nvidia:
                return self._result(bool(status.get("torch_installed")), False, cand.label())
            if status.get("usable"):
                self._log(
                    on_line,
                    f"Verified: torch {status.get('torch_version')} runs CUDA kernels on {status.get('device_name')}.",
                )
                return self._result(True, True, cand.label())
            self._log(on_line, f"Installed {cand.label()} but the GPU check failed: {status.get('reason')}")

        if env.has_nvidia:
            self._log(
                on_line,
                "No available PyTorch build could run on this GPU. ABEL will use the CPU. "
                "Updating the NVIDIA driver usually unlocks newer builds.",
            )
        else:
            self._log(on_line, "PyTorch could not be installed; see the messages above.")
        return self._result(False, False, "")

    def _result(self, success: bool, usable: bool, label: str) -> TorchSetupResult:
        cmd = self._commands[-1] if self._commands else []
        return TorchSetupResult(success, cmd, "\n".join(self._lines), usable, label)
