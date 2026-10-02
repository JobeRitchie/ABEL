"""Launcher step: make sure the installed PyTorch can use this machine's GPU.

Used by ``run_abel.bat`` before the app starts, because Windows locks torch's
DLLs once the app has imported it, so a broken build can only be swapped out
while ABEL is closed.

``python -m abel._gpu_setup --auto`` does nothing when torch is not installed
(it is optional; Dependencies > Install GPU PyTorch adds it). Otherwise it
checks the build against the GPU and, when the GPU is present but unusable
(e.g. a cu126 wheel on an RTX 50-series card), installs a matching build.

A stamp under ``.venv`` records the GPU/driver/torch combination that was last
checked, so routine launches cost one ``nvidia-smi`` call. A failed repair is
also recorded and not retried until something changes, so an unsupported GPU
does not trigger a multi-GB download on every launch. Never blocks the launch:
always exits 0 unless ``--strict``. Set ``ABEL_SKIP_GPU_SETUP=1`` to skip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from abel.services.torch_install_service import (
    TorchInstaller,
    detect_gpu_environment,
    installed_version,
    probe_in_subprocess,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAMP = _PROJECT_ROOT / ".venv" / ".abel_gpu_stamp.json"


def _say(text: str) -> None:
    print(f"[GPU] {text}", flush=True)


def _read_stamp() -> dict:
    try:
        return json.loads(_STAMP.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_stamp(data: dict) -> None:
    try:
        _STAMP.parent.mkdir(parents=True, exist_ok=True)
        _STAMP.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _fingerprint(env) -> str:
    return "||".join([
        env.fingerprint(),
        f"torch={installed_version('torch')}",
        f"torchvision={installed_version('torchvision')}",
        f"py={sys.version_info.major}.{sys.version_info.minor}",
    ])


def run(force: bool = False) -> bool:
    if not installed_version("torch"):
        return True
    env = detect_gpu_environment()
    if not env.has_nvidia:
        return True

    fp = _fingerprint(env)
    stamp = _read_stamp()
    if not force and stamp.get("fingerprint") == fp:
        if stamp.get("usable"):
            return True
        if stamp.get("repair_attempted"):
            _say(f"PyTorch cannot use this GPU ({stamp.get('reason', 'unknown reason')}); using CPU.")
            return False

    _say("Checking that PyTorch can use the GPU (first launch after a change)...")
    status = probe_in_subprocess()
    if status.get("usable"):
        _say(f"OK: torch {status.get('torch_version')} on {status.get('device_name')}.")
        _write_stamp({"fingerprint": fp, "usable": True})
        return True

    _say(f"PyTorch cannot use the GPU: {status.get('reason')}")
    _say("Installing a PyTorch build that matches this GPU. This downloads about 2-3 GB once.")
    result = TorchInstaller().ensure(force=True, on_line=_say, env=env)
    # Re-fingerprint: the torch version changed if the repair installed something.
    _write_stamp({
        "fingerprint": _fingerprint(env),
        "usable": bool(result.gpu_usable),
        "repair_attempted": True,
        "reason": "" if result.gpu_usable else str(status.get("reason", "")),
    })
    return bool(result.gpu_usable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--auto", action="store_true", help="check and repair if needed (launcher mode)")
    parser.add_argument("--force", action="store_true", help="ignore the stamp and re-check")
    parser.add_argument("--strict", action="store_true", help="exit 1 when the GPU stays unusable")
    args = parser.parse_args()
    if os.environ.get("ABEL_SKIP_GPU_SETUP", "").strip() in ("1", "true", "yes"):
        return 0
    try:
        ok = run(force=args.force)
    except Exception as exc:  # never block the launch
        _say(f"GPU check skipped after an error: {exc}")
        ok = False
    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
