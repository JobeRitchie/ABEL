"""GPU architecture matching and PyTorch CUDA-index selection."""

from __future__ import annotations

import sys

from abel.services.torch_install_service import (
    Candidate,
    GpuEnvironment,
    GpuInfo,
    newest_wheel_version,
    parse_driver_cuda,
    parse_gpu_query,
    rank_candidates,
    tag_cuda_version,
)
from abel.utils.torch_cuda import arch_supported, parse_arch

CU126_ARCHS = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU128_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_parse_arch():
    assert parse_arch("sm_120") == ("sm", 12, 0, "")
    assert parse_arch("sm_86") == ("sm", 8, 6, "")
    assert parse_arch("compute_90a") == ("compute", 9, 0, "a")
    assert parse_arch("gfx90a") is None


def test_blackwell_not_covered_by_cu126():
    assert not arch_supported((12, 0), CU126_ARCHS)
    assert not arch_supported((10, 0), CU126_ARCHS)
    assert arch_supported((12, 0), CU128_ARCHS)


def test_same_major_binary_compat():
    # Ada (8.9) runs sm_86 cubins; Hopper 9.x does not run sm_86.
    assert arch_supported((8, 9), CU126_ARCHS)
    assert arch_supported((12, 1), CU128_ARCHS)
    assert not arch_supported((8, 0), ["sm_86"])


def test_ptx_forward_compat_and_arch_specific():
    assert arch_supported((12, 0), ["sm_80", "compute_90"])
    assert not arch_supported((9, 0), ["sm_100a"])
    assert arch_supported((10, 0), ["sm_100a"])
    assert not arch_supported((10, 3), ["sm_100a"])


def test_dropped_old_arch():
    assert not arch_supported((6, 1), CU128_ARCHS)
    assert arch_supported((6, 1), CU126_ARCHS)


def test_nvidia_smi_parsing():
    gpus = parse_gpu_query("NVIDIA GeForce RTX 5090, 12.0, 576.02\nTesla P100, 6.0, 576.02\n")
    assert [g.capability for g in gpus] == [(12, 0), (6, 0)]
    assert gpus[0].name == "NVIDIA GeForce RTX 5090"
    assert parse_gpu_query("Old GPU, , 470.1")[0].capability is None
    assert parse_driver_cuda("| NVIDIA-SMI 576.02   Driver Version: 576.02   CUDA Version: 12.9 |") == (12, 9)
    assert parse_driver_cuda("garbage") is None


def test_tag_cuda_version():
    assert tag_cuda_version("cu128") == (12, 8)
    assert tag_cuda_version("cu130") == (13, 0)
    assert tag_cuda_version("cu118") == (11, 8)
    assert tag_cuda_version("cpu") is None


LIVE = [
    Candidate("cu132", (13, 2), (2, 14, 1)),
    Candidate("cu130", (13, 0), (2, 14, 1)),
    Candidate("cu129", (12, 9), (2, 9, 0)),
    Candidate("cu128", (12, 8), (2, 11, 0)),
    Candidate("cu126", (12, 6), (2, 14, 1)),
]


def _env(cap, driver_cuda):
    return GpuEnvironment([GpuInfo("gpu", cap, "x")], driver_cuda)


def test_blackwell_on_cuda_12_9_driver_skips_cu126():
    ranked = [c.tag for c in rank_candidates(_env((12, 0), (12, 9)), LIVE)]
    assert ranked == ["cu128", "cu129"]  # cu126 cannot target Blackwell at all


def test_blackwell_on_new_driver_prefers_newest():
    ranked = [c.tag for c in rank_candidates(_env((12, 0), (13, 2)), LIVE)]
    assert ranked[0] == "cu132"


def test_ada_on_cuda_12_7_driver_uses_cu126():
    ranked = [c.tag for c in rank_candidates(_env((8, 9), (12, 7)), LIVE)]
    assert ranked == ["cu126"]


def test_pascal_prefers_cu126_even_on_new_driver():
    ranked = [c.tag for c in rank_candidates(_env((6, 1), (13, 2)), LIVE)]
    assert ranked[0] == "cu126"


def test_future_gpu_prefers_newest_toolkit():
    ranked = [c.tag for c in rank_candidates(_env((13, 0), (13, 4)), LIVE)]
    assert ranked == ["cu132", "cu130"]


def test_driver_older_than_every_index_falls_back_to_same_major():
    ranked = [c.tag for c in rank_candidates(_env((8, 6), (12, 2)), LIVE)]
    assert ranked and all(t.startswith("cu12") for t in ranked)


def test_newest_wheel_version_filters_python_and_platform():
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = "win_amd64" if sys.platform == "win32" else "manylinux_2_28_x86_64"
    html = "\n".join([
        f'<a href="/whl/cu128/torch-2.7.0%2Bcu128-{py}-{py}-{plat}.whl#sha256=1">x</a>',
        f'<a href="/whl/cu128/torch-2.8.0%2Bcu128-{py}-{py}-{plat}.whl#sha256=2">x</a>',
        f'<a href="/whl/cu128/torch-2.9.0%2Bcu128-cp27-cp27-{plat}.whl">x</a>',
        f'<a href="/whl/cu128/torch-2.9.0.dev20260101%2Bcu128-{py}-{py}-{plat}.whl">x</a>',
        f'<a href="/whl/cu128/torch-3.0.0%2Bcu128-{py}-{py}-macosx_11_0_arm64.whl">x</a>',
    ])
    assert newest_wheel_version(html) == (2, 8, 0)
    assert newest_wheel_version("") is None


# ── Launcher step (abel._gpu_setup) ─────────────────────────────────────────

import pytest

import abel._gpu_setup as gpu_setup
from abel.services.torch_install_service import TorchSetupResult

BLACKWELL = GpuEnvironment([GpuInfo("RTX 5090", (12, 0), "576.02")], (12, 9))


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    calls = {"probe": 0, "repair": 0}
    state = {"usable": False, "repair_ok": True}
    monkeypatch.setattr(gpu_setup, "_STAMP", tmp_path / "stamp.json")
    monkeypatch.setattr(gpu_setup, "installed_version", lambda pkg: "2.11.0+cu126")
    monkeypatch.setattr(gpu_setup, "detect_gpu_environment", lambda: BLACKWELL)

    def probe():
        calls["probe"] += 1
        return {"usable": state["usable"], "reason": "no kernels for sm_120"}

    class FakeInstaller:
        def ensure(self, force, on_line, env):
            calls["repair"] += 1
            return TorchSetupResult(state["repair_ok"], [], "", state["repair_ok"], "cu128")

    monkeypatch.setattr(gpu_setup, "probe_in_subprocess", probe)
    monkeypatch.setattr(gpu_setup, "TorchInstaller", FakeInstaller)
    return calls, state


def test_launcher_repairs_unusable_gpu_once(launcher):
    calls, _ = launcher
    assert gpu_setup.run() is True
    assert calls == {"probe": 1, "repair": 1}
    assert gpu_setup.run() is True  # stamp says usable: no probe, no repair
    assert calls == {"probe": 1, "repair": 1}


def test_launcher_does_not_retry_failed_repair(launcher):
    calls, state = launcher
    state["repair_ok"] = False
    assert gpu_setup.run() is False
    assert gpu_setup.run() is False
    assert calls["repair"] == 1


def test_launcher_rechecks_after_driver_change(launcher, monkeypatch):
    calls, state = launcher
    state["usable"] = True
    gpu_setup.run()
    newer = GpuEnvironment([GpuInfo("RTX 5090", (12, 0), "580.10")], (13, 0))
    monkeypatch.setattr(gpu_setup, "detect_gpu_environment", lambda: newer)
    gpu_setup.run()
    assert calls == {"probe": 2, "repair": 0}


def test_launcher_skips_without_torch_or_gpu(launcher, monkeypatch):
    calls, _ = launcher
    monkeypatch.setattr(gpu_setup, "detect_gpu_environment", lambda: GpuEnvironment())
    assert gpu_setup.run() is True
    monkeypatch.setattr(gpu_setup, "installed_version", lambda pkg: None)
    assert gpu_setup.run() is True
    assert calls == {"probe": 0, "repair": 0}
