"""Dependency scanning and package installation orchestration."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from abel.models.schemas import DependencySpec


@dataclass
class DependencyActionResult:
    success: bool
    command: list[str]
    output: str


class DependencyService:
    """Inspects and manages optional dependency sets."""

    def __init__(self) -> None:
        self._dependency_specs = [
            DependencySpec(
                package="PySide6",
                purpose="Desktop GUI",
                required_version=">=6.7",
                tier="tier1",
            ),
            DependencySpec(
                package="pydantic",
                purpose="Typed schemas",
                required_version=">=2.7",
                tier="tier1",
            ),
            DependencySpec(
                package="numpy",
                purpose="Numerical ops",
                required_version=">=1.26",
                tier="tier1",
            ),
            DependencySpec(
                package="pandas",
                purpose="Tabular data",
                required_version=">=2.2",
                tier="tier1",
            ),
            DependencySpec(
                package="PyYAML",
                purpose="YAML config",
                required_version=">=6.0",
                tier="tier1",
            ),
            DependencySpec(
                package="python-docx",
                purpose="Word report export",
                required_version=">=1.1",
                tier="tier1",
            ),
            DependencySpec(
                package="openpyxl",
                purpose="Excel workbook export",
                required_version=">=3.1",
                tier="tier1",
            ),
            DependencySpec(
                package="opencv-python-headless",
                purpose="Video decoding and processing",
                required_version=">=4.8",
                tier="tier2_preprocessing",
            ),
            DependencySpec(
                package="scipy",
                purpose="Signal processing",
                required_version=">=1.13",
                tier="tier2_preprocessing",
            ),
            DependencySpec(
                package="imageio",
                purpose="Video IO alternatives",
                required_version=">=2.35",
                tier="tier2_preprocessing",
            ),
            DependencySpec(
                package="imageio-ffmpeg",
                purpose="FFmpeg bindings",
                required_version=">=0.5",
                tier="tier2_preprocessing",
            ),
            DependencySpec(
                package="scikit-learn",
                purpose="Behavior feature modeling",
                required_version=">=1.5",
                tier="tier2_preprocessing",
            ),
            DependencySpec(
                package="xgboost",
                purpose="GPU-capable gradient boosting for behavior models",
                required_version=">=2.0",
                tier="tier2_acceleration",
            ),
            DependencySpec(
                package="torch",
                purpose="GPU-accelerated feature aggregation and windowed stats (CUDA-capable on Windows)",
                required_version=">=2.0",
                tier="tier2_acceleration",
            ),
            DependencySpec(
                package="torchvision",
                purpose="Pretrained video models for fusion inference",
                required_version=">=0.19",
                tier="tier2_acceleration",
            ),
            DependencySpec(
                package="hmmlearn",
                purpose="Hidden Markov Model analysis in Motif tab",
                required_version=">=0.3",
                tier="tier2_analytics",
            ),
            DependencySpec(
                package="umap-learn",
                purpose="UMAP dimensionality reduction for sequence clustering in Motif tab",
                required_version=">=0.5",
                tier="tier2_analytics",
            ),
            DependencySpec(
                package="hdbscan",
                purpose="Density-based clustering for sequence clustering in Motif tab",
                required_version=">=0.8",
                tier="tier2_analytics",
            ),

        ]

    def scan(self) -> list[DependencySpec]:
        scanned: list[DependencySpec] = []
        for spec in self._dependency_specs:
            installed_version = self._get_version(spec.package)
            status = "installed" if installed_version else "missing"
            scanned.append(
                spec.model_copy(
                    update={
                        "installed_version": installed_version,
                        "status": status,
                    }
                )
            )
        return scanned

    def install_packages(
        self,
        packages: list[str],
        on_line: "Callable[[str], None] | None" = None,
    ) -> DependencyActionResult:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                lines.append(line)
                if on_line is not None:
                    on_line(line)
        process.wait()
        return DependencyActionResult(
            success=process.returncode == 0,
            command=command,
            output="\n".join(lines),
        )

    def uninstall_package(
        self,
        package: str,
        on_line: "Callable[[str], None] | None" = None,
    ) -> DependencyActionResult:
        command = [sys.executable, "-m", "pip", "uninstall", "-y", package]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                lines.append(line)
                if on_line is not None:
                    on_line(line)
        process.wait()
        return DependencyActionResult(
            success=process.returncode == 0,
            command=command,
            output="\n".join(lines),
        )

    def recommended_preprocessing(self) -> list[str]:
        return ["opencv-python-headless", "scipy", "imageio", "imageio-ffmpeg"]

    def recommended_gpu_modeling(self) -> list[str]:
        return ["xgboost"]

    def recommended_windows_cuda_fusion(self) -> list[str]:
        if platform.system().lower().startswith("win"):
            return [
                "torch",
                "torchvision",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        return ["torch", "torchvision"]

    def recommended_all(self) -> list[str]:
        return [
            *self.recommended_preprocessing(),
            *self.recommended_gpu_modeling(),
            *self.recommended_windows_cuda_fusion(),
            "scikit-learn>=1.5",
            "python-docx",
            "openpyxl",
            "hmmlearn",
            "umap-learn",
            "hdbscan",
        ]

    def recommended_science_stack(self) -> list[str]:
        return [
            "opencv-python",
            "scipy",
            "imageio",
            "imageio-ffmpeg",
            *self.recommended_gpu_modeling(),
            "torch",
            "torchvision",
            "scikit-learn>=1.5",
        ]

    @staticmethod
    def _get_version(package: str) -> str | None:
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            return None
