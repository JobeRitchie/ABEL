"""Results-directory layout for a validation run (top-level, cross-project).

Results live under a root chosen by the caller (NOT inside any one project's
``derived/``) because a run spans many projects.  That root is normally a saved
session's ``runs/`` folder in the validation workspace — see
:mod:`abel.validation.workspace`::

    <session>/runs/run_<timestamp>/
      session.json          ← the setup this run was launched from (frozen)
      run_manifest.json
      holdout/<project>_holdout_manifest.json
      cells.parquet
      learning_curves/  cross_project/  ablation/  generalization/
      report.html
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from abel.storage.file_store import write_json


class ResultsStore:
    """Creates and writes into one timestamped validation-run directory."""

    SUBDIRS = ("holdout", "learning_curves", "cross_project", "ablation", "generalization", "arrays")

    def __init__(self, root: str | Path, run_id: str | None = None) -> None:
        self.root = Path(root)
        self.run_id = run_id or datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
        self.run_dir = self.root / self.run_id
        for sub in self.SUBDIRS:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── paths ──
    def sub(self, name: str) -> Path:
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cells_path(self) -> Path:
        return self.run_dir / "cells.parquet"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.html"

    # ── writes ──
    def write_manifest(self, manifest: dict) -> Path:
        p = self.run_dir / "run_manifest.json"
        write_json(p, manifest)
        return p

    def write_holdout_manifest(self, project_id: str, manifest: dict) -> Path:
        safe = str(project_id).replace("/", "_").replace("\\", "_")
        p = self.sub("holdout") / f"{safe}_holdout_manifest.json"
        write_json(p, manifest)
        return p

    def save_cells(self, df: pd.DataFrame) -> Path:
        df.to_parquet(self.cells_path, index=False)
        return self.cells_path

    def write_csv(self, df: pd.DataFrame, name: str, subdir: str = "") -> Path:
        target = (self.sub(subdir) if subdir else self.run_dir) / name
        df.to_csv(target, index=False)
        return target
