"""Export a finished run as a self-contained, hand-it-to-someone folder.

A run directory is organised for the *code* — one subfolder per analysis, plus
holdout manifests, a parquet substrate and intermediate arrays.  That is the
wrong shape for a person who wants "the figures and the data".  This flattens it
into::

    <dest>/
      ABEL_validation_report.pdf     the consolidated summary
      summary_report.html            same, in case the PDF is unwanted
      FINDINGS.md                    the findings as plain text
      figures/                       every PNG,  <analysis>__<name>.png
      data/                          every CSV,  <analysis>__<name>.csv
      INDEX.csv                      what each exported file is
      run_manifest.json  cells.parquet  report.html   (provenance / full dump)

Names are prefixed with the analysis they came from, so the flat folders stay
unambiguous when two analyses both ship a ``results.csv``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Copied through verbatim at the top level — provenance and the full dump.
_ROOT_FILES = ("run_manifest.json", "cells.parquet", "report.html",
               "summary_report.html", "ABEL_validation_report.pdf", "FINDINGS.md")

# Subdirectories that hold intermediates rather than results.
_SKIP_DIRS = {"arrays"}


@dataclass
class BundleResult:
    dest: Path
    n_figures: int
    n_tables: int
    pdf: Path | None
    index_path: Path

    def summary(self) -> str:
        pdf_txt = f"report PDF, " if self.pdf else ""
        return (f"Exported {pdf_txt}{self.n_figures} figures and {self.n_tables} data "
                f"tables → {self.dest}")


def _prefixed(subdir: str, name: str) -> str:
    """``discrimination`` + ``confusable_pairs.csv`` → ``discrimination__confusable_pairs.csv``."""
    return f"{subdir}__{name}" if subdir else name


def export_bundle(run_dir: str | Path, dest: str | Path) -> BundleResult:
    """Flatten ``run_dir`` into a shareable folder at ``dest``."""
    run_dir = Path(run_dir)
    dest = Path(dest)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    fig_dir = dest / "figures"
    data_dir = dest / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    n_fig = n_tab = 0

    for src in sorted(run_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(run_dir)
        # rel.parts[0] is the analysis subdir, or the filename itself at the root.
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        if top in _SKIP_DIRS:
            continue

        if src.suffix.lower() == ".png":
            out_name = _prefixed(top, src.name)
            shutil.copyfile(src, fig_dir / out_name)
            index.append({"exported_as": f"figures/{out_name}", "kind": "figure",
                          "analysis": top or "(root)", "source": str(rel)})
            n_fig += 1
        elif src.suffix.lower() == ".csv":
            out_name = _prefixed(top, src.name)
            shutil.copyfile(src, data_dir / out_name)
            index.append({"exported_as": f"data/{out_name}", "kind": "data",
                          "analysis": top or "(root)", "source": str(rel)})
            n_tab += 1

    pdf: Path | None = None
    for name in _ROOT_FILES:
        src = run_dir / name
        if not src.exists():
            continue
        shutil.copyfile(src, dest / name)
        index.append({"exported_as": name, "kind": "report",
                      "analysis": "(report)", "source": name})
        if src.suffix.lower() == ".pdf":
            pdf = dest / name

    index_path = dest / "INDEX.csv"
    pd.DataFrame(index).to_csv(index_path, index=False)

    return BundleResult(dest=dest, n_figures=n_fig, n_tables=n_tab,
                        pdf=pdf, index_path=index_path)
