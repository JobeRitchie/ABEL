"""Safe file I/O helpers with atomic writes and backups."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# PyYAML's pure-Python scanner is the bottleneck on any large config — a 1 MB
# environment_rois.yaml (hand-drawn polygon ROIs are thousands of vertices)
# takes seconds to parse, and the ROI tab parses it several times per subject
# switch.  libyaml is 5-10x faster and ships with the wheel; fall back to the
# Python implementation only where the C extension is genuinely absent.
try:
    from yaml import CSafeDumper as _SafeDumper, CSafeLoader as _SafeLoader
except ImportError:  # pragma: no cover - depends on the installed PyYAML build
    from yaml import SafeDumper as _SafeDumper, SafeLoader as _SafeLoader


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        handle.write(content)
        temp_name = handle.name
    Path(temp_name).replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, default=str))


def atomic_write_parquet(df: "Any", path: Path, **to_parquet_kwargs: Any) -> None:
    """Write *df* to *path* as parquet atomically.

    The DataFrame is first written to a uniquely-named temporary file in the
    destination directory, then moved into place with an atomic ``os.replace``.
    If the process is interrupted mid-write (e.g. the app is closed during a
    long run) only the temp file is ever partial — the canonical parquet at
    *path* is never left truncated/footerless, which would otherwise make it
    unreadable ("Parquet magic bytes not found in footer").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".parquet.tmp",
    )
    import os

    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, **to_parquet_kwargs)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    text = yaml.dump(data, Dumper=_SafeDumper, sort_keys=False, allow_unicode=False)
    atomic_write_text(path, text)


def read_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_SafeLoader)
    return loaded or {}


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.{timestamp}.bak"
    shutil.copy2(path, backup_path)
    return backup_path
