"""Free disk space in a finished project without breaking it.

A project accumulates large files that nothing needs once the work is done:
a full copy of the training set for every retrain, one-off repair backups, scratch caches and the review
clips.  :func:`scan_project` measures each category and :func:`cleanse_project`
deletes the ones the user picks.

Everything ABEL needs to keep working is left alone: raw data, extracted
pose/context/R3D features, models, labels, review decisions, the live training
set, temporal-refinement results and config.  Each category says what (if
anything) has to be rebuilt later and whether that needs the raw videos.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CLEANUP_LOG_NAME = "disk_cleanup_log.json"


@dataclass
class CleanupCategory:
    """One kind of removable data and what it costs to remove it."""

    key: str
    title: str
    description: str
    default_on: bool
    paths: list[Path] = field(default_factory=list)
    n_bytes: int = 0
    n_files: int = 0
    # Extra caution shown next to the category (e.g. "needs the raw videos").
    warning: str = ""


def _measure(path: Path) -> tuple[int, int]:
    """Bytes and file count under ``path`` (a file or a directory)."""
    try:
        if path.is_file():
            return path.stat().st_size, 1
    except OSError:
        return 0, 0
    n_bytes = n_files = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                n_bytes += f.stat().st_size
                n_files += 1
        except OSError:
            continue
    return n_bytes, n_files


def _subdirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def _training_snapshots(root: Path) -> list[Path]:
    """Every training-set snapshot except the newest.

    A full copy of the training set is written on every retrain and nothing
    reads them back.  The newest is kept as a spare copy of the live set.
    """
    snap_dir = root / "derived" / "training_sets" / "snapshots"
    if not snap_dir.is_dir():
        return []
    snaps = sorted(snap_dir.glob("training_set_*.parquet"))
    return snaps[:-1]


def _old_backups(root: Path) -> list[Path]:
    """Safety copies left by one-off repairs, merges and edits.

    ``backups/`` also holds dated copies of project.yaml as plain files; those
    are tiny and kept.  ``derived/model_backups`` is NOT a backup in this sense
    (it parks a model during a retrain and is restored on open) and is never
    touched.
    """
    out: list[Path] = []
    out.extend(_subdirs(root / "backups"))
    out.extend(_subdirs(root / "derived" / "backups"))
    removed_models = root / "derived" / "models_removed_backup"
    if removed_models.is_dir():
        out.append(removed_models)
    derived = root / "derived"
    if derived.is_dir():
        out.extend(sorted(derived.glob("behavior_bouts_bak_*")))
    ts_dir = derived / "training_sets"
    if ts_dir.is_dir():
        out.extend(sorted(ts_dir.glob("training_set.parquet.bak_*")))
        out.extend(sorted(ts_dir.glob("training_set.parquet.corrupt-backup*")))
    return out


def _scratch_caches(root: Path) -> list[Path]:
    """Temporal-refinement tab cache: rebuilt automatically when needed."""
    d = root / "derived" / "temporal_refinement" / "tab_cache"
    return _subdirs(d) + sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []


def _review_clips(root: Path) -> list[Path]:
    """Clip folders cut from this project's own videos.

    Clips copied in from another project (``<source>__session_x``) are kept:
    their source video is not in this project, so they could not be regenerated.
    """
    clips = root / "derived" / "clips"
    return [d for d in _subdirs(clips) if "__" not in d.name]


def _labeled_video_exports(root: Path) -> list[Path]:
    d = root / "exports" / "labeled_videos"
    return [d] if d.is_dir() else []


def _representation_cache(root: Path) -> list[Path]:
    """Frame and segment feature tables built from the extracted features."""
    rep = root / "derived" / "representations"
    names = ("frame_features.parquet", "segment_features.parquet", "representations.manifest.json")
    return [rep / n for n in names if (rep / n).is_file()]


_CATEGORY_SPECS: list[tuple[str, str, str, bool, Callable[[Path], list[Path]], str]] = [
    (
        "training_snapshots",
        "Old training-set snapshots",
        "A full copy of the training set saved on every retrain. Nothing reads "
        "them. The live training set and the newest snapshot are kept.",
        True,
        _training_snapshots,
        "",
    ),
    (
        "old_backups",
        "Old repair and edit backups",
        "Safety copies made before label repairs, behavior merges and similar "
        "one-off edits. Only needed to undo one of those edits by hand.",
        True,
        _old_backups,
        "",
    ),
    (
        "scratch_caches",
        "Temporal refinement scratch cache",
        "A working copy of the frame features. Rebuilt automatically the next "
        "time temporal refinement needs it.",
        True,
        _scratch_caches,
        "",
    ),
    (
        "review_clips",
        "Review clips",
        "The short video clips cut for the Review tab. Your review decisions "
        "and labels are kept. Clips can be recut later with Review > "
        "Regenerate Missing Clips. Clips imported from other projects are kept.",
        True,
        _review_clips,
        "Recutting clips needs the original videos.",
    ),
    (
        "labeled_video_exports",
        "Exported labeled videos",
        "Videos with behavior labels drawn on them, made from the Export tab. "
        "Not used by the project. Keep them if you share or present them.",
        False,
        _labeled_video_exports,
        "Re-exporting needs the original videos.",
    ),
    (
        "representation_cache",
        "Frame and segment feature tables",
        "The largest files in most projects. They are rebuilt automatically "
        "from the extracted features the next time you train, retrain or run "
        "inference, or right away with Features > Extract Pose Features, which "
        "skips sessions already extracted. The original videos are not needed "
        "and the rebuilt tables are identical.",
        False,
        _representation_cache,
        "Until rebuilt, clip suggestions and clip mining have no segments to "
        "work from. The model scores stored in the table return with the next "
        "retrain.",
    ),
]


def category_keys() -> list[str]:
    return [spec[0] for spec in _CATEGORY_SPECS]


def scan_project(project_root: Path) -> list[CleanupCategory]:
    """Measure every cleanup category.  Read-only."""
    root = Path(project_root)
    out: list[CleanupCategory] = []
    for key, title, desc, default_on, finder, warning in _CATEGORY_SPECS:
        try:
            paths = finder(root)
        except OSError as exc:
            logger.warning("Disk cleanup scan of %s failed: %s", key, exc)
            paths = []
        cat = CleanupCategory(key, title, desc, default_on, paths, warning=warning)
        for p in paths:
            b, n = _measure(p)
            cat.n_bytes += b
            cat.n_files += n
        out.append(cat)
    return out


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def cleanse_project(
    project_root: Path,
    keys: list[str] | set[str],
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Delete the chosen categories.  Returns bytes/files freed and any errors.

    Paths are found again at delete time (not taken from an earlier scan), so a
    file written since the dialog opened is judged by the same rules.  Anything
    that resolves outside the project folder is refused.
    """
    root = Path(project_root)
    chosen = [c for c in scan_project(root) if c.key in set(keys)]
    freed_bytes = freed_files = 0
    errors: list[str] = []
    removed: dict[str, int] = {}
    for cat in chosen:
        if progress_cb is not None:
            progress_cb(f"Removing {cat.title.lower()}…")
        cat_bytes = 0
        for p in cat.paths:
            if not _inside(p, root) or p.resolve() == root.resolve():
                errors.append(f"Refused (outside the project): {p}")
                continue
            b, n = _measure(p)
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
                freed_bytes += b
                freed_files += n
                cat_bytes += b
            except OSError as exc:
                errors.append(f"{p}: {exc}")
        removed[cat.key] = cat_bytes
    _append_log(root, removed, errors)
    return {
        "freed_bytes": freed_bytes,
        "freed_files": freed_files,
        "removed": removed,
        "errors": errors,
    }


def _append_log(root: Path, removed: dict[str, int], errors: list[str]) -> None:
    """Record what was removed, so a later reader knows why files are missing."""
    log_path = root / "logs" / CLEANUP_LOG_NAME
    try:
        entries = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        if not isinstance(entries, list):
            entries = []
    except (OSError, ValueError):
        entries = []
    entries.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "removed_bytes": removed,
        "errors": errors[:50],
    })
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write the disk cleanup log: %s", exc)


def format_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


# Folders that never contain an ABEL project, skipped when searching a folder.
_SEARCH_SKIP = {
    "AppData", ".git", ".venv", "venv", "node_modules", "site-packages",
    "__pycache__", "derived", "raw", "exports",
}


def find_projects(folder: Path, max_depth: int = 3) -> list[Path]:
    """ABEL projects (folders with a project.yaml) in ``folder`` and below.

    A project's own subfolders are not searched further.
    """
    found: list[Path] = []

    def _walk(d: Path, depth: int) -> None:
        if (d / "project.yaml").is_file():
            found.append(d)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            return
        for child in children:
            if child.name in _SEARCH_SKIP or child.name.startswith(("$", ".")):
                continue
            _walk(child, depth + 1)

    _walk(Path(folder), 0)
    return found


def scan_projects(project_roots: list[Path]) -> list[CleanupCategory]:
    """:func:`scan_project` summed over several projects, category by category."""
    totals: dict[str, CleanupCategory] = {}
    for root in project_roots:
        for cat in scan_project(root):
            agg = totals.setdefault(cat.key, CleanupCategory(
                cat.key, cat.title, cat.description, cat.default_on, warning=cat.warning,
            ))
            agg.paths.extend(cat.paths)
            agg.n_bytes += cat.n_bytes
            agg.n_files += cat.n_files
    return [totals[k] for k in category_keys() if k in totals]


def cleanse_projects(
    project_roots: list[Path],
    keys: list[str] | set[str],
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """:func:`cleanse_project` on each project; results are summed."""
    out: dict = {"freed_bytes": 0, "freed_files": 0, "errors": [], "projects": 0}
    for i, root in enumerate(project_roots, 1):
        if progress_cb is not None:
            progress_cb(f"Cleaning {Path(root).name} ({i}/{len(project_roots)})…")
        res = cleanse_project(root, keys)
        out["freed_bytes"] += res["freed_bytes"]
        out["freed_files"] += res["freed_files"]
        out["errors"].extend(res["errors"])
        out["projects"] += 1
    return out
