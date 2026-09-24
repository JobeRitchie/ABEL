"""Keep a trained model recoverable while its replacement is being built.

Retraining writes into ``derived/models/<model_version>``.  If the old model is
deleted or overwritten first and the run is then interrupted (the app is
closed, crashes, or the user presses Stop), the project is left with no model
or a half-written one.  :func:`protect_model_dir` parks the existing model in
``derived/model_backups/`` for the duration of the run and only discards that
copy once the new model is complete.  A backup that is still there when a
project is opened means the run never finished, so
:func:`recover_interrupted_models` puts the old model back.

Backups live outside ``derived/models`` on purpose: several scanners treat
every directory there as a model.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

BACKUP_DIRNAME = "model_backups"
_OWNER_SUFFIX = ".owner.json"
_COPYING_SUFFIX = ".copying"
_DISCARD_SUFFIX = ".discard"

# Backups held by a run in this process.  A nested guard (Retrain-all around a
# single retrain) must see its parent's backup as in use, not as left over.
_active_backups: set[str] = set()


def backup_root(project_root: Path) -> Path:
    return Path(project_root) / "derived" / BACKUP_DIRNAME


def _backup_root_for(model_dir: Path) -> Path:
    # derived/models/<name> -> derived/model_backups
    return model_dir.parent.parent / BACKUP_DIRNAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if os.name == "nt":
        # os.kill(pid, 0) terminates the process on Windows, so ask the OS.
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _owner_path(backup: Path) -> Path:
    return backup.with_name(backup.name + _OWNER_SUFFIX)


def _write_owner(backup: Path, model_dir: Path) -> None:
    _owner_path(backup).write_text(
        json.dumps({
            "pid": os.getpid(),
            "model_dir": str(model_dir),
            "started_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }),
        encoding="utf-8",
    )


def _owned_by_live_process(backup: Path) -> bool:
    if str(backup.resolve()) in _active_backups:
        return True
    try:
        info = json.loads(_owner_path(backup).read_text(encoding="utf-8"))
        pid = int(info.get("pid", 0))
    except (OSError, ValueError, TypeError):
        return False
    return pid != os.getpid() and _pid_alive(pid)


def _clear_owner(backup: Path) -> None:
    try:
        _owner_path(backup).unlink(missing_ok=True)
    except OSError:
        pass


def _restore(model_dir: Path, backup: Path) -> None:
    """Replace whatever is at ``model_dir`` with ``backup``.

    Each step leaves the backup in place until it has been moved back, so an
    interruption here is recovered by the next call.
    """
    if model_dir.exists():
        # Park the partial model beside the backup, not in derived/models,
        # where a leftover would be listed as a model.
        discard = backup.with_name(backup.name + _DISCARD_SUFFIX)
        if discard.exists():
            shutil.rmtree(discard, ignore_errors=True)
        os.replace(model_dir, discard)
        os.replace(backup, model_dir)
        shutil.rmtree(discard, ignore_errors=True)
    else:
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(backup, model_dir)
    _clear_owner(backup)


def recover_model_dir(model_dir: Path) -> bool:
    """Restore ``model_dir`` from an interrupted run's backup, if one exists."""
    model_dir = Path(model_dir)
    root = _backup_root_for(model_dir)
    backup = root / model_dir.name
    partial_copy = root / (model_dir.name + _COPYING_SUFFIX)
    if partial_copy.exists():
        # A copy that never finished: the live model was not touched yet.
        shutil.rmtree(partial_copy, ignore_errors=True)
    if not backup.is_dir() or _owned_by_live_process(backup):
        return False
    _restore(model_dir, backup)
    logger.warning(
        "Restored model '%s' from its backup: the retrain that replaced it did not finish.",
        model_dir.name,
    )
    return True


def recover_interrupted_models(project_root: Path) -> list[str]:
    """Restore every model whose retrain was interrupted. Returns their names."""
    root = backup_root(project_root)
    if not root.is_dir():
        return []
    models_dir = Path(project_root) / "derived" / "models"
    restored: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.endswith((_COPYING_SUFFIX, _DISCARD_SUFFIX)):
            continue
        try:
            if recover_model_dir(models_dir / entry.name):
                restored.append(entry.name)
        except OSError as exc:
            logger.error(
                "Could not restore model '%s' from %s: %s. The backup was left in place.",
                entry.name, entry, exc,
            )
    for entry in root.iterdir():
        for suffix in (_COPYING_SUFFIX, _DISCARD_SUFFIX):
            if entry.name.endswith(suffix) and not _owned_by_live_process(
                root / entry.name[: -len(suffix)]
            ):
                shutil.rmtree(entry, ignore_errors=True)
    return restored


def _discard_backup(backup: Path) -> None:
    """Drop a backup once its replacement is complete.

    Renamed first so a delete that fails half-way can never be mistaken for an
    interrupted run and restored over the new model.
    """
    discard = backup.with_name(backup.name + _DISCARD_SUFFIX)
    if discard.exists():
        shutil.rmtree(discard, ignore_errors=True)
    os.replace(backup, discard)
    _clear_owner(backup)
    shutil.rmtree(discard, ignore_errors=True)


class ModelDirGuard:
    """Handle yielded by :func:`protect_model_dir`."""

    def __init__(self) -> None:
        self.keep_new = True

    def restore_previous(self) -> None:
        """Ask for the previous model to be put back when the block ends."""
        self.keep_new = False


@contextmanager
def protect_model_dir(model_dir: "Path | None", *, start_fresh: bool = False) -> Iterator[ModelDirGuard]:
    """Keep the existing model at ``model_dir`` until its replacement finishes.

    ``start_fresh=True`` moves the old model aside so the new one is built in
    an empty directory (no stale artifacts carry over).  Otherwise the old
    model is copied and the new run overwrites files in place, as before.

    The previous model comes back if the block raises (including a user
    cancel), if the caller calls ``guard.restore_previous()``, or, when the
    process dies mid-run, the next time the project is opened.
    """
    guard = ModelDirGuard()
    if model_dir is None:
        yield guard
        return
    model_dir = Path(model_dir)
    root = _backup_root_for(model_dir)
    backup = root / model_dir.name

    if str(backup.resolve()) in _active_backups:
        # An enclosing run already holds this model's backup and decides its fate.
        yield guard
        return
    recover_model_dir(model_dir)
    if not model_dir.is_dir():
        yield guard
        return

    root.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        # Owned by another running ABEL; do not touch it.
        raise RuntimeError(
            f"Model '{model_dir.name}' is being retrained by another ABEL window. "
            "Wait for it to finish first."
        )
    _write_owner(backup, model_dir)
    key = str(backup.resolve())
    _active_backups.add(key)
    try:
        moved = False
        if start_fresh:
            try:
                os.replace(model_dir, backup)
                moved = True
            except OSError as exc:
                logger.warning("Could not move model '%s' aside (%s); copying it instead.", model_dir.name, exc)
        if not moved:
            partial = root / (model_dir.name + _COPYING_SUFFIX)
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(model_dir, partial)
            os.replace(partial, backup)
            if start_fresh:
                for child in model_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
    except BaseException:
        _active_backups.discard(key)
        _clear_owner(backup)
        raise
    logger.info("Backed up model '%s' to %s until the retrain finishes.", model_dir.name, backup)

    try:
        try:
            yield guard
        finally:
            _active_backups.discard(key)
    except BaseException:
        _restore(model_dir, backup)
        logger.warning("Retrain of '%s' did not finish; the previous model was restored.", model_dir.name)
        raise
    if guard.keep_new:
        _discard_backup(backup)
    else:
        _restore(model_dir, backup)
        logger.info("Retrain of '%s' produced no new model; the previous model was restored.", model_dir.name)
