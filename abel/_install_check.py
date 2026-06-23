"""Decide whether ABEL needs a (re)install. Exit 0 = up to date, 1 = install.

Used by ``run_abel.bat``. ABEL is installed editable (``pip install -e .``), so
source edits are live without reinstalling. A reinstall is only needed on first
run, when the venv was recreated, or when ``pyproject.toml`` changes
(dependencies / version / entry points). The launcher writes a stamp file after
a successful install; this module exits 0 only when that stamp exists and is at
least as new as ``pyproject.toml``.

Run as ``python -m abel._install_check``. If the package itself cannot be
imported (not installed yet), the ``-m`` invocation fails with a non-zero exit,
which the launcher correctly treats as "needs install".
"""

from __future__ import annotations

import sys
from pathlib import Path

# project_root/abel/_install_check.py -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAMP = _PROJECT_ROOT / ".venv" / ".abel_install_stamp"
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


def needs_install() -> bool:
    try:
        if not _STAMP.exists():
            return True
        if _PYPROJECT.exists() and _STAMP.stat().st_mtime < _PYPROJECT.stat().st_mtime:
            return True
    except OSError:
        return True
    return False


if __name__ == "__main__":
    sys.exit(1 if needs_install() else 0)
