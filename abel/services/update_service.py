"""In-app updater — check the git remote and pull the latest ABEL version.

Mirrors the manual updater pattern: the user explicitly clicks *Check for
Updates* (we never auto-check on launch). When an update exists, *Install
Update* runs ``git pull`` and relaunches the app via the normal launcher so the
dependency/editable-install step re-runs if ``pyproject.toml`` changed.

Network shares owned by a different account make git refuse to operate with a
"dubious ownership" error; ``ensure_safe_directory`` adds a global
``safe.directory`` exception so the updater still works there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Hide the transient console window git would otherwise flash on Windows.
SUBPROCESS_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@dataclass
class UpdateStatus:
    """Result of an update check."""

    behind: int | None = None   # commits the local branch is behind origin/main
    error: str = ""
    current: str = ""           # short HEAD hash

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def update_available(self) -> bool:
        return self.behind is not None and self.behind > 0


class UpdateService:
    """Git-backed updater for an ABEL source checkout."""

    REMOTE = "origin"
    BRANCH = "main"

    def __init__(self, repo_root: Path | None = None) -> None:
        self._repo_root = Path(repo_root) if repo_root else self._detect_repo_root()

    @staticmethod
    def _detect_repo_root() -> Path:
        # .../abel/services/update_service.py -> repo root (parent of the abel pkg)
        return Path(__file__).resolve().parents[2]

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def is_git_repo(self) -> bool:
        return (self._repo_root / ".git").exists()

    # ------------------------------------------------------------------
    # git plumbing
    # ------------------------------------------------------------------

    def _git(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self._repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=SUBPROCESS_NO_WINDOW,
        )

    def ensure_safe_directory(self) -> None:
        """Add a global safe.directory exception if git reports dubious ownership."""
        try:
            r = self._git("rev-parse", "--is-inside-work-tree", timeout=10)
        except Exception:
            return  # let the real git call surface its own error
        if r.returncode == 0:
            return
        if "dubious ownership" not in (r.stderr or "").lower():
            return
        candidates = {str(self._repo_root), str(self._repo_root).replace("\\", "/")}
        for val in candidates:
            try:
                subprocess.run(
                    ["git", "config", "--global", "--add", "safe.directory", val],
                    capture_output=True, text=True, timeout=10,
                    creationflags=SUBPROCESS_NO_WINDOW,
                )
            except Exception:
                pass

    def current_commit(self) -> str:
        try:
            r = self._git("rev-parse", "--short", "HEAD", timeout=10)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> UpdateStatus:
        """Fetch remote metadata and report how many commits HEAD is behind."""
        if not self.is_git_repo():
            return UpdateStatus(
                error="Not a git checkout — in-app updates are unavailable for "
                "this install."
            )
        self.ensure_safe_directory()
        try:
            fetch = self._git("fetch", self.REMOTE, timeout=30)
            if fetch.returncode != 0:
                return UpdateStatus(error=fetch.stderr.strip() or "git fetch failed")
            rev = self._git(
                "rev-list", f"HEAD..{self.REMOTE}/{self.BRANCH}", "--count", timeout=15
            )
            if rev.returncode != 0:
                return UpdateStatus(error=rev.stderr.strip() or "git rev-list failed")
            behind = int((rev.stdout or "0").strip() or "0")
            return UpdateStatus(behind=behind, current=self.current_commit())
        except FileNotFoundError:
            return UpdateStatus(error="git not found — please install Git.")
        except subprocess.TimeoutExpired:
            return UpdateStatus(error="Timed out — check your network connection.")
        except Exception as exc:  # pragma: no cover - defensive
            return UpdateStatus(error=str(exc))

    def pull(self, line_cb: Callable[[str], None]) -> bool:
        """Run ``git pull`` streaming output through ``line_cb``; return success."""
        if not self.is_git_repo():
            line_cb("Not a git checkout — cannot update.")
            return False
        self.ensure_safe_directory()
        try:
            proc = subprocess.Popen(
                ["git", "pull", self.REMOTE, self.BRANCH],
                cwd=str(self._repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=SUBPROCESS_NO_WINDOW,
            )
        except FileNotFoundError:
            line_cb("git not found — please install Git.")
            return False
        except Exception as exc:  # pragma: no cover - defensive
            line_cb(f"Error: {exc}")
            return False
        if proc.stdout is not None:
            for line in proc.stdout:
                line_cb(line.rstrip("\n"))
        proc.wait()
        return proc.returncode == 0

    # ------------------------------------------------------------------
    # Relaunch
    # ------------------------------------------------------------------

    def relaunch(self) -> None:
        """Start a fresh ABEL process (independent of this one).

        Prefer the launcher (``run_abel.bat``) so the editable-install / venv
        step re-runs when dependencies changed; fall back to re-running the
        module directly.
        """
        repo = str(self._repo_root)
        bat = self._repo_root / "run_abel.bat"
        if os.name == "nt" and bat.exists():
            CREATE_NEW_CONSOLE = 0x00000010
            subprocess.Popen(
                [str(bat)], cwd=repo, close_fds=True,
                creationflags=CREATE_NEW_CONSOLE,
            )
            return
        args = [sys.executable, "-m", "abel.main"]
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                args, cwd=repo, close_fds=True,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(args, cwd=repo, close_fds=True, start_new_session=True)
