"""Tests for the in-app updater service."""

from __future__ import annotations

from pathlib import Path

import abel.services.update_service as mod
from abel.services.update_service import (
    GIT_MISSING_MSG,
    UpdateService,
    UpdateStatus,
    find_git,
)


def test_status_flags() -> None:
    assert UpdateStatus(behind=3).update_available
    assert not UpdateStatus(behind=0).update_available
    assert UpdateStatus(behind=None).update_available is False
    assert UpdateStatus(behind=0).ok
    assert not UpdateStatus(error="boom").ok
    assert not UpdateStatus(error="boom").update_available


def test_not_a_git_repo(tmp_path: Path) -> None:
    svc = UpdateService(repo_root=tmp_path)
    assert not svc.is_git_repo()
    st = svc.check()
    assert not st.ok
    assert "not a git checkout" in st.error.lower()


def test_is_git_repo_detection(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert UpdateService(repo_root=tmp_path).is_git_repo()


def test_repo_root_default_points_at_package() -> None:
    svc = UpdateService()
    assert (svc.repo_root / "abel" / "__init__.py").exists()
    assert (svc.repo_root / "pyproject.toml").exists()


class _R:
    def __init__(self, rc: int = 0, out: str = "", err: str = "") -> None:
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def test_check_parses_behind_count(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    svc = UpdateService(repo_root=tmp_path)

    def fake_git(*args: str, timeout: float = 30.0) -> _R:
        if args[0] == "fetch":
            return _R(0)
        if args[0] == "rev-list":
            return _R(0, "4\n")
        if args[0] == "rev-parse":
            return _R(0, "abc1234\n")
        return _R(0)

    monkeypatch.setattr(svc, "_git", fake_git)
    st = svc.check()
    assert st.ok
    assert st.behind == 4
    assert st.update_available
    assert st.current == "abc1234"


def test_check_reports_fetch_error(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    svc = UpdateService(repo_root=tmp_path)

    def fake_git(*args: str, timeout: float = 30.0) -> _R:
        if args[0] == "rev-parse":
            return _R(0)  # safe-directory probe ok
        if args[0] == "fetch":
            return _R(1, "", "could not resolve host github.com")
        return _R(0)

    monkeypatch.setattr(svc, "_git", fake_git)
    st = svc.check()
    assert not st.ok
    assert "could not resolve host" in st.error


def test_find_git_prefers_path(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: r"C:\path\git.exe")
    assert find_git() == r"C:\path\git.exe"


def test_find_git_falls_back_to_program_files(monkeypatch, tmp_path: Path) -> None:
    # git is installed but not on PATH (the stale-PATH GUI scenario).
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(mod.os, "name", "nt")
    git_exe = tmp_path / "Git" / "cmd" / "git.exe"
    git_exe.parent.mkdir(parents=True)
    git_exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert find_git() == str(git_exe)


def test_check_reports_missing_git(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    svc = UpdateService(repo_root=tmp_path)
    monkeypatch.setattr(svc, "git_executable", lambda: None)
    st = svc.check()
    assert not st.ok
    assert st.error == GIT_MISSING_MSG


def test_pull_reports_missing_git(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    svc = UpdateService(repo_root=tmp_path)
    monkeypatch.setattr(svc, "git_executable", lambda: None)
    lines: list[str] = []
    assert svc.pull(lines.append) is False
    assert GIT_MISSING_MSG in lines


def test_safe_directory_uses_path_from_error(monkeypatch, tmp_path: Path) -> None:
    svc = UpdateService(repo_root=tmp_path)
    monkeypatch.setattr(svc, "git_executable", lambda: "git")

    reported = "C:/some/weird//share"

    def fake_git(*args: str, timeout: float = 30.0) -> _R:
        return _R(128, "", f"fatal: detected dubious ownership in repository at '{reported}'")

    monkeypatch.setattr(svc, "_git", fake_git)

    added: list[str] = []

    def fake_run(cmd, **kwargs):
        added.append(cmd[-1])
        return _R(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    svc.ensure_safe_directory()
    assert reported in added


def test_relaunch_prefers_launcher_bat(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "run_abel.bat").write_text("echo hi", encoding="utf-8")
    svc = UpdateService(repo_root=tmp_path)
    spawned: dict = {}

    def fake_popen(cmd, **kwargs):
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs

        class _P:
            pass

        return _P()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.os, "name", "nt")
    svc.relaunch()
    assert spawned["cmd"] == [str(tmp_path / "run_abel.bat")]
