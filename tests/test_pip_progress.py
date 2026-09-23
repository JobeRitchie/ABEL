"""The launcher's pip wrapper turns raw pip output into readable progress."""

from __future__ import annotations

from abel import _pip_progress
from abel._pip_progress import _Console, _Reporter, _parse_size, _short_name

PIP_OUTPUT = """\
Obtaining file:///C:/abel
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
Collecting PySide6>=6.7 (from abel==0.22.0)
  Downloading pyside6-6.11.2-cp39-abi3-win_amd64.whl.metadata (5.5 kB)
Collecting numpy>=1.26 (from abel==0.22.0)
  Using cached numpy-2.5.3-cp312-cp312-win_amd64.whl.metadata (60 kB)
Downloading pyside6_addons-6.11.2-cp39-abi3-win_amd64.whl (168.2 MB)
Progress 0 of 168200000
Progress 84100000 of 168200000
Progress 168200000 of 168200000
Using cached numpy-2.5.3-cp312-cp312-win_amd64.whl (12.6 MB)
Requirement already satisfied: six in c:\\venv\\lib\\site-packages
  Building editable for abel (pyproject.toml): started
  Building editable for abel (pyproject.toml): finished with status 'done'
Installing collected packages: numpy, pyside6_addons, abel
Successfully installed abel-0.22.0 numpy-2.5.3 pyside6_addons-6.11.2

[notice] A new release of pip is available: 26.1.2 -> 26.2.1
"""


def _lines(out: str) -> list[str]:
    return [ln.strip() for ln in out.replace("\r", "\n").split("\n") if ln.strip()]


def test_reporter_summarizes_pip_output(capsys):
    rep = _Reporter(_Console())
    for line in PIP_OUTPUT.splitlines(keepends=True):
        rep.handle(line)
    lines = _lines(capsys.readouterr().out)

    assert "Resolving   PySide6>=6.7" in lines
    assert "Downloading pyside6_addons 6.11.2 (168.2 MB)" in lines
    assert "Cached      numpy 2.5.3" in lines
    assert any(ln.startswith("[####") and "50.0%" in ln for ln in lines)
    assert lines.count("Building editable for abel (pyproject.toml)...") == 1
    assert any(ln.startswith("Installing  3 packages") for ln in lines)
    # Metadata fetches, satisfied requirements and pip's notice are noise.
    assert not any(".metadata" in ln or "already satisfied" in ln or "notice" in ln for ln in lines)
    assert rep.n_downloaded == 1
    assert rep.bytes_downloaded == 168.2e6
    assert rep.installed == ["abel-0.22.0", "numpy-2.5.3", "pyside6_addons-6.11.2"]


def test_errors_are_shown_unabridged(capsys):
    rep = _Reporter(_Console())
    long_error = "ERROR: Could not find a version that satisfies the requirement " + "x" * 300
    rep.handle(long_error + "\n")
    assert long_error in capsys.readouterr().out


def test_helpers():
    assert _short_name("numpy-2.1.0-cp312-cp312-win_amd64.whl") == "numpy 2.1.0"
    assert _short_name("python_dateutil-2.9.0.post0-py2.py3-none-any.whl") == "python_dateutil 2.9.0.post0"
    assert _parse_size("12.6 MB") == 12.6e6
    assert _parse_size("250 kB") == 250e3
    assert _parse_size("") == 0.0


def test_run_logs_and_reports_exit_code(tmp_path, capsys):
    log = tmp_path / "launcher.log"
    code = _pip_progress.run(
        ["install", "--no-index", "--dry-run", "pip"], "Checking pip...", str(log)
    )
    assert code == 0
    assert "Requirement already satisfied: pip" in log.read_text(encoding="utf-8")

    code = _pip_progress.run(["no-such-pip-command"], "Bad call...", None)
    out = capsys.readouterr().out
    assert code != 0
    assert "[ERROR] pip failed" in out


def test_usage_without_separator():
    assert _pip_progress.main(["--label", "x"]) == 2
