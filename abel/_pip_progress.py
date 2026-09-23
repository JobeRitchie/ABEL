"""Run ``pip install`` with readable, live progress in the launcher window.

Used by ``run_abel.bat``. The launcher used to run pip with ``--quiet`` and
send everything to the log, so a first install sat silent for minutes. This
wrapper streams pip's output, rewrites it as short status lines (what is being
resolved, downloaded, built and installed), draws a download bar, and prints a
heartbeat while pip works without saying anything (unpacking PySide6 alone can
take a minute). The full raw pip output still goes to the launcher log.

Stdlib only: it runs before ABEL's dependencies are installed.

Usage::

    python -m abel._pip_progress --log LOGFILE --label "Installing ABEL" -- <pip install args>
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

_HEARTBEAT_AFTER_S = 3.0

_RE_PROGRESS = re.compile(r"^Progress (\d+) of (\d+)\s*$")
_RE_COLLECTING = re.compile(r"^Collecting (\S+)")
_RE_DOWNLOADING = re.compile(r"^\s*Downloading (\S+)(?: \(([^)]+)\))?")
_RE_CACHED = re.compile(r"^\s*Using cached (\S+)")
_RE_STEP = re.compile(r"^\s*(.+?): (started|finished with status '(\w+)')\s*$")
_RE_INSTALLING = re.compile(r"^Installing collected packages: (.+)$")
_RE_UNINSTALL = re.compile(r"^\s*Attempting uninstall: (\S+)")
_RE_SUCCESS = re.compile(r"^Successfully installed (.+)$")
_RE_ALREADY = re.compile(r"^Requirement already satisfied: (\S+)")


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} GB"


def _parse_size(text: str) -> float:
    """pip's ``12.6 MB`` / ``250 kB`` -> bytes (0 when unparseable)."""
    try:
        value, unit = text.split()
        return float(value) * {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9}[unit]
    except (ValueError, KeyError):
        return 0.0


def _short_name(filename: str) -> str:
    """``numpy-2.1.0-cp312-cp312-win_amd64.whl`` -> ``numpy 2.1.0``."""
    base = filename.rsplit("/", 1)[-1]
    for ext in (".whl.metadata", ".whl", ".tar.gz", ".zip"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("-")
    if len(parts) >= 2 and parts[1][:1].isdigit():
        return f"{parts[0]} {parts[1]}"
    return base


def _supports_raw_progress() -> bool:
    """``--progress-bar raw`` exists from pip 24.1 on."""
    try:
        from importlib.metadata import version

        major, minor = (int(p) for p in version("pip").split(".")[:2])
    except Exception:
        return False
    return (major, minor) >= (24, 1)


class _Console:
    """Permanent lines plus one in-place status line that later output replaces."""

    def __init__(self) -> None:
        self._status_len = 0
        self._width = max(40, shutil.get_terminal_size((100, 24)).columns - 1)

    def line(self, text: str, *, wrap: bool = False) -> None:
        self.clear()
        print(text if wrap else text[: self._width], flush=True)

    def status(self, text: str) -> None:
        text = text[: self._width]
        pad = max(0, self._status_len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self._status_len = len(text)

    def clear(self) -> None:
        if self._status_len:
            sys.stdout.write("\r" + " " * self._status_len + "\r")
            self._status_len = 0


class _Reporter:
    def __init__(self, console: _Console) -> None:
        self.c = console
        self.phase = "Starting pip"
        self.n_downloaded = 0
        self.bytes_downloaded = 0
        self.installed: list[str] = []
        self._dl_name = ""
        self._dl_start = 0.0
        self._dl_total = 0

    def handle(self, line: str) -> None:
        text = line.rstrip()
        if not text.strip():
            return

        m = _RE_PROGRESS.match(text)
        if m:
            self._progress(int(m.group(1)), int(m.group(2)))
            return
        if self._dl_total:
            self._finish_download()

        if text.startswith("[notice]"):
            return
        if text.startswith(("ERROR", "error:")) or "WARNING" in text[:10]:
            self.c.line("    " + text, wrap=True)
            return

        m = _RE_COLLECTING.match(text)
        if m:
            self.phase = "Resolving dependencies"
            self.c.line(f"    Resolving   {m.group(1)}")
            return
        m = _RE_DOWNLOADING.match(text)
        if m:
            fname, size = m.group(1), m.group(2) or ""
            if fname.endswith(".metadata"):
                return
            self._dl_name = _short_name(fname)
            self.n_downloaded += 1
            self.bytes_downloaded += _parse_size(size)
            self._dl_start = time.monotonic()
            self.phase = f"Downloading {self._dl_name}"
            self.c.line(f"    Downloading {self._dl_name}" + (f" ({size})" if size else ""))
            return
        m = _RE_CACHED.match(text)
        if m:
            if not m.group(1).endswith(".metadata"):
                self.c.line(f"    Cached      {_short_name(m.group(1))}")
            return
        m = _RE_ALREADY.match(text)
        if m:
            return
        m = _RE_STEP.match(text)
        if m:
            if m.group(2) == "started":
                self.phase = m.group(1)
                self.c.line(f"    {m.group(1)}...")
            elif m.group(3) not in (None, "done"):
                self.c.line(f"    {m.group(1)}: {m.group(3)}", wrap=True)
            return
        m = _RE_UNINSTALL.match(text)
        if m:
            self.c.line(f"    Replacing   {m.group(1)}")
            return
        m = _RE_INSTALLING.match(text)
        if m:
            pkgs = [p.strip() for p in m.group(1).split(",") if p.strip()]
            self.phase = f"Installing {len(pkgs)} packages"
            if any(p.lower().startswith(("pyside6", "torch")) for p in pkgs):
                self.phase += " (unpacking the large Qt/torch wheels takes a few minutes)"
            self.c.line(f"    Installing  {len(pkgs)} packages: {', '.join(pkgs)}")
            return
        m = _RE_SUCCESS.match(text)
        if m:
            self.installed = m.group(1).split()
            return

    def _progress(self, done: int, total: int) -> None:
        self._dl_total = total
        frac = done / total if total else 0.0
        width = 24
        filled = int(frac * width)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(1e-3, time.monotonic() - self._dl_start)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        self.c.status(
            f"      [{bar}] {frac * 100:5.1f}%  {_fmt_bytes(done)} / {_fmt_bytes(total)}"
            f"  {_fmt_bytes(rate)}/s  ~{_fmt_elapsed(eta)} left"
        )

    def _finish_download(self) -> None:
        self._dl_total = 0
        self.c.clear()

    def heartbeat(self, total_s: float) -> None:
        if self._dl_total:
            return
        spinner = "|/-\\"[int(total_s) % 4]
        self.c.status(f"    {spinner} {self.phase}... ({_fmt_elapsed(total_s)} elapsed)")


def _reader(stream, q: "queue.Queue[str | None]") -> None:
    for line in iter(stream.readline, ""):
        q.put(line)
    q.put(None)


def run(pip_args: list[str], label: str, log_path: str | None) -> int:
    cmd = [sys.executable, "-m", "pip", *pip_args, "--no-color", "--disable-pip-version-check"]
    if _supports_raw_progress():
        cmd += ["--progress-bar", "raw"]
    else:
        cmd += ["--progress-bar", "off"]

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log = open(log_path, "a", encoding="utf-8", errors="replace") if log_path else None
    if log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()

    console = _Console()
    reporter = _Reporter(console)
    console.line(f"[INFO] {label}")
    start = time.monotonic()
    last_output = start

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    q: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()

    try:
        while True:
            try:
                line = q.get(timeout=1.0)
            except queue.Empty:
                now = time.monotonic()
                if now - last_output >= _HEARTBEAT_AFTER_S:
                    reporter.heartbeat(now - start)
                continue
            if line is None:
                break
            last_output = time.monotonic()
            if log and not _RE_PROGRESS.match(line):
                log.write(line)
                log.flush()
            reporter.handle(line)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        console.line("[WARN] Install interrupted.")
        return 130
    finally:
        if log:
            log.close()

    elapsed = _fmt_elapsed(time.monotonic() - start)
    if code == 0:
        parts = []
        if reporter.installed:
            parts.append(f"{len(reporter.installed)} packages installed")
        if reporter.n_downloaded:
            parts.append(
                f"{reporter.n_downloaded} files, {_fmt_bytes(reporter.bytes_downloaded)} downloaded"
            )
        detail = ", ".join(parts) if parts else "nothing to change"
        console.line(f"[INFO] Done in {elapsed} ({detail}).")
    else:
        console.line(f"[ERROR] pip failed after {elapsed} (exit code {code}).")
    return code


def main(argv: list[str]) -> int:
    log_path = None
    label = "Running pip"
    if "--" not in argv:
        print("usage: python -m abel._pip_progress [--log FILE] [--label TEXT] -- <pip args>")
        return 2
    split = argv.index("--")
    opts, pip_args = argv[:split], argv[split + 1 :]
    i = 0
    while i < len(opts):
        if opts[i] == "--log" and i + 1 < len(opts):
            log_path = opts[i + 1]
            i += 2
        elif opts[i] == "--label" and i + 1 < len(opts):
            label = opts[i + 1]
            i += 2
        else:
            i += 1
    return run(pip_args, label, log_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
