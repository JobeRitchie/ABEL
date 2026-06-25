"""Info tab — app version, in-app updater, and version history.

A sub-notebook keeps the "About & Updates" controls and the "Version History"
changelog together. Updates are **manual only**: nothing is checked on launch;
the user clicks *Check for Updates*, and *Install Update* pulls the latest
version and relaunches the app.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel import __version__
from abel.changelog import VERSION_DATE, format_changelog
from abel.services.update_service import UpdateService, UpdateStatus

logger = logging.getLogger("abel")


class _CheckWorker(QThread):
    """Runs the (network) update check off the UI thread."""

    # Named ``done`` (not ``finished``) so it doesn't shadow QThread's built-in
    # ``finished`` signal, which Qt uses for thread-lifecycle management.
    done = Signal(object)  # UpdateStatus

    def __init__(self, svc: UpdateService) -> None:
        super().__init__()
        self._svc = svc

    def run(self) -> None:
        try:
            status = self._svc.check()
        except Exception as exc:  # pragma: no cover - defensive
            status = UpdateStatus(error=str(exc))
        self.done.emit(status)


class _PullWorker(QThread):
    """Runs ``git pull`` off the UI thread, streaming output lines."""

    line = Signal(str)
    done = Signal(bool)

    def __init__(self, svc: UpdateService) -> None:
        super().__init__()
        self._svc = svc

    def run(self) -> None:
        try:
            ok = self._svc.pull(self.line.emit)
        except Exception as exc:  # pragma: no cover - defensive
            self.line.emit(f"Error: {exc}")
            ok = False
        self.done.emit(ok)


class InfoTab(QWidget):
    """About / updates / version history."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._svc = UpdateService()
        self._check_worker: _CheckWorker | None = None
        self._pull_worker: _PullWorker | None = None

        inner = QTabWidget()
        inner.setTabPosition(QTabWidget.TabPosition.North)
        inner.addTab(self._build_about_tab(), "About & Updates")
        inner.addTab(self._build_history_tab(), "Version History")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(inner)

    # ── About & Updates subtab ─────────────────────────────────────────

    def _build_about_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel("ABEL")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #90CAF9;")
        subtitle = QLabel("Active-learning Behavior Estimation and Labeling")
        subtitle.setStyleSheet("font-size: 12px; color: #607D8B;")
        version = QLabel(f"Version {__version__}   •   {VERSION_DATE}")
        version.setStyleSheet("font-size: 13px; font-weight: 700; color: #B0BEC5;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)
        layout.addSpacing(8)

        # ── Application Updates section ────────────────────────────────
        updates_header = QLabel("Application Updates")
        updates_header.setStyleSheet(
            "font-size: 14px; font-weight: 800; color: #90CAF9; padding-top: 6px;"
        )
        layout.addWidget(updates_header)

        desc = QLabel(
            "Check for and install the latest version from GitHub. ABEL never "
            "updates on its own — use the button below. Installing pulls the "
            "update and restarts the app."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #607D8B;")
        layout.addWidget(desc)

        status_row = QHBoxLayout()
        status_caption = QLabel("Status:")
        status_caption.setStyleSheet("font-size: 12px; color: #78909C;")
        self._status_label = QLabel("Not checked")
        self._status_label.setStyleSheet("font-size: 12px; color: #90A4AE;")
        status_row.addWidget(status_caption)
        status_row.addWidget(self._status_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        btn_row = QHBoxLayout()
        self._check_btn = QPushButton("Check for Updates")
        self._check_btn.clicked.connect(self._start_check)
        self._install_btn = QPushButton("Install Update")
        self._install_btn.setEnabled(False)
        self._install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self._check_btn)
        btn_row.addWidget(self._install_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        log_caption = QLabel("Update log:")
        log_caption.setStyleSheet("font-size: 11px; color: #78909C; padding-top: 4px;")
        layout.addWidget(log_caption)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "QPlainTextEdit { background: #0A1929; color: #B0BEC5;"
            " font-family: Consolas, monospace; font-size: 11px;"
            " border: 1px solid #1E3A5F; }"
        )
        self._log.setMinimumHeight(160)
        layout.addWidget(self._log, 1)

        if not self._svc.is_git_repo():
            self._status_label.setText("Updates unavailable (not a git checkout)")
            self._status_label.setStyleSheet("font-size: 12px; color: #FFB74D;")
            self._check_btn.setEnabled(False)

        return w

    # ── Version History subtab ─────────────────────────────────────────

    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)

        header = QLabel("Version History — What's Changed")
        header.setStyleSheet("font-size: 14px; font-weight: 800; color: #90CAF9;")
        layout.addWidget(header)

        view = QTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        view.setStyleSheet(
            "QTextEdit { background: #0A1929; color: #CFD8DC;"
            " font-family: Consolas, monospace; font-size: 12px;"
            " border: 1px solid #1E3A5F; }"
        )
        view.setPlainText(format_changelog())
        layout.addWidget(view, 1)
        return w

    # ── No-op project hook (for main-window lazy-init uniformity) ───────

    def set_project(self, project_root: Path) -> None:  # noqa: D401
        """Info is app-global; nothing project-specific to load."""
        return

    # ── Check ──────────────────────────────────────────────────────────

    def _start_check(self) -> None:
        if self._check_worker is not None:
            return
        self._check_btn.setEnabled(False)
        self._install_btn.setEnabled(False)
        self._set_status("Checking…", "#90A4AE")
        self._check_worker = _CheckWorker(self._svc)
        self._check_worker.done.connect(self._on_check_done)
        self._check_worker.start()

    def _on_check_done(self, status: UpdateStatus) -> None:
        if self._check_worker is not None:
            self._check_worker.wait()
            self._check_worker.deleteLater()
            self._check_worker = None
        self._check_btn.setEnabled(self._svc.is_git_repo())
        if not status.ok:
            # Keep the status line short; put the full (possibly multi-line)
            # git error in the log where it can be read and copied.
            first_line = status.error.splitlines()[0] if status.error else "unknown error"
            self._set_status(f"⚠ Could not check: {first_line}", "#FFB74D")
            self._append(f"Check failed:\n{status.error}")
            self._install_btn.setEnabled(False)
            return
        if status.update_available:
            n = status.behind
            self._set_status(
                f"⚠ Update available — {n} commit(s) behind", "#EF5350"
            )
            self._install_btn.setEnabled(True)
            self._append(f"Update available: {n} new commit(s) on {self._svc.REMOTE}/{self._svc.BRANCH}.")
        else:
            self._set_status("✓ Up to date", "#66BB6A")
            self._install_btn.setEnabled(False)
            self._append("Already up to date.")

    # ── Install ────────────────────────────────────────────────────────

    def _start_install(self) -> None:
        if self._pull_worker is not None:
            return
        reply = QMessageBox.question(
            self, "Install Update",
            "This will download and apply the latest version from GitHub, then "
            "restart ABEL.\n\nUnsaved work will be lost. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._check_btn.setEnabled(False)
        self._install_btn.setEnabled(False)
        self._log.clear()
        self._append("Pulling latest version from GitHub…")
        self._append("=" * 60)
        self._set_status("Updating…", "#90A4AE")
        self._pull_worker = _PullWorker(self._svc)
        self._pull_worker.line.connect(self._append)
        self._pull_worker.done.connect(self._on_pull_done)
        self._pull_worker.start()

    def _on_pull_done(self, ok: bool) -> None:
        if self._pull_worker is not None:
            self._pull_worker.wait()
            self._pull_worker.deleteLater()
            self._pull_worker = None
        if ok:
            self._append("=" * 60)
            self._append("✓ Update successful. Restarting ABEL in 2 seconds…")
            self._set_status("✓ Updated — restarting", "#66BB6A")
            QTimer.singleShot(2000, self._restart)
        else:
            self._append("✗ Update failed. See the log above.")
            self._set_status("⚠ Update failed", "#EF5350")
            self._check_btn.setEnabled(True)
            QMessageBox.critical(
                self, "Update Failed",
                "git pull reported an error. Check the update log for details.",
            )

    def _restart(self) -> None:
        try:
            self._svc.relaunch()
        except Exception as exc:
            QMessageBox.critical(
                self, "Restart Failed",
                f"The update was applied but ABEL could not relaunch "
                f"automatically:\n{exc}\n\nPlease close and reopen ABEL.",
            )
            return
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # ── helpers ────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"font-size: 12px; color: {color};")

    def _append(self, line: str) -> None:
        self._log.appendPlainText(line)
