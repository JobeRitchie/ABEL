"""Free Up Disk Space dialog.

Works on the open project or on every project found in a chosen folder.
Measures each removable category in the background, lets the user tick which
to remove, confirms, then deletes in the background.  See
:mod:`abel.services.project_cleanup_service` for what each category holds.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from abel.services.project_cleanup_service import (
    CleanupCategory,
    cleanse_projects,
    find_projects,
    format_bytes,
    scan_projects,
)
from abel.workers.task_worker import TaskWorker, active_task_count

_DESC_STYLE = "font-size: 11px; color: #B0BEC5;"
_WARN_STYLE = "font-size: 11px; color: #FFB74D;"


class ProjectCleanupDialog(QDialog):
    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current = Path(project_root)
        self._folder: Path | None = None
        self._roots: list[Path] = [self._current]
        self._categories: list[CleanupCategory] = []
        self._checks: dict[str, QCheckBox] = {}
        self._empty: set[str] = set()
        self._busy = False
        self.freed_bytes = 0

        self.setWindowTitle("Free Up Disk Space")
        self.setModal(True)

        intro = QLabel(
            "Remove large files a project no longer needs. Raw data, extracted "
            "features, models, labels, review decisions and analysis results are "
            "always kept, so you can keep working on the project later."
        )
        intro.setWordWrap(True)

        self._this_radio = QRadioButton(f"This project ({self._current.name})")
        self._folder_radio = QRadioButton("All projects in a folder:")
        self._this_radio.setChecked(True)
        self._folder_label = QLabel("none chosen")
        self._folder_label.setStyleSheet(_DESC_STYLE)
        self._browse_btn = QPushButton("Choose Folder…")
        self._browse_btn.clicked.connect(self._choose_folder)
        self._this_radio.toggled.connect(self._on_scope_changed)
        scope_row = QHBoxLayout()
        scope_row.addWidget(self._this_radio)
        scope_row.addSpacing(16)
        scope_row.addWidget(self._folder_radio)
        scope_row.addWidget(self._browse_btn)
        scope_row.addWidget(self._folder_label, 1)

        self._status = QLabel("Measuring project folder…")
        self._status.setStyleSheet("font-weight: 700;")

        self._list_host = QWidget()
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._list_host)

        self._total = QLabel("")
        self._total.setStyleSheet("font-weight: 700;")

        self._buttons = QDialogButtonBox()
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setEnabled(False)
        self._buttons.addButton(self._remove_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        close_btn = self._buttons.addButton(QDialogButtonBox.StandardButton.Close)
        # Enter must never start a deletion.
        self._remove_btn.setAutoDefault(False)
        close_btn.setDefault(True)
        self._remove_btn.clicked.connect(self._confirm_and_remove)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(scope_row)
        layout.addWidget(self._status)
        layout.addWidget(scroll, 1)
        layout.addWidget(self._total)
        layout.addWidget(self._buttons)

        # Size from the font so the dialog is legible under display scaling.
        em = self.fontMetrics().averageCharWidth()
        line = self.fontMetrics().height()
        self.resize(em * 95, line * 40)

        self._start_scan()

    # ── Scope ────────────────────────────────────────────────────────────
    def _choose_folder(self) -> None:
        start = str(self._folder or self._current.parent)
        folder = QFileDialog.getExistingDirectory(self, "Folder With ABEL Projects", start)
        if not folder:
            if self._folder is None:
                self._this_radio.setChecked(True)
            return
        self._folder = Path(folder)
        self._folder_label.setText(str(self._folder))
        if self._folder_radio.isChecked():
            self._start_scan()
        else:
            self._folder_radio.setChecked(True)

    def _on_scope_changed(self) -> None:
        if self._folder_radio.isChecked() and self._folder is None:
            self._choose_folder()
            return
        self._start_scan()

    def _target_name(self) -> str:
        if len(self._roots) == 1:
            return self._roots[0].name
        return f"{len(self._roots)} projects"

    # ── Scan ─────────────────────────────────────────────────────────────
    def _scan_target(self) -> tuple[list[Path], list[CleanupCategory]]:
        if self._folder_radio.isChecked() and self._folder is not None:
            roots = find_projects(self._folder)
        else:
            roots = [self._current]
        return roots, scan_projects(roots)

    def _start_scan(self) -> None:
        self._set_busy(True)
        self._status.setText("Finding and measuring projects…")
        worker = TaskWorker(self._scan_target)
        worker.signals.finished.connect(self._on_scanned)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_scanned(self, result: tuple[list[Path], list[CleanupCategory]]) -> None:
        self._set_busy(False)
        self._roots, categories = result
        self._categories = categories
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._checks.clear()
        self._empty = {c.key for c in categories if c.n_bytes == 0}
        reclaimable = 0
        for cat in categories:
            reclaimable += cat.n_bytes
            self._list_layout.addWidget(self._category_row(cat))
        self._list_layout.addStretch(1)
        if not self._roots:
            self._status.setText("No ABEL projects (folders with a project.yaml) found there.")
        elif reclaimable == 0:
            self._status.setText("Nothing to remove: this project is already lean.")
        else:
            self._status.setText(
                f"Up to {format_bytes(reclaimable)} can be removed from {self._target_name()}."
            )
        self._update_total()

    def _category_row(self, cat: CleanupCategory) -> QWidget:
        row = QWidget()
        lay = QVBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        empty = cat.n_bytes == 0
        check = QCheckBox(
            f"{cat.title}  ({format_bytes(cat.n_bytes)}, {cat.n_files:,} files)"
            if not empty else f"{cat.title}  (nothing to remove)"
        )
        check.setChecked(cat.default_on and not empty)
        check.setEnabled(not empty)
        check.toggled.connect(self._update_total)
        self._checks[cat.key] = check
        lay.addWidget(check)
        desc = QLabel(cat.description)
        desc.setWordWrap(True)
        desc.setStyleSheet(_DESC_STYLE)
        desc.setContentsMargins(check.sizeHint().height(), 0, 0, 0)
        lay.addWidget(desc)
        if cat.warning and not empty:
            warn = QLabel(cat.warning)
            warn.setWordWrap(True)
            warn.setStyleSheet(_WARN_STYLE)
            warn.setContentsMargins(check.sizeHint().height(), 0, 0, 0)
            lay.addWidget(warn)
        return row

    def _selected(self) -> list[CleanupCategory]:
        return [c for c in self._categories if self._checks.get(c.key) and self._checks[c.key].isChecked()]

    def _update_total(self) -> None:
        sel = self._selected()
        total = sum(c.n_bytes for c in sel)
        self._total.setText(f"Selected: {format_bytes(total)}" if sel else "Nothing selected.")
        self._remove_btn.setEnabled(bool(sel) and total > 0 and not self._busy)

    # ── Remove ───────────────────────────────────────────────────────────
    def _confirm_and_remove(self) -> None:
        sel = self._selected()
        if not sel or not self._roots:
            return
        if active_task_count() > 0:
            QMessageBox.warning(
                self, "Free Up Disk Space",
                "A task is still running in this project. Wait for it to finish, "
                "then try again.",
            )
            return
        lines = "\n".join(f"  • {c.title}: {format_bytes(c.n_bytes)}" for c in sel)
        if len(self._roots) > 1:
            names = ", ".join(r.name for r in self._roots[:12])
            more = f" and {len(self._roots) - 12} more" if len(self._roots) > 12 else ""
            lines += f"\n\nProjects: {names}{more}"
        reply = QMessageBox.question(
            self,
            "Remove Files?",
            f"These files will be permanently deleted from {self._target_name()}:\n\n"
            f"{lines}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True)
        self._status.setText("Removing files…")
        worker = TaskWorker(cleanse_projects, list(self._roots), [c.key for c in sel])
        worker.signals.finished.connect(self._on_removed)
        worker.signals.failed.connect(self._on_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_removed(self, result: dict) -> None:
        self._set_busy(False)
        freed = int(result.get("freed_bytes", 0))
        self.freed_bytes += freed
        errors = list(result.get("errors") or [])
        msg = f"Freed {format_bytes(freed)} ({int(result.get('freed_files', 0)):,} files)."
        if errors:
            shown = "\n".join(errors[:10])
            more = f"\n…and {len(errors) - 10} more" if len(errors) > 10 else ""
            QMessageBox.warning(
                self, "Free Up Disk Space",
                f"{msg}\n\nSome files could not be removed (they may be open in "
                f"another program):\n{shown}{more}",
            )
        else:
            QMessageBox.information(self, "Free Up Disk Space", msg)
        self._start_scan()

    def _on_failed(self, tb: str) -> None:
        self._set_busy(False)
        self._status.setText("The operation failed.")
        last = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        QMessageBox.critical(self, "Free Up Disk Space", last)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for key, check in self._checks.items():
            check.setEnabled(not busy and key not in self._empty)
        self._remove_btn.setEnabled(not busy and bool(self._selected()))
        for w in (self._this_radio, self._folder_radio, self._browse_btn):
            w.setEnabled(not busy)
        close_btn = self._buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setEnabled(not busy)

    def reject(self) -> None:  # noqa: D102 - stay open while files are being deleted
        if self._busy:
            return
        super().reject()

    def keyPressEvent(self, event) -> None:  # noqa: N802, D102
        if self._busy and event.key() == Qt.Key.Key_Escape:
            return
        super().keyPressEvent(event)
