"""Dependencies management tab."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.services.dependency_service import DependencyService
from abel.services.smoke_test_service import SmokeTestService
from abel.workers.task_worker import TaskWorker


class DependenciesTab(QWidget):
    """Beginner-friendly dependency inspection and installation UI."""

    def __init__(self, dependency_service: DependencyService, parent=None) -> None:
        super().__init__(parent)
        self._deps = dependency_service
        self._smoke = SmokeTestService()
        self._logger = logging.getLogger("abel")
        self._pool = QThreadPool.globalInstance()

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Package", "Purpose", "Installed", "Required", "Status", "Tier"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Dependency actions and validation logs appear here.")

        self.explanation = QLabel(
            "Tier 1 packages are required to run the core app.\n"
            "Tier 2 packages enable preprocessing, modeling acceleration, and video fusion.\n"
            "On Windows, the app prefers GPU for XGBoost/PyTorch paths when available and falls back to CPU.\n"
            "Report export tools (Word/Excel) require python-docx and openpyxl."
        )
        self.explanation.setWordWrap(True)

        refresh_btn = QPushButton("Refresh")
        install_all_btn = QPushButton("Install All Dependencies")
        uninstall_selected_btn = QPushButton("Uninstall Selected")
        smoke_test_btn = QPushButton("Run Smoke Test")
        copy_report_btn = QPushButton("Copy Diagnostic Report")

        refresh_btn.clicked.connect(self.refresh)
        install_all_btn.clicked.connect(
            lambda: self.install_packages(self._deps.recommended_all())
        )
        uninstall_selected_btn.clicked.connect(self.uninstall_selected)
        smoke_test_btn.clicked.connect(self.run_smoke_test)
        copy_report_btn.clicked.connect(self.copy_report)

        button_row = QHBoxLayout()
        for btn in [
            refresh_btn,
            install_all_btn,
            uninstall_selected_btn,
            smoke_test_btn,
            copy_report_btn,
        ]:
            button_row.addWidget(btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.explanation)
        layout.addLayout(button_row)
        layout.addWidget(self.table)
        layout.addWidget(QLabel("Dependency Logs"))
        layout.addWidget(self.log_panel)

        self.refresh()

    def refresh(self) -> None:
        specs = self._deps.scan()
        self.table.setRowCount(len(specs))
        for row, spec in enumerate(specs):
            values = [
                spec.package,
                spec.purpose,
                spec.installed_version or "-",
                spec.required_version,
                spec.status,
                spec.tier,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self._append_log("Environment scan complete.")

    def install_packages(self, packages: list[str]) -> None:
        if not packages:
            return
        self._append_log(f"Installing: {', '.join(packages)}")
        worker = TaskWorker(self._deps.install_packages, packages)
        # Stream each pip output line live into the log panel
        worker.kwargs["on_line"] = worker.signals.line_emitted.emit
        worker.signals.line_emitted.connect(self._append_log)
        worker.signals.finished.connect(self._on_action_result)
        worker.signals.failed.connect(self._on_action_error)
        self._pool.start(worker)

    def uninstall_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            self._append_log("No rows selected for uninstall.")
            return
        item = self.table.item(rows[0], 0)
        if item is None:
            self._append_log("Could not resolve selected package.")
            return
        package = item.text()
        self._append_log(f"Uninstalling: {package}")
        worker = TaskWorker(self._deps.uninstall_package, package)
        worker.kwargs["on_line"] = worker.signals.line_emitted.emit
        worker.signals.line_emitted.connect(self._append_log)
        worker.signals.finished.connect(self._on_action_result)
        worker.signals.failed.connect(self._on_action_error)
        self._pool.start(worker)

    def _on_action_result(self, result) -> None:
        command_text = " ".join(result.command)
        self._append_log(f"Command: {command_text}")
        self._append_log(result.output)
        self._append_log("Action succeeded." if result.success else "Action failed.")
        self._logger.info("Dependency action completed: %s", command_text)
        self.refresh()

    def _on_action_error(self, traceback_text: str) -> None:
        self._append_log("Dependency action crashed.")
        self._append_log(traceback_text)
        self._logger.exception("Dependency action crashed\n%s", traceback_text)

    def run_smoke_test(self) -> None:
        """Run synthetic-data smoke tests in a background thread."""
        self._append_log("Starting smoke test ...")
        worker = TaskWorker(self._smoke.run_all)
        worker.kwargs["on_line"] = worker.signals.line_emitted.emit
        worker.signals.line_emitted.connect(self._append_log)
        worker.signals.finished.connect(self._on_smoke_result)
        worker.signals.failed.connect(self._on_action_error)
        self._pool.start(worker)

    def _on_smoke_result(self, report) -> None:
        if report.all_passed:
            self._append_log("All smoke tests passed.")
        else:
            failed = [r.name for r in report.results if not r.passed]
            self._append_log(f"Some tests failed: {', '.join(failed)}")

    def copy_report(self) -> None:
        report_lines = []
        for row in range(self.table.rowCount()):
            row_values = []
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                row_values.append(item.text() if item is not None else "")
            report_lines.append(" | ".join(row_values))
        report = "\n".join(report_lines)
        QApplication.clipboard().setText(report)
        self._append_log(f"Diagnostic report ready ({len(report_lines)} packages).")

    def _append_log(self, message: str) -> None:
        self.log_panel.append(message)
        # Keep the view scrolled to the bottom so live output is always visible
        self.log_panel.verticalScrollBar().setValue(
            self.log_panel.verticalScrollBar().maximum()
        )
