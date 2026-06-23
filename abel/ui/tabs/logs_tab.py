"""Logs tab for viewing project/application logs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.core.constants import GLOBAL_LOG_DIR


class LogsTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None

        self._status = QLabel("No project loaded.")
        self._text = QTextEdit()
        self._text.setReadOnly(True)

        refresh_btn = QPushButton("Refresh Logs")
        refresh_btn.clicked.connect(self._refresh)

        row = QHBoxLayout()
        row.addWidget(refresh_btn)
        row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addLayout(row)
        layout.addWidget(self._text, 1)

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._status.setText(f"Project logs: {project_root / 'logs'}")
        self._refresh()

    def _refresh(self) -> None:
        sections: list[str] = []

        app_log = GLOBAL_LOG_DIR / "abel_app.log"
        if app_log.exists():
            text = app_log.read_text(encoding="utf-8", errors="ignore")
            sections.append("=== App Log ===\n" + "\n".join(text.splitlines()[-300:]))

        if self._project_root:
            project_log = self._project_root / "logs" / "project.log"
            if project_log.exists():
                text = project_log.read_text(encoding="utf-8", errors="ignore")
                sections.append("=== Project Log ===\n" + "\n".join(text.splitlines()[-300:]))

        self._text.setPlainText("\n\n".join(sections) if sections else "No logs found.")
