"""Help tab with workflow guidance."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class HelpTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None

        text = QLabel(
            "<h3>ABEL Workflow</h3>"
            "<ol>"
            "<li>Import videos and pose files in <b>Data Import</b>.</li>"
            "<li>Define behaviors and add seed examples.</li>"
            "<li>Run <b>Active Learning</b> to compute pose+context features, train the target behavior model, and rank segment candidates.</li>"
            "<li>Extract clips for top-ranked segments in <b>Clip Extraction</b>.</li>"
            "<li>Review and relabel segments in <b>Review</b>, then retrain for the next round.</li>"
            "<li>Export confirmed behaviors and quantitative outputs in <b>Export</b>.</li>"
            "</ol>"
            "<p>Project artifacts are written inside your project folder under <code>derived/</code> and <code>exports/</code>.</p>"
        )
        text.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(text)
        layout.addStretch()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
