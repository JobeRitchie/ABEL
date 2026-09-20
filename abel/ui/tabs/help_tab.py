"""Help tab with workflow guidance."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class HelpTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root: Path | None = None

        text = QLabel(
            "<h3>ABEL Workflow</h3>"
            "<p>Tabs run left to right. You do not need all of them for every "
            "project.</p>"
            "<ol>"
            "<li><b>Data Import</b>: add videos and pose files, pair them, and "
            "set each session's subject and session type.</li>"
            "<li><b>Behaviors</b>: define the behaviors you want to score and "
            "write a description each reviewer can work from.</li>"
            "<li><b>ROI</b>: draw the zones the assay depends on, such as open "
            "arms or a chamber. Zone features are built from these.</li>"
            "<li><b>Features</b>: choose which pose, context and video features "
            "to extract, then extract them.</li>"
            "<li><b>Active Learning</b>: add <b>Seeds</b>, train in "
            "<b>Learning</b>, extract clips in <b>Clips</b>, and confirm or "
            "correct them in <b>Review</b>. Retrain and repeat.</li>"
            "<li><b>Temporal</b>: turn segment scores into frame-level bouts in "
            "<b>Refinement</b>, then tune onset and cleanup in <b>Review</b>.</li>"
            "<li><b>Analytics</b>: plot bout counts, durations, time in zone, "
            "transitions and motifs, and run group comparisons.</li>"
            "<li><b>Validation</b>: measure how well a model generalizes with a "
            "held-out split, leave-one-subject-out, or session quality.</li>"
            "<li><b>Export</b>: write the spreadsheets, workbooks and labeled "
            "videos you need outside ABEL.</li>"
            "<li><b>Direct Use</b>: apply a trained model to new videos without "
            "relabeling, and send corrections back with Transfer Feedback.</li>"
            "<li><b>Model Refinement</b>: import models from another project and "
            "map their behaviors onto this one.</li>"
            "</ol>"
            "<p><b>Info</b> holds the version history, the methods reference and "
            "the methods write-up helper. <b>Settings</b> holds preferences, "
            "dependencies and logs.</p>"
            "<p>Project artifacts are written inside your project folder under "
            "<code>derived/</code> and <code>exports/</code>.</p>"
        )
        text.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.addWidget(text)
        holder_layout.addStretch()
        scroll.setWidget(holder)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
