"""Welcome/startup widget shown before project load."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StartupWidget(QWidget):
    create_project_requested = Signal()
    open_project_requested = Signal()
    dependencies_requested = Signal()
    recent_project_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # ── Hero header ──────────────────────────────────────────────────────
        title = QLabel("ABEL")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-family: 'Segoe UI', 'Arial Black', sans-serif;"
            "font-size: 52px;"
            "font-weight: 900;"
            "letter-spacing: 6px;"
            "color: #E8F4FF;"
            "padding: 18px 0 4px 0;"
        )

        acronym = QLabel(
            "<span style='color:#E8F4FF;'>A</span>ctive-learning &nbsp;"
            "<span style='color:#E8F4FF;'>B</span>ehavior &nbsp;"
            "<span style='color:#E8F4FF;'>E</span>stimation and &nbsp;"
            "<span style='color:#E8F4FF;'>L</span>abeling"
        )
        acronym.setAlignment(Qt.AlignmentFlag.AlignCenter)
        acronym.setTextFormat(Qt.TextFormat.RichText)
        acronym.setStyleSheet(
            "font-size: 15px; font-weight: 700; letter-spacing: 2px; color: #64B5F6; padding-bottom: 6px;"
        )

        author = QLabel("Jobe Ritchie")
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #90CAF9; letter-spacing: 1px;"
        )

        institution = QLabel("University of North Carolina Chapel Hill")
        institution.setAlignment(Qt.AlignmentFlag.AlignCenter)
        institution.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #64B5F6;"
        )

        meta = QLabel("v0.5.0  ·  6/23/26")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #546E7A; padding-bottom: 10px;"
        )

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #1565C0; margin: 4px 0;")

        subtitle = QLabel(
            "Local-first behavior modeling and review for DLC-tracked rodent videos."
        )
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; font-weight: 600; color: #90CAF9; padding: 6px 0;")

        guidance = QLabel(
            "Core workflows: project setup, Active Learning, clip extraction/review, Direct Use, and exports.\n"
            "Optional dependencies extend preprocessing, GPU backends, and video fusion.\n"
            "Recommended order: create/open project -> install dependencies -> import data -> run workflows."
        )
        guidance.setWordWrap(True)
        guidance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        guidance.setStyleSheet("font-size: 11px; font-weight: 600; color: #78909C; padding-bottom: 10px;")

        # ── Action buttons ───────────────────────────────────────────────────
        self.create_btn = QPushButton("\u2795  Create New Project")
        self.open_btn = QPushButton("\ud83d\udcc2  Open Project")
        self.deps_btn = QPushButton("\ud83d\udce6  Dependencies")

        for btn in (self.create_btn, self.open_btn, self.deps_btn):
            btn.setMinimumHeight(38)

        self.btn_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.btn_row.setSpacing(10)
        self.btn_row.addWidget(self.create_btn)
        self.btn_row.addWidget(self.open_btn)
        self.btn_row.addWidget(self.deps_btn)

        # ── Recent projects ──────────────────────────────────────────────────
        recent_label = QLabel("Recent Projects")
        recent_label.setStyleSheet("font-weight: 700; font-size: 12px; color: #90CAF9; padding-top: 8px;")
        self.recent_list = QListWidget()
        self.recent_list.setToolTip("Double-click a recent project to open it.")
        self.recent_list.setMaximumHeight(160)

        self.create_btn.clicked.connect(self.create_project_requested.emit)
        self.open_btn.clicked.connect(self.open_project_requested.emit)
        self.deps_btn.clicked.connect(self.dependencies_requested.emit)
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self.recent_project_requested.emit(item.text())
        )

        # ── Layout ───────────────────────────────────────────────────────────
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(80, 30, 80, 30)
        self.outer.setSpacing(4)
        self.outer.addWidget(title)
        self.outer.addWidget(acronym)
        self.outer.addWidget(author)
        self.outer.addWidget(institution)
        self.outer.addWidget(meta)
        self.outer.addWidget(divider)
        self.outer.addWidget(subtitle)
        self.outer.addWidget(guidance)
        self.outer.addLayout(self.btn_row)
        self.outer.addSpacing(8)
        self.outer.addWidget(recent_label)
        self.outer.addWidget(self.recent_list)
        self.outer.addStretch(1)

        self._apply_responsive_layout()

    def set_recent_projects(self, paths: list[str]) -> None:
        self.recent_list.clear()
        self.recent_list.addItems(paths)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        width = max(0, self.width())
        compact = width < 1020

        horizontal_margin = 24 if compact else 80
        self.outer.setContentsMargins(horizontal_margin, 24, horizontal_margin, 24)

        if compact:
            self.btn_row.setDirection(QBoxLayout.Direction.TopToBottom)
            self.recent_list.setMaximumHeight(120)
        else:
            self.btn_row.setDirection(QBoxLayout.Direction.LeftToRight)
            self.recent_list.setMaximumHeight(160)
