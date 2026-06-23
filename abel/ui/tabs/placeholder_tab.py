"""Generic placeholder tab for upcoming workflow stages."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderTab(QWidget):
    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(f"<h3>{title}</h3><p>{description}</p>")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
