"""Settings tab for global app settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import AppSettings
from abel.services.settings_service import SettingsService


class SettingsTab(QWidget):
    def __init__(
        self,
        settings_service: SettingsService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = settings_service
        self._project_root: Path | None = None

        self._autosave = QSpinBox()
        self._autosave.setRange(5, 600)

        self._recent = QSpinBox()
        self._recent.setRange(1, 50)

        self._updates = QCheckBox("Check updates on startup")

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save)

        box = QGroupBox("App Settings")
        form = QFormLayout(box)
        form.addRow("Autosave seconds:", self._autosave)
        form.addRow("Max recent projects:", self._recent)
        form.addRow("", self._updates)

        self._status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(save_btn)
        layout.addWidget(self._status)
        layout.addStretch()

        self._load()

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root

    def _load(self) -> None:
        cfg = self._service.load_app_settings()
        self._autosave.setValue(cfg.autosave_seconds)
        self._recent.setValue(cfg.max_recent_projects)
        self._updates.setChecked(cfg.check_updates_on_startup)

    def _save(self) -> None:
        cfg = AppSettings(
            autosave_seconds=self._autosave.value(),
            max_recent_projects=self._recent.value(),
            check_updates_on_startup=self._updates.isChecked(),
        )
        self._service.save_app_settings(cfg)
        self._status.setText("Settings saved.")
        QMessageBox.information(self, "Settings", "Settings saved.")
