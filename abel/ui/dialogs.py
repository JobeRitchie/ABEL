"""Dialogs for project creation and app workflows."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import ProjectConfig, SourceMode


class ProjectWizardDialog(QDialog):
    """Collects initial project configuration from the user."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create New ABEL Project")

        self.project_name = QLineEdit()
        self.root_path = QLineEdit(str(Path.home()))
        self.assay_name = QLineEdit("open_field")
        self.species = QLineEdit("mouse")
        self.single_animal = QCheckBox("Single-animal project")
        self.single_animal.setChecked(True)

        self.default_fps = QSpinBox()
        self.default_fps.setRange(1, 240)
        self.default_fps.setValue(30)

        self.clip_duration = QLineEdit("2.0")
        self.crop_margin = QSpinBox()
        self.crop_margin.setRange(0, 1000)
        self.crop_margin.setValue(40)

        self.video_mode = QComboBox()
        self.video_mode.addItems([SourceMode.REFERENCE.value, SourceMode.COPY.value])
        self.pose_mode = QComboBox()
        self.pose_mode.addItems([SourceMode.REFERENCE.value, SourceMode.COPY.value])

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._choose_root)

        buttons = QHBoxLayout()
        create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        create_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(create_btn)
        buttons.addWidget(cancel_btn)

        root_row = QHBoxLayout()
        root_row.addWidget(self.root_path)
        root_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow("Project name", self.project_name)
        form.addRow("Project root", root_row)
        form.addRow("Assay", self.assay_name)
        form.addRow("Species", self.species)
        form.addRow("Animal mode", self.single_animal)
        form.addRow("Default FPS", self.default_fps)
        form.addRow("Default clip duration (sec)", self.clip_duration)
        form.addRow("Default crop margin (px)", self.crop_margin)
        form.addRow("Video import mode", self.video_mode)
        form.addRow("Pose import mode", self.pose_mode)

        form_container = QWidget()
        form_container.setLayout(form)

        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setWidget(form_container)

        layout = QVBoxLayout(self)
        layout.addWidget(form_scroll)
        layout.addLayout(buttons)

        self._apply_screen_aware_size()

    def _apply_screen_aware_size(self) -> None:
        """Ensure dialog remains fully reachable on smaller screens."""
        app = QApplication.instance()
        if app is None:
            self.resize(680, 560)
            return

        screen = self.screen() or app.primaryScreen()
        if screen is None:
            self.resize(680, 560)
            return

        available = screen.availableGeometry()
        target_width = min(680, max(520, int(available.width() * 0.7)))
        target_height = min(560, max(420, int(available.height() * 0.75)))
        self.resize(target_width, target_height)

    def to_project_config(self) -> tuple[Path, ProjectConfig]:
        root = Path(self.root_path.text().strip())
        cfg = ProjectConfig(
            project_name=self.project_name.text().strip(),
            assay_name=self.assay_name.text().strip(),
            species=self.species.text().strip(),
            single_animal=self.single_animal.isChecked(),
            expected_pose_formats=["csv", "h5"],
            default_fps=float(self.default_fps.value()),
            default_clip_duration_sec=float(self.clip_duration.text().strip() or 2.0),
            default_crop_margin_px=int(self.crop_margin.value()),
            video_source_mode=SourceMode(self.video_mode.currentText()),
            pose_source_mode=SourceMode(self.pose_mode.currentText()),
        )
        cfg.behavior_model.segment_window_frames = max(8, round(cfg.default_clip_duration_sec * cfg.default_fps))
        return root, cfg

    def _choose_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select project root")
        if selected:
            self.root_path.setText(selected)
