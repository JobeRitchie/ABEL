"""Home/project overview tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_KV_KEY_STYLE = "font-size: 11px; font-weight: 600; color: #78909C;"
_KV_VAL_STYLE = "font-size: 11px; color: #B0BEC5;"


def _stat_label(value: str, description: str) -> QWidget:
    """Small vertical stat card."""
    w = QWidget()
    w.setStyleSheet(
        "background: #0F2744; border: 1px solid #1565C0; border-radius: 6px; padding: 6px;"
    )
    layout = QVBoxLayout(w)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.setSpacing(2)
    val_lbl = QLabel(value)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    val_lbl.setStyleSheet("font-size: 28px; font-weight: 900; color: #42A5F5; background: transparent; border: none;")
    desc_lbl = QLabel(description)
    desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    desc_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #78909C; background: transparent; border: none;")
    layout.addWidget(val_lbl)
    layout.addWidget(desc_lbl)
    return w


class HomeTab(QWidget):
    # Emitted when user clicks "Create Snapshot Workflow"
    snapshot_requested = Signal()
    # Emitted when user clicks "Open Another Project"
    open_project_requested = Signal()
    # Emitted when user clicks "Direct Use Workflow"
    direct_use_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._project_name = QLabel("Project: Not loaded")
        self._project_name.setStyleSheet("font-size: 16px; font-weight: 800; color: #90CAF9;")
        self._project_path = QLabel("Path: —")
        self._project_path.setStyleSheet("font-size: 11px; color: #546E7A;")
        self._status = QLabel(
            "Create or open a project to access Active Learning, clip review, temporal refinement, Direct Use, and exports."
        )
        self._status.setStyleSheet("font-size: 12px; font-weight: 700; color: #64B5F6; padding: 4px 0;")
        self._status.setWordWrap(True)

        # Stat cards
        stats_box = QGroupBox("Project Stats")
        stats_grid = QGridLayout(stats_box)
        stats_grid.setSpacing(8)

        self._stat_sessions = _stat_label("—", "Sessions")
        self._stat_behaviors = _stat_label("—", "Behaviors")
        self._stat_seeds = _stat_label("—", "Seeds")
        self._stat_clips = _stat_label("—", "Clips")

        stats_grid.addWidget(self._stat_sessions, 0, 0)
        stats_grid.addWidget(self._stat_behaviors, 0, 1)
        stats_grid.addWidget(self._stat_seeds, 0, 2)
        stats_grid.addWidget(self._stat_clips, 0, 3)

        # ── Pipeline Settings ─────────────────────────────────────────────
        self._pipeline_box = QGroupBox("Pipeline Settings")
        pipeline_grid = QGridLayout(self._pipeline_box)
        pipeline_grid.setSpacing(4)
        pipeline_grid.setContentsMargins(10, 8, 10, 8)
        pipeline_grid.setColumnStretch(1, 1)
        pipeline_grid.setColumnStretch(3, 1)

        self._pipeline_values: dict[str, QLabel] = {}
        _pipeline_specs = [
            ("window_disp",    "Window size"),
            ("stride_disp",    "Stride"),
            ("fps",            "Frame rate"),
            ("model_version",  "Active model"),
            ("classifier",     "Classifier"),
            ("query_mode",     "Query mode"),
        ]
        for i, (key, label_text) in enumerate(_pipeline_specs):
            row, pair = divmod(i, 2)
            col_k = pair * 2
            col_v = col_k + 1
            k_lbl = QLabel(f"{label_text}:")
            k_lbl.setStyleSheet(_KV_KEY_STYLE)
            k_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v_lbl = QLabel("—")
            v_lbl.setStyleSheet(_KV_VAL_STYLE)
            pipeline_grid.addWidget(k_lbl, row, col_k)
            pipeline_grid.addWidget(v_lbl, row, col_v)
            self._pipeline_values[key] = v_lbl

        # ── Model Performance ────────────────────────────────────────────
        self._metrics_box = QGroupBox("Model Performance  (frame level)")
        metrics_row = QHBoxLayout(self._metrics_box)
        metrics_row.setSpacing(6)
        metrics_row.setContentsMargins(10, 8, 10, 8)

        self._metric_card_labels: dict[str, QLabel] = {}
        _metric_specs = [
            ("frame_f1",        "F1 Score"),
            ("frame_precision", "Precision"),
            ("frame_recall",    "Recall"),
            ("frame_pr_auc",    "PR-AUC"),
            ("segment_f1",      "Segment F1"),
        ]
        for key, label_text in _metric_specs:
            card = QWidget()
            card.setStyleSheet(
                "background: #0A2236; border: 1px solid #1565C0; border-radius: 6px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)
            card_layout.setSpacing(2)
            val_lbl = QLabel("—")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(
                "font-size: 20px; font-weight: 800; color: #66BB6A;"
                " background: transparent; border: none;"
            )
            desc_lbl = QLabel(label_text)
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_lbl.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: #78909C;"
                " background: transparent; border: none;"
            )
            card_layout.addWidget(val_lbl)
            card_layout.addWidget(desc_lbl)
            metrics_row.addWidget(card)
            self._metric_card_labels[key] = val_lbl
        self._metrics_box.hide()

        # Modeled behaviors list
        behaviors_box = QGroupBox("Modeled Behaviors")
        behaviors_layout = QVBoxLayout(behaviors_box)
        self._behavior_list = QListWidget()
        self._behavior_list.setMaximumHeight(140)
        self._behavior_list.setStyleSheet(
            "QListWidget { background: #0A1929; border: 1px solid #1565C0; border-radius: 4px; color: #B0BEC5; font-size: 12px; }"
            "QListWidget::item { padding: 3px 6px; }"
        )
        self._no_behaviors_label = QLabel("No behaviors defined yet.")
        self._no_behaviors_label.setStyleSheet("color: #546E7A; font-size: 11px; font-style: italic; padding: 6px;")
        behaviors_layout.addWidget(self._no_behaviors_label)
        behaviors_layout.addWidget(self._behavior_list)
        self._behavior_list.hide()

        # Actions
        self.create_project_btn = QPushButton("Create New Project")
        self.open_folder_btn = QPushButton("📂 Open Project Folder")
        self.open_outputs_btn = QPushButton("📊 Show Output Files")
        self.open_models_btn = QPushButton("🧠 Open Model Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_outputs_btn.setEnabled(False)
        self.open_models_btn.setEnabled(False)

        self._open_project_btn = QPushButton("📁 Open Another Project")
        self._open_project_btn.setToolTip("Browse for and open an existing ABEL project")

        self._snapshot_btn = QPushButton("📸 Create Snapshot Workflow")
        self._snapshot_btn.setEnabled(False)
        self._snapshot_btn.setToolTip(
            "Export the current pipeline, model, and settings as a reusable Snapshot Workflow"
        )

        self._direct_use_btn = QPushButton("⚡ Direct Use Workflow")
        self._direct_use_btn.setToolTip(
            "Apply a trained model from any project to new data"
        )

        self._open_project_btn.clicked.connect(self.open_project_requested)
        self._snapshot_btn.clicked.connect(self.snapshot_requested)
        self._direct_use_btn.clicked.connect(self.direct_use_requested)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.create_project_btn)
        btn_row.addWidget(self._open_project_btn)
        btn_row.addWidget(self.open_folder_btn)
        btn_row.addWidget(self.open_outputs_btn)
        btn_row.addWidget(self.open_models_btn)
        btn_row.addStretch()

        btn_row_2 = QHBoxLayout()
        btn_row_2.addWidget(self._snapshot_btn)
        btn_row_2.addWidget(self._direct_use_btn)
        btn_row_2.addStretch()

        # Wrap everything in a scroll area so the page handles any screen height
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        content_layout.addWidget(self._project_name)
        content_layout.addWidget(self._project_path)
        content_layout.addWidget(self._status)
        content_layout.addWidget(stats_box)
        content_layout.addWidget(self._pipeline_box)
        content_layout.addWidget(self._metrics_box)
        content_layout.addWidget(behaviors_box)
        content_layout.addLayout(btn_row)
        content_layout.addLayout(btn_row_2)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)
        layout.addWidget(scroll)

    def update_project(self, name: str, path: Path) -> None:
        self._project_name.setText(f"Project: {name}")
        self._project_path.setText(f"Path: {path}")
        self._status.setText(
            "Ready - import data, define behaviors, run Active Learning, create Snapshot workflows, apply Direct Use, and export results."
        )
        self.open_folder_btn.setEnabled(True)
        self.open_outputs_btn.setEnabled(True)
        self.open_models_btn.setEnabled(True)
        self._snapshot_btn.setEnabled(True)

    def update_stats(self, stats: dict) -> None:
        def _val(key: str) -> str:
            v = stats.get(key)
            return str(v) if v is not None else "0"

        # Update the value label inside each stat card
        for card, key in (
            (self._stat_sessions, "sessions"),
            (self._stat_behaviors, "behaviors"),
            (self._stat_seeds, "seeds"),
            (self._stat_clips, "clips"),
        ):
            val_lbl = card.layout().itemAt(0).widget()
            if val_lbl:
                val_lbl.setText(_val(key))

        # Pipeline settings
        pipeline = stats.get("pipeline") or {}
        if pipeline:
            fps = pipeline.get("fps") or 30.0
            win_fr = pipeline.get("window_frames") or 0
            str_fr = pipeline.get("stride_frames") or 0
            win_sec = pipeline.get("window_sec")
            str_sec = pipeline.get("stride_sec")
            win_disp = (
                f"{win_fr} fr  ({win_sec} s)" if win_sec is not None and win_fr
                else (f"{win_fr} fr" if win_fr else "—")
            )
            str_disp = (
                f"{str_fr} fr  ({str_sec} s)" if str_sec is not None and str_fr
                else (f"{str_fr} fr" if str_fr else "—")
            )
            self._pipeline_values["window_disp"].setText(win_disp)
            self._pipeline_values["stride_disp"].setText(str_disp)
            self._pipeline_values["fps"].setText(f"{fps} fps")
            self._pipeline_values["model_version"].setText(str(pipeline.get("model_version") or "—"))
            self._pipeline_values["classifier"].setText(str(pipeline.get("classifier") or "—"))
            self._pipeline_values["query_mode"].setText(str(pipeline.get("query_mode") or "—"))

        # Model performance metrics
        model_metrics = stats.get("model_metrics")
        if model_metrics:
            for key, lbl in self._metric_card_labels.items():
                lbl.setText(str(model_metrics.get(key) or "—"))
            self._metrics_box.show()
        else:
            self._metrics_box.hide()

        # Update behavior names list
        behavior_names = stats.get("behavior_names", [])
        self._behavior_list.clear()
        if behavior_names:
            self._no_behaviors_label.hide()
            self._behavior_list.show()
            for name in behavior_names:
                item = QListWidgetItem(f"  •  {name}")
                self._behavior_list.addItem(item)
        else:
            self._no_behaviors_label.show()
            self._behavior_list.hide()

