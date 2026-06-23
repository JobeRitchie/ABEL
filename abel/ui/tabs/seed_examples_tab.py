"""Seed Examples tab — annotate short behavior bouts directly on video.

Video playback uses OpenCV frame-by-frame rendering (degrades gracefully when
OpenCV is not installed — the frame-number fields remain usable without preview).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from abel.models.schemas import SeedExample
from abel.services.behavior_service import BehaviorService
from abel.services.import_service import ImportService
from abel.services.seed_service import SeedService
from abel.storage.file_store import read_yaml

logger = logging.getLogger("abel")


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Frame player widget
# ---------------------------------------------------------------------------

class _FramePlayer(QWidget):
    """Minimal frame-by-frame video player backed by OpenCV."""

    frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap = None
        self._n_frames = 0
        self._fps = 30.0
        self._cur_frame = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._last_bgr = None

        self._display = QLabel("No video loaded")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(320, 240)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display.setStyleSheet("background: #060E18; color: #546E7A; font-size: 13px;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self.toggle_play)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(lambda: self.seek(self._cur_frame - 1))

        self._next_btn = QPushButton("▶▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(lambda: self.seek(self._cur_frame + 1))

        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setStyleSheet("font-size: 11px; font-weight: 600;")

        ctrl = QHBoxLayout()
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._next_btn)
        ctrl.addWidget(self._slider, 1)
        ctrl.addWidget(self._frame_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display, 1)
        layout.addLayout(ctrl)

        self._set_controls_enabled(False)

    # Public API -------------------------------------------------------

    def load_video(self, path: str) -> bool:
        self.close_video()
        if not _has_cv2():
            self._display.setText(
                "OpenCV not installed.\nInstall opencv-python in the\nDependencies tab to enable preview."
            )
            return False
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._display.setText(f"Cannot open:\n{path}")
            return False
        self._cap = cap
        self._n_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._slider.setMaximum(self._n_frames - 1)
        self._set_controls_enabled(True)
        self.seek(0)
        return True

    def close_video(self) -> None:
        self._playing = False
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._n_frames = 0
        self._cur_frame = 0
        self._slider.setMaximum(0)
        self._display.setText("No video loaded")
        self._frame_label.setText("Frame: 0 / 0")
        self._set_controls_enabled(False)

    @property
    def current_frame(self) -> int:
        return self._cur_frame

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def fps(self) -> float:
        return self._fps

    def seek(self, frame: int) -> None:
        if self._cap is None:
            return
        frame = max(0, min(frame, self._n_frames - 1))
        import cv2  # noqa: PLC0415
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, bgr = self._cap.read()
        if ret:
            self._cur_frame = frame
            self._render(bgr)
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._frame_label.setText(f"Frame: {frame} / {self._n_frames - 1}")
        self.frame_changed.emit(frame)

    def toggle_play(self) -> None:
        if self._cap is None:
            return
        self._playing = not self._playing
        self._play_btn.setText("⏸" if self._playing else "▶")
        if self._playing:
            interval = max(1, int(1000 / self._fps))
            self._timer.start(interval)
        else:
            self._timer.stop()

    # Private ----------------------------------------------------------

    def _advance(self) -> None:
        if self._cur_frame >= self._n_frames - 1:
            self.toggle_play()
            return
        self.seek(self._cur_frame + 1)

    def _on_slider(self, value: int) -> None:
        if self._cap is not None:
            self.seek(value)

    def resizeEvent(self, event) -> None:
        if self._last_bgr is not None:
            self._render(self._last_bgr)
        super().resizeEvent(event)

    def _render(self, bgr) -> None:
        import cv2  # noqa: PLC0415
        self._last_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        dw = max(1, self._display.width())
        dh = max(1, self._display.height())
        pix = QPixmap.fromImage(qimg).scaled(
            dw, dh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._display.setPixmap(pix)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self._play_btn, self._prev_btn, self._next_btn, self._slider):
            w.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class SeedExamplesTab(QWidget):
    """Annotate short seed examples directly on imported video sessions."""

    def __init__(
        self,
        seed_service: SeedService,
        behavior_service: BehaviorService,
        import_service: ImportService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._seeds = seed_service
        self._behaviors = behavior_service
        self._imports = import_service
        self._project_root: Path | None = None
        self._manifest = None
        self._co_occurring_enabled: bool = False

        # ── Left: session selector + seed list ─────────────────────────
        session_label = QLabel("Session:")
        self._session_combo = QComboBox()
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        refresh_btn = QPushButton("⟳ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.setToolTip("Reload sessions from the import manifest")
        refresh_btn.clicked.connect(self.refresh)
        load_video_btn = QPushButton("▶ Load Video")
        load_video_btn.setFixedWidth(100)
        load_video_btn.setToolTip("Load the selected session's video into the player")
        load_video_btn.clicked.connect(self._load_current_session)

        self._seed_table = QTableWidget(0, 4)
        self._seed_table.setHorizontalHeaderLabels(["Behavior", "Start", "End", "Label"])
        self._seed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._seed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._seed_table.setColumnWidth(0, 120)
        self._seed_table.setColumnWidth(1, 65)
        self._seed_table.setColumnWidth(2, 65)
        self._seed_table.setColumnWidth(3, 70)
        self._seed_table.setToolTip("Double-click a row to jump the video to that seed's start frame")
        self._seed_table.cellDoubleClicked.connect(self._on_seed_double_clicked)

        copy_seed_btn = QPushButton("Copy Seed to Other Subjects…")
        copy_seed_btn.setToolTip(
            "Duplicate the selected seed's behavior, frame range, and label to\n"
            "other subjects' sessions. Useful when a behavior occurs at the same\n"
            "frame range across multiple videos."
        )
        copy_seed_btn.clicked.connect(self._copy_seed_to_sessions)

        delete_seed_btn = QPushButton("Delete Selected Seed")
        delete_seed_btn.clicked.connect(self._delete_seed)

        self._relink_btn = QPushButton("⚠  Relink Orphaned Seeds")
        self._relink_btn.setToolTip(
            "Some seeds reference session IDs that no longer exist in the manifest.\n"
            "Click to remap them to the correct current sessions."
        )
        self._relink_btn.setStyleSheet(
            "QPushButton { background-color: #3E2000; color: #FFB74D; "
            "border: 1px solid #E65100; border-radius: 4px; padding: 4px 10px; font-weight: 600; }"
            "QPushButton:hover { background-color: #5E3000; }"
        )
        self._relink_btn.hide()
        self._relink_btn.clicked.connect(self._relink_orphaned_seeds)

        left_layout = QVBoxLayout()
        session_row = QHBoxLayout()
        session_row.addWidget(session_label)
        session_row.addWidget(self._session_combo, 1)
        session_row.addWidget(load_video_btn)
        session_row.addWidget(refresh_btn)
        left_layout.addLayout(session_row)
        self._seed_status = QLabel("Seeds in this project: 0")
        self._seed_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #78909C; padding: 2px 0;")
        left_layout.addWidget(self._seed_status)
        left_layout.addWidget(self._relink_btn)
        left_layout.addWidget(self._seed_table, 1)
        left_layout.addWidget(copy_seed_btn)
        left_layout.addWidget(delete_seed_btn)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ── Right: player + annotation controls ────────────────────────
        self._player = _FramePlayer()
        self._player.frame_changed.connect(self._on_frame_changed)

        # Annotation controls
        ann_group = QGroupBox("Mark Seed Bout")
        ann_layout = QFormLayout(ann_group)

        self._behavior_combo = QComboBox()
        self._behavior_combo.currentIndexChanged.connect(self._sync_assume_negatives_btn)

        # Multi-select list shown only in co-occurring mode
        self._behavior_list = QListWidget()
        self._behavior_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._behavior_list.setMaximumHeight(90)
        self._behavior_list.setToolTip(
            "Select one or more behaviors for this bout window.\n"
            "Hold Ctrl or Shift to multi-select, or click individual rows to toggle.\n"
            "A separate seed entry will be added for each selected behavior."
        )
        self._behavior_list.setVisible(False)
        self._co_occurring_label = QLabel("Behaviors (multi-select):")
        self._co_occurring_label.setVisible(False)

        self._start_spin = _FrameSpinBox()
        self._end_spin = _FrameSpinBox()
        mark_start_btn = QPushButton("⬅ Mark Start (current frame)")
        mark_end_btn = QPushButton("Mark End (current frame) ➡")
        mark_start_btn.clicked.connect(lambda: self._start_spin.setValue(self._player.current_frame))
        mark_end_btn.clicked.connect(lambda: self._end_spin.setValue(self._player.current_frame))

        _pos_style = (
            "QPushButton{background:#0D2B0D;color:#81C784;border:2px solid #388E3C;"
            "border-radius:4px;font-weight:700;padding:5px 10px;}"
            "QPushButton:checked{background:#2E7D32;color:#FFFFFF;border:2px solid #A5D6A7;}"
        )
        _neg_style = (
            "QPushButton{background:#2B0D0D;color:#EF9A9A;border:2px solid #C62828;"
            "border-radius:4px;font-weight:700;padding:5px 10px;}"
            "QPushButton:checked{background:#B71C1C;color:#FFFFFF;border:2px solid #EF9A9A;}"
        )
        self._label_pos = QPushButton("✓ Positive")
        self._label_neg = QPushButton("✗ Negative")
        for btn, style in (
            (self._label_pos, _pos_style),
            (self._label_neg, _neg_style),
        ):
            btn.setCheckable(True)
            btn.setStyleSheet(style)
        self._label_pos.setChecked(True)
        label_grp = QButtonGroup(self)
        label_grp.setExclusive(True)
        label_grp.addButton(self._label_pos)
        label_grp.addButton(self._label_neg)
        label_row = QHBoxLayout()
        label_row.addWidget(self._label_pos)
        label_row.addWidget(self._label_neg)

        add_btn = QPushButton("＋ Add Seed")
        add_btn.clicked.connect(self._add_seed)

        self._assume_neg_btn = QPushButton("🔄 Assume All Non-Positive Frames = Negative")
        self._assume_neg_btn.setCheckable(True)
        self._assume_neg_btn.setToolTip(
            "When enabled, every frame NOT covered by a positive seed is treated as\n"
            "a negative example during training. You only need to mark positive bouts."
        )
        self._assume_neg_btn.setStyleSheet(
            "QPushButton{background:#0D1B2A;color:#78909C;border:1px solid #37474F;"
            "border-radius:4px;font-weight:600;padding:5px;}"
            "QPushButton:checked{background:#0D2B3E;color:#4FC3F7;border:1px solid #0288D1;}"
        )
        self._assume_neg_btn.toggled.connect(self._on_assume_negatives_toggled)

        ann_layout.addRow("Behavior:", self._behavior_combo)
        ann_layout.addRow(self._co_occurring_label, self._behavior_list)
        ann_layout.addRow("Start frame:", self._start_spin)
        ann_layout.addRow("End frame:", self._end_spin)
        ann_layout.addRow("", mark_start_btn)
        ann_layout.addRow("", mark_end_btn)
        ann_layout.addRow("Label:", label_row)
        ann_layout.addRow("", add_btn)
        ann_layout.addRow("", self._assume_neg_btn)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._player, 3)
        right_layout.addWidget(ann_group)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 640])

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        no_project = QLabel(
            "Open or create a project first, then import sessions to annotate seed examples."
        )
        no_project.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_project.setWordWrap(True)
        no_project.setStyleSheet("color: #546E7A; font-size: 13px; padding: 20px;")
        self._no_project_label = no_project
        root.addWidget(no_project)
        root.addWidget(splitter)
        splitter.hide()
        self._splitter = splitter

        # Tab-level keyboard shortcuts — active whenever this tab is visible,
        # regardless of which child widget has focus.
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(
            self._player.toggle_play
        )
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(
            lambda: self._player.seek(self._player.current_frame - 1)
        )
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            lambda: self._player.seek(self._player.current_frame + 1)
        )
        # Hold Shift for 10-frame jumps
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(
            lambda: self._player.seek(self._player.current_frame - 10)
        )
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(
            lambda: self._player.seek(self._player.current_frame + 10)
        )
        # S / E  = mark start / end at current frame
        QShortcut(QKeySequence(Qt.Key.Key_S), self).activated.connect(
            lambda: self._start_spin.setValue(self._player.current_frame)
        )
        QShortcut(QKeySequence(Qt.Key.Key_E), self).activated.connect(
            lambda: self._end_spin.setValue(self._player.current_frame)
        )

    # ------------------------------------------------------------------
    # Project binding
    # ------------------------------------------------------------------

    def set_project(self, project_root: Path) -> None:
        self._project_root = project_root
        self._no_project_label.hide()
        self._splitter.show()
        # Defer I/O to avoid blocking the tab switch.
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        QTimer.singleShot(0, lambda: self._deferred_project_init(project_root))

    def _deferred_project_init(self, project_root: Path) -> None:
        if self._project_root != project_root:
            return
        self._seeds.set_project(project_root)
        self._manifest = self._imports.load_manifest(project_root)
        # Backfill the session registry from the current manifest so it always
        # has an up-to-date record even for projects created before the registry existed.
        if self._manifest:
            self._imports.update_registry(project_root, self._manifest)
        self._load_co_occurring_setting()
        self.refresh()
        # If the default-selected session has no seeds but others do, jump to the
        # first session that actually has seeds so the user sees their work immediately.
        self._auto_select_session_with_seeds()
        self._check_orphaned_seeds()
        n = len(self._seeds.seeds)
        logger.info("Seed examples loaded: %d from %s", n, project_root)

    def _load_co_occurring_setting(self) -> None:
        self._co_occurring_enabled = False
        if not self._project_root:
            return
        cfg_path = self._project_root / "project.yaml"
        if not cfg_path.exists():
            return
        raw = read_yaml(cfg_path, {})
        model = raw.get("behavior_model") or {}
        self._co_occurring_enabled = bool(model.get("allow_co_occurring_behaviors", False))
        # Toggle which behavior selector is visible
        self._behavior_combo.setVisible(not self._co_occurring_enabled)
        self._behavior_list.setVisible(self._co_occurring_enabled)
        self._co_occurring_label.setVisible(self._co_occurring_enabled)

    def refresh(self) -> None:
        self._refresh_behavior_combo()
        self._refresh_session_combo()
        self._refresh_seed_table()

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _refresh_behavior_combo(self) -> None:
        self._behavior_combo.blockSignals(True)
        current = self._behavior_combo.currentText()
        self._behavior_combo.clear()
        self._behavior_list.clear()
        for b in self._behaviors.behaviors:
            self._behavior_combo.addItem(b.name, userData=b.behavior_id)
            from PySide6.QtWidgets import QListWidgetItem  # noqa: PLC0415
            item = QListWidgetItem(b.name)
            item.setData(Qt.ItemDataRole.UserRole, b.behavior_id)
            self._behavior_list.addItem(item)
        idx = self._behavior_combo.findText(current)
        if idx >= 0:
            self._behavior_combo.setCurrentIndex(idx)
        self._behavior_combo.blockSignals(False)
        self._sync_assume_negatives_btn()

    def _sync_assume_negatives_btn(self) -> None:
        """Sync the assume-negatives toggle to the persisted value for the current behavior."""
        bid = self._behavior_combo.currentData()
        if bid is None:
            return
        state = self._seeds.get_assume_negative(bid)
        self._assume_neg_btn.blockSignals(True)
        self._assume_neg_btn.setChecked(state)
        self._assume_neg_btn.blockSignals(False)

    def _on_assume_negatives_toggled(self, checked: bool) -> None:
        bid = self._behavior_combo.currentData()
        if bid:
            self._seeds.set_assume_negative(bid, checked)

    def _refresh_session_combo(self) -> None:
        # Always re-read the manifest so newly imported sessions appear immediately.
        if self._project_root:
            self._manifest = self._imports.load_manifest(self._project_root)
        self._session_combo.blockSignals(True)
        current = self._session_combo.currentText()
        self._session_combo.clear()
        if self._manifest:
            for s in self._manifest.linked_sessions:
                video = next(
                    (v for v in self._manifest.videos if v.asset_id == s.video_asset_id), None
                )
                label = Path(video.source_path).name if video else s.session_id
                self._session_combo.addItem(label, userData=s.session_id)
        idx = self._session_combo.findText(current)
        if idx >= 0:
            self._session_combo.setCurrentIndex(idx)
        self._session_combo.blockSignals(False)
        # Auto-load the selected session's video now that signals are unblocked
        self._load_current_session()

    # ------------------------------------------------------------------
    # Orphaned seed detection & relinking
    # ------------------------------------------------------------------

    def _orphaned_session_ids(self) -> set[str]:
        """Return seed session IDs that have no matching entry in the current manifest."""
        if not self._manifest:
            return set()
        manifest_ids = {s.session_id for s in self._manifest.linked_sessions}
        seed_ids = {s.session_id for s in self._seeds.seeds}
        return seed_ids - manifest_ids

    def _check_orphaned_seeds(self) -> None:
        """Show/hide the relink banner depending on whether orphaned seeds exist."""
        orphaned = self._orphaned_session_ids()
        if orphaned:
            n = sum(1 for s in self._seeds.seeds if s.session_id in orphaned)
            self._relink_btn.setText(f"\u26a0  {n} seeds have missing sessions \u2014 click to relink")
            self._relink_btn.show()
        else:
            self._relink_btn.hide()

    def _relink_orphaned_seeds(self) -> None:
        """Use the session registry to auto-match orphaned seed session IDs, then
        show a dialog only for any that still can't be resolved automatically."""
        orphaned = self._orphaned_session_ids()
        if not orphaned or not self._manifest or not self._project_root:
            QMessageBox.information(self, "No Orphaned Seeds", "All seed sessions are present in the manifest.")
            return

        # ── Auto-match via registry ─────────────────────────────────────
        auto_map: dict[str, str] = {}    # old_id → new_id  (resolved automatically)
        need_manual: set[str] = set()    # old_ids that still need user help

        for old_id in orphaned:
            new_id = self._imports.find_new_session_for_video(
                self._project_root, old_id, self._manifest
            )
            if new_id:
                auto_map[old_id] = new_id
            else:
                need_manual.add(old_id)

        # ── Manual dialog for anything the registry couldn't resolve ────
        manual_map: dict[str, str] = {}
        if need_manual:
            # Build label for each current manifest session: prefer video filename
            manifest_options: list[tuple[str, str]] = []
            for s in self._manifest.linked_sessions:
                video = next((v for v in self._manifest.videos if v.asset_id == s.video_asset_id), None)
                label = Path(video.source_path).name if video else s.session_id
                manifest_options.append((label, s.session_id))
            manifest_options.sort(key=lambda x: x[0])

            dlg = QDialog(self)
            dlg.setWindowTitle("Relink Orphaned Seeds")
            dlg.setMinimumWidth(560)
            layout = QVBoxLayout(dlg)

            auto_note = ""
            if auto_map:
                resolved_lines = "".join(
                    f"<li>{old} → {new}</li>"
                    for old, new in auto_map.items()
                )
                auto_note = (
                    f"<b style='color:#81C784;'>Auto-resolved via session registry ({len(auto_map)}):</b>"
                    f"<ul>{resolved_lines}</ul>"
                )

            header = QLabel(
                "<b>Seeds found with session IDs no longer in the import manifest.</b><br>"
                "This happens when videos are re-imported (new random session IDs are generated).<br><br>"
                + auto_note +
                "<b>The following could not be matched automatically — please choose:</b>"
            )
            header.setWordWrap(True)
            header.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(header)

            combos: dict[str, QComboBox] = {}
            form = QFormLayout()
            for old_id in sorted(need_manual):
                n = sum(1 for s in self._seeds.seeds if s.session_id == old_id)
                combo = QComboBox()
                combo.addItem("-- skip (leave as-is) --", userData=None)
                for label, new_id in manifest_options:
                    combo.addItem(label, userData=new_id)
                combos[old_id] = combo
                form.addRow(f"{old_id}\n({n} seeds):", combo)
            layout.addLayout(form)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            for old_id, combo in combos.items():
                new_id = combo.currentData()
                if new_id:
                    manual_map[old_id] = new_id

        # ── Apply all remappings ────────────────────────────────────────
        full_map = {**auto_map, **manual_map}
        remapped = 0
        for old_id, new_id in full_map.items():
            for seed in self._seeds.seeds:
                if seed.session_id == old_id:
                    self._seeds.update(seed.seed_id, seed.model_copy(update={"session_id": new_id}))
                    remapped += 1

        self._check_orphaned_seeds()
        self._refresh_session_combo()
        self._auto_select_session_with_seeds()
        self._refresh_seed_table()

        if remapped:
            auto_count = sum(
                1 for s in self._seeds.seeds
                if s.session_id in auto_map.values()
            )
            QMessageBox.information(
                self, "Relink Complete",
                f"Remapped {remapped} seed(s).\n"
                f"  Auto-resolved via registry: {len(auto_map)} session(s)\n"
                f"  Manually mapped: {len(manual_map)} session(s)"
            )

    def _auto_select_session_with_seeds(self) -> None:
        """If the currently selected session has no seeds, jump to the first that does."""
        if self._selected_session_id() is not None:
            current_session_seeds = [
                s for s in self._seeds.seeds
                if s.session_id == self._selected_session_id()
            ]
            if current_session_seeds:
                return  # Already on a session with seeds — nothing to do
        # Find the first combo item whose session_id has at least one seed
        session_ids_with_seeds = {s.session_id for s in self._seeds.seeds}
        for i in range(self._session_combo.count()):
            sid = self._session_combo.itemData(i)
            if sid in session_ids_with_seeds:
                self._session_combo.setCurrentIndex(i)
                self._refresh_seed_table()
                return

    def _refresh_seed_table(self) -> None:
        behavior_by_id = {b.behavior_id: b.name for b in self._behaviors.behaviors}
        selected_session_id = self._selected_session_id()
        visible_seeds = [
            seed for seed in self._seeds.seeds
            if selected_session_id is not None and seed.session_id == selected_session_id
        ]
        self._seed_table.setRowCount(0)
        for seed in visible_seeds:
            row = self._seed_table.rowCount()
            self._seed_table.insertRow(row)
            self._seed_table.setItem(row, 0, QTableWidgetItem(behavior_by_id.get(seed.behavior_id, seed.behavior_id)))
            self._seed_table.setItem(row, 1, QTableWidgetItem(str(seed.start_frame)))
            self._seed_table.setItem(row, 2, QTableWidgetItem(str(seed.end_frame)))
            self._seed_table.setItem(row, 3, QTableWidgetItem(seed.label_type))
            item = self._seed_table.item(row, 0)
            if item:
                item.setData(Qt.ItemDataRole.UserRole, seed.seed_id)
        total = len(self._seeds.seeds)
        n = len(visible_seeds)
        suffix = f"  ({total} total in project)" if total != n else ""
        self._seed_status.setText(f"Seeds in selected session: {n}{suffix}")

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def _on_session_changed(self, idx: int) -> None:
        self._load_current_session()
        self._refresh_seed_table()

    def _load_current_session(self) -> None:
        """Load the video for the currently selected session into the player."""
        idx = self._session_combo.currentIndex()
        if idx < 0 or not self._manifest or not self._project_root:
            return
        session_id = self._session_combo.itemData(idx)
        if not session_id:
            return
        video_path = self._imports.video_path_for_session(self._manifest, session_id)
        if video_path and video_path.exists():
            self._player.load_video(str(video_path))
        else:
            self._player.close_video()
            if video_path:
                logger.warning("Video not found: %s", video_path)

    def _on_seed_double_clicked(self, row: int, _col: int) -> None:
        """Seek the video player to the start frame of the double-clicked seed row."""
        start_item = self._seed_table.item(row, 1)
        if start_item is None:
            return
        try:
            frame = int(start_item.text())
        except ValueError:
            return
        self._player.seek(frame)

    def _on_frame_changed(self, frame: int) -> None:
        pass  # Could be used to update timeline overlays in future

    # ------------------------------------------------------------------
    # Seed actions
    # ------------------------------------------------------------------

    def _add_seed(self) -> None:
        if self._behavior_combo.count() == 0 and self._behavior_list.count() == 0:
            QMessageBox.warning(self, "No Behaviors", "Define behaviors first in the Behavior Definitions tab.")
            return

        session_id = self._session_combo.currentData() or "unknown"
        start_frame = self._start_spin.value()
        end_frame = self._end_spin.value()

        if end_frame <= start_frame:
            QMessageBox.warning(self, "Invalid Range", "End frame must be greater than start frame.")
            return

        label_type = "positive" if self._label_pos.isChecked() else "negative"

        if self._co_occurring_enabled:
            # Collect all selected behaviors from the list widget
            selected_items = self._behavior_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Behavior Selected", "Select at least one behavior in the list.")
                return
            behavior_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        else:
            behavior_ids = [self._behavior_combo.currentData()]

        for behavior_id in behavior_ids:
            seed = SeedExample(
                seed_id="",
                behavior_id=behavior_id,
                session_id=session_id,
                start_frame=start_frame,
                end_frame=end_frame,
                label_type=label_type,
                quality_flag="clean",
                notes="",
            )
            self._seeds.add(seed)
            logger.info("Seed added and saved: %s %s-%s", behavior_id, start_frame, end_frame)

        self._refresh_seed_table()
        # Brief "Saved" confirmation
        session_count = self._seed_table.rowCount()
        added = len(behavior_ids)
        suffix = f" ({added} behaviors)" if added > 1 else ""
        self._seed_status.setText(f"Seeds in selected session: {session_count}  ✓ Saved{suffix}")
        self._seed_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #66BB6A; padding: 2px 0;")
        QTimer.singleShot(2000, self._reset_seed_status)

    def _reset_seed_status(self) -> None:
        n = self._seed_table.rowCount()
        self._seed_status.setText(f"Seeds in selected session: {n}")
        self._seed_status.setStyleSheet("font-size: 11px; font-weight: 600; color: #78909C; padding: 2px 0;")

    def _selected_seed(self) -> SeedExample | None:
        """Return the SeedExample for the currently selected table row, if any."""
        row = self._seed_table.currentRow()
        if row < 0:
            return None
        item = self._seed_table.item(row, 0)
        if item is None:
            return None
        seed_id = item.data(Qt.ItemDataRole.UserRole)
        return self._seeds.get(seed_id) if seed_id else None

    def _copy_seed_to_sessions(self) -> None:
        """Copy the selected seed's behavior/frame-range/label to other subjects."""
        seed = self._selected_seed()
        if seed is None:
            QMessageBox.information(
                self, "No Seed Selected", "Select a seed in the table to copy."
            )
            return
        if not self._manifest:
            return

        behavior_name = {
            b.behavior_id: b.name for b in self._behaviors.behaviors
        }.get(seed.behavior_id, seed.behavior_id)

        # Build the list of candidate target sessions (everything except the source).
        options: list[tuple[str, str]] = []  # (label, session_id)
        for s in self._manifest.linked_sessions:
            if s.session_id == seed.session_id:
                continue
            video = next(
                (v for v in self._manifest.videos if v.asset_id == s.video_asset_id), None
            )
            fname = Path(video.source_path).name if video else s.session_id
            label = f"{s.subject_id} — {fname}" if s.subject_id else fname
            options.append((label, s.session_id))

        if not options:
            QMessageBox.information(
                self, "No Other Sessions",
                "There are no other sessions to copy this seed to.",
            )
            return
        options.sort(key=lambda x: x[0].lower())

        from PySide6.QtWidgets import QListWidgetItem  # noqa: PLC0415

        dlg = QDialog(self)
        dlg.setWindowTitle("Copy Seed to Other Subjects")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)

        header = QLabel(
            f"Copy seed <b>{behavior_name}</b> "
            f"(frames {seed.start_frame}–{seed.end_frame}, {seed.label_type}) "
            "to the selected subjects' sessions:"
        )
        header.setWordWrap(True)
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        target_list = QListWidget()
        target_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for label, sid in options:
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, sid)
            target_list.addItem(it)
        layout.addWidget(target_list, 1)

        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select All")
        sel_none_btn = QPushButton("Select None")
        sel_all_btn.clicked.connect(target_list.selectAll)
        sel_none_btn.clicked.connect(target_list.clearSelection)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        target_ids = [
            target_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(target_list.count())
            if target_list.item(i).isSelected()
        ]
        if not target_ids:
            return

        created = self._seeds.copy_to_sessions(seed.seed_id, target_ids)
        self._refresh_seed_table()
        skipped = len(target_ids) - len(created)
        msg = f"Copied seed to {len(created)} session(s)."
        if skipped:
            msg += f"\n{skipped} skipped (already had an identical seed)."
        QMessageBox.information(self, "Seed Copied", msg)

    def _delete_seed(self) -> None:
        rows = self._seed_table.selectedItems()
        if not rows:
            return
        item = self._seed_table.item(self._seed_table.currentRow(), 0)
        if not item:
            return
        seed_id = item.data(Qt.ItemDataRole.UserRole)
        if seed_id:
            self._seeds.delete(seed_id)
            self._refresh_seed_table()

    def _selected_session_id(self) -> str | None:
        session_id = self._session_combo.currentData()
        if not session_id:
            return None
        return str(session_id)


class _FrameSpinBox(QWidget):
    """Frame number spinner (large range, integer)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QSpinBox
        self._spin = QSpinBox()
        self._spin.setRange(0, 99_999_999)
        self._spin.setSingleStep(1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._spin)

    def value(self) -> int:
        return self._spin.value()

    def setValue(self, v: int) -> None:
        self._spin.setValue(v)
