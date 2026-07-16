"""Data import tab for videos and DeepLabCut/SLEAP pose files."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, Signal

from abel.models.schemas import ImportManifest, ImportNameSettings, SourceMode
from abel.services import keypoint_mapping
from abel.services.import_service import ImportService
from abel.services.pose_processing_service import PoseProcessingService
from abel.utils.sleap_converter import is_sleap_pose_file
from abel.ui.animal_identity_dialog import AnimalIdentityDialog
from abel.ui.body_part_rename_dialog import BodyPartRenameDialog
from abel.ui.keypoint_mapping_dialog import KeypointMappingDialog
from abel.ui.pixel_scale_calibration_dialog import PixelScaleCalibrationDialog


# Session-table column indices (see _populate_table header order).
_COL_SUBJECT = 1
_COL_SESSION_TYPE = 2
_COL_PXMM = 7
_EDITABLE_COLS = {_COL_SUBJECT, _COL_SESSION_TYPE, _COL_PXMM}


class _SortableTableItem(QTableWidgetItem):
    """Table item that sorts numerically when both cells hold numbers.

    Plain ``QTableWidgetItem`` sorts lexically, so "10" would sort before "2"
    for the Score/px-mm columns. Fall back to case-insensitive text order for
    non-numeric cells (subject, session type, filenames, paths).
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:  # type: ignore[override]
        a = self.text().strip()
        b = other.text().strip() if isinstance(other, QTableWidgetItem) else ""
        try:
            return float(a) < float(b)
        except (ValueError, TypeError):
            return a.casefold() < b.casefold()


class DataImportTab(QWidget):
    """Imports video and pose files and builds linked sessions."""

    _copy_progress_signal = Signal(str)
    _copy_log_signal = Signal(str)
    _blocking_done_signal = Signal()  # emitted (from a worker thread) when a wait-dialog task finishes
    num_animals_changed = Signal(int)  # emitted when the user changes the project's animal count

    def __init__(self, import_service: ImportService, parent=None) -> None:
        super().__init__(parent)
        self._import_service = import_service
        self._logger = logging.getLogger("abel")
        self._project_root: Path | None = None
        self._video_paths: list[Path] = []
        self._pose_paths: list[Path] = []
        self._manifest = ImportManifest()
        self._is_populating_table = False
        self.status = QLabel("No project loaded.")
        self.session_table = QTableWidget(0, 10)
        self.session_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.session_table.setHorizontalHeaderLabels(
            ["Session ID", "Subject", "Session Type", "Video Asset", "Pose Asset", "Score", "Notes", "px/mm", "Video Path", "Pose Path"]
        )
        # Click a header to sort by that column (e.g. group by Session Type).
        self.session_table.setSortingEnabled(True)
        self.session_table.horizontalHeader().setSortIndicatorShown(True)
        self.session_table.itemChanged.connect(self._on_session_item_changed)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)

        self._subject_regex_input = QLineEdit(self._manifest.subject_name_settings.subject_regex)
        self._subject_group_spin = QSpinBox()
        self._subject_group_spin.setMinimum(0)
        self._subject_group_spin.setMaximum(12)
        self._subject_group_spin.setValue(self._manifest.subject_name_settings.subject_group_index)
        self._session_regex_input = QLineEdit(self._manifest.subject_name_settings.session_regex)
        self._session_group_spin = QSpinBox()
        self._session_group_spin.setMinimum(0)
        self._session_group_spin.setMaximum(12)
        self._session_group_spin.setValue(self._manifest.subject_name_settings.session_group_index)
        self._preview_filename = QLineEdit("DG01BehavioralCamera0.avi")
        self._preview_subject = QLabel("DG01")
        self._preview_session = QLabel("")

        # Project-level: number of animals tracked per session (editable here so
        # it can be corrected without recreating the project).
        self._num_animals_spin = QSpinBox()
        self._num_animals_spin.setRange(1, 20)
        self._num_animals_spin.setValue(1)
        self._num_animals_spin.setToolTip(
            "Number of animals tracked per session for this project.\n"
            "Saved to project.yaml; sets single_animal = (value <= 1).\n"
            "Multi-animal enables per-individual (and optional social) features."
        )
        self._set_num_animals_btn = QPushButton("Set")
        self._set_num_animals_btn.setToolTip("Save the number of animals to this project.")
        self._set_num_animals_btn.clicked.connect(self._apply_num_animals)

        import_video_btn = QPushButton("Import Videos")
        import_pose_btn = QPushButton("Import Pose (DLC / SLEAP)")
        auto_match_btn = QPushButton("Auto Match")
        save_manifest_btn = QPushButton("Save Import Manifest")
        remove_session_btn = QPushButton("Remove Selected Session(s)")
        calibrate_scale_btn = QPushButton("Calibrate px/mm")
        keypoint_map_btn = QPushButton("Keypoint Mapping")
        keypoint_map_btn.setToolTip(
            "Map keypoints from imported pose files onto the project's canonical\n"
            "keypoint names when DLC files name them differently."
        )
        rename_parts_btn = QPushButton("Rename Body Parts")
        rename_parts_btn.setToolTip(
            "Give body parts brand-new names of your choosing. The new names are\n"
            "used by all subsequent processing (feature extraction, context\n"
            "features, trained models)."
        )
        reapply_subject_btn = QPushButton("Apply Parsing Settings")
        test_pattern_btn = QPushButton("Test Pattern")

        import_video_btn.clicked.connect(self._import_videos)
        import_pose_btn.clicked.connect(self._import_pose)
        auto_match_btn.clicked.connect(self._build_manifest)
        save_manifest_btn.clicked.connect(self._save_manifest)
        remove_session_btn.clicked.connect(self._remove_selected_sessions)
        calibrate_scale_btn.clicked.connect(self._open_pixel_scale_calibrator)
        keypoint_map_btn.clicked.connect(self._open_keypoint_mapping)
        rename_parts_btn.clicked.connect(self._open_body_part_rename)
        identity_map_btn = QPushButton("Map Animal Identities")
        identity_map_btn.setToolTip(
            "For multi-animal pose files: assign each detected individual (Mouse1,\n"
            "Mouse2…) a real subject identity (e.g. green/black) used as its animal_id."
        )
        identity_map_btn.clicked.connect(self._open_identity_map)
        self._identity_map_btn = identity_map_btn
        reapply_subject_btn.clicked.connect(self._apply_subject_settings)
        test_pattern_btn.clicked.connect(self._update_subject_preview)

        copy_pxmm_all_btn = QPushButton("Apply px/mm to All Sessions")
        copy_pxmm_all_btn.setToolTip(
            "Copy a px/mm value to every linked session at once.\n"
            "Useful when all sessions share the same camera/arena geometry."
        )
        copy_pxmm_all_btn.clicked.connect(self._apply_pxmm_to_all_sessions)

        copy_pxmm_selected_btn = QPushButton("Apply px/mm to Selected")
        copy_pxmm_selected_btn.setToolTip(
            "Copy a px/mm value to table-selected sessions only."
        )
        copy_pxmm_selected_btn.clicked.connect(self._apply_pxmm_to_selected_sessions)

        button_row = QHBoxLayout()
        for btn in [
            import_video_btn,
            import_pose_btn,
            auto_match_btn,
            save_manifest_btn,
            remove_session_btn,
            calibrate_scale_btn,
            keypoint_map_btn,
            rename_parts_btn,
            identity_map_btn,
        ]:
            button_row.addWidget(btn)

        # Keypoint-consistency warning banner (hidden unless a mismatch exists).
        self._keypoint_warning = QLabel("")
        self._keypoint_warning.setWordWrap(True)
        self._keypoint_warning.setStyleSheet(
            "color: #FFB74D; background: #2E2A0A; border: 1px solid #6D5A00;"
            " border-radius: 4px; padding: 6px;"
        )
        self._keypoint_warning.hide()

        pxmm_row = QHBoxLayout()
        pxmm_row.addWidget(copy_pxmm_all_btn)
        pxmm_row.addWidget(copy_pxmm_selected_btn)
        pxmm_row.addStretch(1)

        copy_files_btn = QPushButton("Copy Files to Project")
        copy_files_btn.setToolTip(
            "Copy all referenced video and pose files from their original\n"
            "locations into the project's raw/ folder.  This makes the\n"
            "project self-contained and avoids reading from external drives."
        )
        copy_files_btn.clicked.connect(self._copy_files_to_project)

        self._copy_status_label = QLabel("")
        self._copy_status_label.setWordWrap(True)

        # Wire thread-safe signals for copy progress updates.
        self._copy_progress_signal.connect(self._copy_status_label.setText)
        self._copy_log_signal.connect(self._append_log)

        relocate_video_btn = QPushButton("Set Video Path")
        relocate_video_btn.setToolTip("Point videos to a new folder (matches by filename).")
        relocate_video_btn.clicked.connect(lambda: self._relocate_sources("video"))

        relocate_pose_btn = QPushButton("Set DLC Path")
        relocate_pose_btn.setToolTip(
            "Point pose files to a new folder. Matches by filename, then falls back to "
            "the DLC stem so a different DLC run of the same recordings still matches."
        )
        relocate_pose_btn.clicked.connect(lambda: self._relocate_sources("pose"))

        copy_row = QHBoxLayout()
        copy_row.addWidget(copy_files_btn)
        copy_row.addWidget(relocate_video_btn)
        copy_row.addWidget(relocate_pose_btn)
        copy_row.addWidget(self._copy_status_label, 1)

        import_page = QWidget()
        import_layout = QVBoxLayout(import_page)
        import_layout.addWidget(QLabel("Import videos and pose files, then auto-link sessions."))
        import_layout.addWidget(self.status)
        import_layout.addLayout(button_row)
        import_layout.addLayout(pxmm_row)
        import_layout.addLayout(copy_row)
        import_layout.addWidget(self._keypoint_warning)
        import_layout.addWidget(self.session_table)

        settings_form = QFormLayout()
        settings_form.addRow("Subject regex:", self._subject_regex_input)
        settings_form.addRow("Subject capture group:", self._subject_group_spin)
        settings_form.addRow("Session regex:", self._session_regex_input)
        settings_form.addRow("Session capture group:", self._session_group_spin)
        settings_form.addRow("Filename preview:", self._preview_filename)
        settings_form.addRow("Extracted subject:", self._preview_subject)
        settings_form.addRow("Extracted session:", self._preview_session)

        settings_btn_row = QHBoxLayout()
        settings_btn_row.addWidget(test_pattern_btn)
        settings_btn_row.addWidget(reapply_subject_btn)
        settings_btn_row.addStretch(1)

        settings_help = QLabel(
            "Subject regex pulls the subject ID from the filename prefix (e.g. CBMRE01 from "
            "CBMRE01_TestingDay2.mp4 or CBMRE01_AcclimationDLC_Resnet50_....csv). "
            "Session regex extracts the session type from the part after the subject (e.g. TestingDay2)."
        )
        settings_help.setWordWrap(True)

        # Project settings (num animals) — separate from filename parsing.
        num_animals_row = QHBoxLayout()
        num_animals_row.addWidget(QLabel("Number of animals:"))
        num_animals_row.addWidget(self._num_animals_spin)
        num_animals_row.addWidget(self._set_num_animals_btn)
        num_animals_row.addStretch(1)
        project_help = QLabel(
            "Animals tracked per session. Use 2 for a two-mouse dominance session; "
            "1 for single-animal. Saved to the project immediately."
        )
        project_help.setWordWrap(True)

        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.addWidget(settings_help)
        settings_layout.addLayout(settings_form)
        settings_layout.addLayout(settings_btn_row)
        settings_layout.addSpacing(12)
        settings_layout.addWidget(project_help)
        settings_layout.addLayout(num_animals_row)
        settings_layout.addStretch(1)

        self._import_tabs = QTabWidget()
        self._import_tabs.addTab(import_page, "Imports")
        self._import_tabs.addTab(settings_page, "Parsing Settings")

        layout = QVBoxLayout(self)
        layout.addWidget(self._import_tabs)
        layout.addWidget(QLabel("Import logs"))
        layout.addWidget(self.log_panel)

        self._update_subject_preview()

    def set_project_root(self, project_root: Path | None) -> None:
        self._reset_for_project_switch()
        self._project_root = project_root
        self.status.setText(
            f"Project: {project_root}" if project_root else "No project loaded."
        )
        if project_root:
            self._load_manifest()
            self._load_num_animals_from_project()

    def _load_num_animals_from_project(self) -> None:
        """Populate the num-animals spinbox from the project's project.yaml."""
        val = 1
        try:
            import yaml
            cfg_path = Path(self._project_root) / "project.yaml"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                val = int(cfg.get("num_animals", 1) or 1)
        except Exception:
            val = 1
        self._num_animals_spin.blockSignals(True)
        self._num_animals_spin.setValue(max(1, min(20, val)))
        self._num_animals_spin.blockSignals(False)

    def _apply_num_animals(self) -> None:
        """Persist the number of animals to the project (via main_window)."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        n = int(self._num_animals_spin.value())
        self.num_animals_changed.emit(n)
        self._append_log(f"Number of animals set to {n} (single_animal={n <= 1}).")

    def _reset_for_project_switch(self) -> None:
        """Clear transient import/upload state before loading another project."""
        self._video_paths = []
        self._pose_paths = []
        self._manifest = ImportManifest()
        self._set_subject_settings_ui(self._manifest.subject_name_settings)
        self.session_table.setRowCount(0)
        self.log_panel.clear()
        self._keypoint_warning.hide()

    def _load_manifest(self) -> None:
        """Read a previously saved manifest and restore tab state."""
        if not self._project_root:
            return
        manifest = self._import_service.load_manifest(self._project_root)
        if manifest is None:
            self._manifest = ImportManifest()
            self._set_subject_settings_ui(self._manifest.subject_name_settings)
            self.session_table.setRowCount(0)
            self._video_paths = []
            self._pose_paths = []
            self._append_log("No saved import manifest found for this project.")
            return
        self._manifest = manifest
        self._set_subject_settings_ui(manifest.subject_name_settings)
        self._video_paths = [Path(v.source_path) for v in manifest.videos]
        self._pose_paths = [Path(p.source_path) for p in manifest.poses]
        self._populate_table(manifest)
        self._append_log(
            f"Loaded manifest: {len(manifest.videos)} videos, "
            f"{len(manifest.poses)} pose files, "
            f"{len(manifest.linked_sessions)} linked sessions."
        )
        self._check_keypoint_consistency()

    def _import_videos(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv)",
        )
        existing_resolved = {str(p.resolve()) for p in self._video_paths}
        added = [Path(p) for p in selected if str(Path(p).resolve()) not in existing_resolved]
        self._video_paths.extend(added)
        self._append_log(
            f"Added {len(added)} new video file(s); {len(self._video_paths)} total."
        )

    def _import_pose(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select pose files (DeepLabCut or SLEAP)",
            "",
            "Pose files (*.csv *.h5 *.hdf5 *.slp);;"
            "DeepLabCut (*.csv *.h5 *.hdf5);;SLEAP (*.slp)",
        )
        if not selected:
            return
        paths = [Path(p) for p in selected]
        sleap_paths = [p for p in paths if is_sleap_pose_file(p)]
        dlc_paths = [p for p in paths if not is_sleap_pose_file(p)]

        # SLEAP predictions aren't a format ABEL reads directly; offer to convert
        # them to a DeepLabCut .h5 that flows through the normal import path.
        if sleap_paths:
            resp = QMessageBox.question(
                self,
                "Convert SLEAP files?",
                f"{len(sleap_paths)} SLEAP prediction file(s) (.slp) were selected.\n\n"
                "ABEL reads DeepLabCut-format pose files. Convert these to a "
                "compatible DeepLabCut .h5 now?\n\n"
                "Your originals are left untouched — a matching '<name>.sleap.h5' "
                "is written next to each and imported in its place.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if resp == QMessageBox.Yes:
                self._append_log(
                    f"Converting {len(sleap_paths)} SLEAP file(s) to DeepLabCut format..."
                )

                def _cb(i: int, total: int, name: str) -> None:
                    if name != "done":
                        self._append_log(f"  [{i + 1}/{total}] {name}")
                    QApplication.processEvents()

                converted, failures = self._import_service.convert_sleap_poses(
                    sleap_paths, progress_cb=_cb
                )
                dlc_paths.extend(converted)
                self._append_log(
                    f"Converted {len(converted)} of {len(sleap_paths)} SLEAP file(s)."
                )
                if failures:
                    detail = "\n".join(f"- {p.name}: {err}" for p, err in failures[:8])
                    if len(failures) > 8:
                        detail += f"\n… and {len(failures) - 8} more."
                    QMessageBox.warning(
                        self,
                        "Some SLEAP files could not be converted",
                        f"{len(failures)} file(s) failed:\n\n{detail}",
                    )
            else:
                self._append_log(
                    f"Skipped {len(sleap_paths)} SLEAP file(s) (not converted)."
                )

        existing_resolved = {str(p.resolve()) for p in self._pose_paths}
        added = [p for p in dlc_paths if str(p.resolve()) not in existing_resolved]
        self._pose_paths.extend(added)
        self._append_log(
            f"Added {len(added)} new pose file(s); {len(self._pose_paths)} total."
        )

    def _build_manifest(self) -> None:
        settings = self._subject_settings_from_ui()

        # Auto Match probes every video/pose file's metadata, which can be slow
        # when the source files live on a network drive. Run that I/O off the GUI
        # thread behind a modal wait dialog so the window stays responsive. The
        # keypoint probe (also file I/O) piggybacks on the same worker and its
        # result is handed back for the on-thread GUI update below.
        def _work() -> dict:
            if self._manifest.linked_sessions:
                # Additive path: preserve existing sessions and only add new ones.
                self._import_service.merge_new_files(
                    self._manifest,
                    self._video_paths,
                    self._pose_paths,
                    settings=settings,
                )
            else:
                # Fresh project: build from scratch.
                self._manifest = self._import_service.build_manifest(
                    self._video_paths,
                    self._pose_paths,
                    subject_name_settings=settings,
                )
            return self._pose_keypoint_sets()

        file_sets, error = self._run_blocking(
            "Auto Matching",
            "Scanning imported videos and pose files and linking sessions…\n"
            "Please wait — reading from the source folders (slow over a network drive).",
            _work,
        )
        if error is not None:
            self._logger.error("Auto match failed", exc_info=error)
            QMessageBox.critical(
                self,
                "Auto Match Failed",
                f"Could not finish matching the imported files:\n\n{error}",
            )
            return

        self._populate_table(self._manifest)
        linked = self._manifest.linked_sessions
        self._append_log(
            f"Manifest updated: {len(self._manifest.videos)} videos, "
            f"{len(self._manifest.poses)} pose files, {len(linked)} linked sessions."
        )
        missing_scale = sum(
            1 for session in linked
            if self._import_service.pixels_per_mm_for_session(self._manifest, session.session_id) is None
        )
        if missing_scale > 0:
            self._append_log(
                f"Scale helper: {missing_scale} session(s) have no px/mm value. "
                "Enter px/mm in the import table to enable physical-unit metric scaling."
            )
        self._check_keypoint_consistency(file_sets)
        self._save_manifest(silent=True)

    def _populate_table(self, manifest: ImportManifest) -> None:
        """Fill the session table with human-readable filenames."""
        self._is_populating_table = True
        # Suspend sorting while filling: with it live, each setItem could reorder
        # rows mid-populate and scramble the row→value mapping.
        sorting_was_enabled = self.session_table.isSortingEnabled()
        self.session_table.setSortingEnabled(False)
        video_by_id = {v.asset_id: v for v in manifest.videos}
        pose_by_id = {p.asset_id: p for p in manifest.poses}
        linked = manifest.linked_sessions
        self.session_table.setRowCount(len(linked))
        for row, session in enumerate(linked):
            video = video_by_id.get(session.video_asset_id)
            pose = pose_by_id.get(session.pose_asset_id)
            video_name = Path(video.source_path).name if video else session.video_asset_id
            pose_name = Path(pose.source_path).name if pose else session.pose_asset_id
            subject_name = session.subject_id or (video.subject_id if video else "") or ""
            # For multi-animal sessions, surface the assigned identities (and any
            # swap corrections) so the result of the identity tool is visible.
            inds = list(getattr(session, "individuals", []) or [])
            if inds:
                id_map = getattr(session, "individual_subject_map", {}) or {}
                names = [id_map.get(i, i) for i in inds]
                subject_name = " / ".join(names)
                n_corr = len(getattr(session, "identity_corrections", []) or [])
                if n_corr:
                    subject_name += f"  (⚠ {n_corr} swap fix)"

            # Active path: prefer local copy, fall back to source.
            video_active = (video.local_path or video.source_path) if video else ""
            pose_active = (pose.local_path or pose.source_path) if pose else ""

            session_type = self._import_service.effective_session_type(manifest, session)

            values = [
                session.session_id,
                subject_name,
                session_type,
                video_name,
                pose_name,
                f"{session.pairing_score:.2f}",
                session.pairing_notes,
                "" if session.pixels_per_mm is None else f"{float(session.pixels_per_mm):.6g}",
                str(Path(video_active).parent) if video_active else "",
                str(Path(pose_active).parent) if pose_active else "",
            ]
            for col, value in enumerate(values):
                item = _SortableTableItem(value)
                item.setToolTip(value)  # full path visible on hover
                if col not in _EDITABLE_COLS:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.session_table.setItem(row, col, item)
        self.session_table.setSortingEnabled(sorting_was_enabled)
        self._is_populating_table = False

    def _on_session_item_changed(self, item: QTableWidgetItem) -> None:
        if self._is_populating_table:
            return
        if item.column() not in _EDITABLE_COLS:
            return
        row = item.row()
        # Resolve by Session ID (column 0), not row index: sorting reorders rows
        # so the visual row no longer lines up with linked_sessions positionally.
        id_item = self.session_table.item(row, 0)
        if id_item is None:
            return
        session = next(
            (s for s in self._manifest.linked_sessions if s.session_id == id_item.text()),
            None,
        )
        if session is None:
            return
        if item.column() == _COL_SUBJECT:
            self._import_service.update_session_subject(self._manifest, session.session_id, item.text())
            self._save_manifest(silent=True)
            self._append_log(f"Updated subject for {session.session_id}: {item.text().strip() or 'unset'}")
            return

        if item.column() == _COL_SESSION_TYPE:
            self._import_service.update_session_type(self._manifest, session.session_id, item.text())
            self._save_manifest(silent=True)
            self._append_log(
                f"Updated session type for {session.session_id}: "
                f"{item.text().strip() or 'unset (regex-derived)'}"
            )
            return

        raw = item.text().strip()
        if not raw:
            self._import_service.update_session_pixels_per_mm(self._manifest, session.session_id, None)
            self._save_manifest(silent=True)
            self._append_log(
                f"Cleared px/mm for {session.session_id}. "
                "Distance features will stay in pixel units until set."
            )
            return

        try:
            val = float(raw)
        except Exception:
            val = -1.0
        if val <= 0:
            QMessageBox.warning(
                self,
                "Invalid px/mm",
                "Pixels per millimeter must be a positive number, or left blank.",
            )
            self._is_populating_table = True
            item.setText("" if session.pixels_per_mm is None else f"{float(session.pixels_per_mm):.6g}")
            self._is_populating_table = False
            return

        self._import_service.update_session_pixels_per_mm(self._manifest, session.session_id, val)
        self._save_manifest(silent=True)
        self._append_log(
            f"Updated px/mm for {session.session_id}: {val:.6g}. "
            "Physical-unit feature scaling is now enabled for this session."
        )

    def _subject_settings_from_ui(self) -> ImportNameSettings:
        return ImportNameSettings(
            subject_regex=self._subject_regex_input.text().strip() or ImportNameSettings().subject_regex,
            subject_group_index=int(self._subject_group_spin.value()),
            session_regex=self._session_regex_input.text().strip(),
            session_group_index=int(self._session_group_spin.value()),
        )

    def _set_subject_settings_ui(self, settings: ImportNameSettings) -> None:
        self._subject_regex_input.setText(settings.subject_regex)
        self._subject_group_spin.setValue(settings.subject_group_index)
        self._session_regex_input.setText(settings.session_regex)
        self._session_group_spin.setValue(settings.session_group_index)
        self._update_subject_preview()

    def _update_subject_preview(self) -> None:
        settings = self._subject_settings_from_ui()
        subject = self._import_service.extract_subject_name(Path(self._preview_filename.text()), settings)
        session = self._import_service.extract_session_type(Path(self._preview_filename.text()), settings)
        self._preview_subject.setText(subject or "(no match)")
        self._preview_session.setText(session or "(no match)")

    def _apply_subject_settings(self) -> None:
        settings = self._subject_settings_from_ui()
        self._update_subject_preview()
        if not self._manifest.videos and not self._manifest.poses:
            QMessageBox.information(self, "No Imports", "Import videos and pose files first.")
            return
        self._import_service.apply_subject_name_settings(self._manifest, settings)
        self._populate_table(self._manifest)
        self._save_manifest(silent=True)
        self._append_log("Subject parsing settings applied to current manifest.")

    def _remove_selected_sessions(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        rows = sorted({index.row() for index in self.session_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "No Selection", "Select one or more sessions to remove.")
            return

        session_ids: list[str] = []
        for row in rows:
            item = self.session_table.item(row, 0)
            if item is not None:
                session_ids.append(item.text())
        if not session_ids:
            return

        answer = QMessageBox.question(
            self,
            "Remove Sessions",
            "Remove selected sessions and clear associated data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Removal prunes many large per-session caches (parquet stores, clip
        # trees), which can take a while. Run it off the GUI thread behind a
        # modal busy dialog so the window keeps pumping events instead of going
        # "(Not Responding)".
        summary = self._run_session_removal(session_ids)
        if summary is None:
            return  # an error was reported to the user

        self._video_paths = [Path(v.source_path) for v in self._manifest.videos]
        self._pose_paths = [Path(p.source_path) for p in self._manifest.poses]
        self._populate_table(self._manifest)
        self._save_manifest(silent=True)
        message = (
            f"Removed {summary['sessions']} session(s), deleted {summary['files']} file(s), "
            f"cleaned {summary['rows']} row(s) of associated data."
        )
        if summary.get("remapped"):
            message += (
                f" Re-pointed {summary['remapped']} label/decision/seed(s) at the "
                "sessions that still hold the same recordings."
            )
        self._append_log(message)

    def _run_blocking(self, title: str, message: str, work):
        """Run *work* on a worker thread behind a modal indeterminate wait dialog.

        Returns ``(result, error)`` where exactly one is meaningful: ``error`` is
        the caught exception (or ``None`` on success). The GUI thread blocks in a
        local event loop until the worker signals completion, so the dialog keeps
        painting and the window stays responsive instead of "(Not Responding)".

        ``work`` must be self-contained (no Qt widget access) — only its return
        value crosses back; do all GUI updates in the caller after this returns.
        """
        import threading
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QProgressDialog

        progress = QProgressDialog(
            message,
            None,  # no cancel button — these tasks aren't safely interruptible
            0,
            0,  # min == max == 0 → indeterminate "busy" bar
            self,
        )
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        box: dict = {}
        loop = QEventLoop()

        def _work() -> None:
            try:
                box["result"] = work()
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                box["error"] = exc
            finally:
                # Queued across threads → loop.quit runs on the GUI thread.
                self._blocking_done_signal.emit()

        self._blocking_done_signal.connect(loop.quit)
        try:
            threading.Thread(target=_work, daemon=True).start()
            loop.exec()
        finally:
            self._blocking_done_signal.disconnect(loop.quit)
            progress.close()

        return box.get("result"), box.get("error")

    def _run_session_removal(self, session_ids: list[str]) -> dict | None:
        """Delete sessions on a worker thread, showing a modal wait dialog.

        Returns the removal summary dict, or ``None`` if it failed (the error is
        shown to the user).
        """
        summary, error = self._run_blocking(
            "Removing Sessions",
            "Removing sessions and clearing associated data…\nPlease wait.",
            lambda: self._import_service.remove_sessions(
                self._project_root, self._manifest, session_ids
            ),
        )
        if error is not None:
            self._logger.error("Session removal failed", exc_info=error)
            QMessageBox.critical(
                self,
                "Removal Failed",
                f"Could not finish removing the selected session(s):\n\n{error}",
            )
            return None
        return summary

    # ── Keypoint consistency ─────────────────────────────────────────

    def _saved_keypoint_aliases(self) -> dict[str, str]:
        """Project-level {data_name: canonical_name} rename map."""
        if not self._project_root:
            return {}
        from abel.storage.file_store import read_json
        data = read_json(self._project_root / "config" / "keypoint_aliases.json", {})
        return {str(k): str(v) for k, v in data.items() if str(k) and str(v)}

    def _canonical_keypoints(self, file_sets: dict[str, list[str]]) -> list[str]:
        """Determine the project's canonical keypoint scheme.

        Prefers the keypoints the project's pose features were already built
        with; otherwise falls back to the most common set among imported files.
        Saved renames are then applied so a deliberately renamed body part is
        reported under its new name (and so isn't flagged as a mismatch against
        its original name in :meth:`_check_keypoint_consistency`).
        """
        raw_canonical: list[str] = []
        if self._project_root:
            fp = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
            if fp.exists():
                try:
                    import pyarrow.parquet as pq  # noqa: PLC0415
                    cols = list(pq.read_schema(fp).names)
                    raw_canonical = sorted(
                        c[: -len("_velocity_x")] for c in cols if c.endswith("_velocity_x")
                    )
                except Exception:
                    raw_canonical = []
        if not raw_canonical:
            # Fallback: most common keypoint set among imported files.
            from collections import Counter
            counter: Counter[frozenset[str]] = Counter()
            for kps in file_sets.values():
                if kps:
                    counter[frozenset(kps)] += 1
            if not counter:
                return []
            raw_canonical = sorted(counter.most_common(1)[0][0])

        aliases = self._saved_keypoint_aliases()
        return sorted({aliases.get(k, k) for k in raw_canonical})

    def _pose_keypoint_sets(self) -> dict[str, list[str]]:
        """Probe each linked pose file for its (normalized) keypoint names."""
        out: dict[str, list[str]] = {}
        pose_by_id = {p.asset_id: p for p in self._manifest.poses}
        for session in self._manifest.linked_sessions:
            pose = pose_by_id.get(session.pose_asset_id)
            if pose is None:
                continue
            path = Path(pose.local_path or pose.source_path)
            if not path.exists():
                continue
            try:
                meta = PoseProcessingService.probe_metadata(path)
                out[path.name] = list(meta.get("body_parts", []))
            except Exception:
                continue
        return out

    def _check_keypoint_consistency(self, file_sets: dict[str, list[str]] | None = None) -> None:
        """Flag pose files whose keypoints don't match the project scheme.

        *file_sets* may be supplied by a caller that already probed the pose files
        (e.g. off the GUI thread) to avoid re-reading them here; otherwise they're
        probed on demand.
        """
        if not self._manifest.linked_sessions:
            self._keypoint_warning.hide()
            return
        if file_sets is None:
            file_sets = self._pose_keypoint_sets()
        if not file_sets:
            self._keypoint_warning.hide()
            return
        canonical = set(self._canonical_keypoints(file_sets))
        if not canonical:
            self._keypoint_warning.hide()
            return
        aliases = self._saved_keypoint_aliases()

        mismatched: list[str] = []
        for fname, kps in file_sets.items():
            effective = {aliases.get(k, k) for k in kps}
            if canonical - effective:  # canonical keypoint(s) not provided
                mismatched.append(fname)

        if not mismatched:
            self._keypoint_warning.hide()
            return

        self._keypoint_warning.setText(
            f"⚠ {len(mismatched)} of {len(file_sets)} pose file(s) use keypoint names "
            "that don't match the project's scheme. Their features won't align with "
            "the rest of the project (or a trained model). Click "
            "“Keypoint Mapping” to remap them."
        )
        self._keypoint_warning.show()
        self._append_log(
            "Keypoint check: "
            f"{len(mismatched)} pose file(s) need remapping ({', '.join(mismatched[:3])}"
            f"{'…' if len(mismatched) > 3 else ''})."
        )

    def _open_keypoint_mapping(self) -> None:
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import and link sessions first.")
            return
        file_sets = self._pose_keypoint_sets()
        if not file_sets:
            QMessageBox.information(
                self, "Keypoint Mapping", "No readable pose files to inspect.")
            return
        canonical = self._canonical_keypoints(file_sets)
        if not canonical:
            QMessageBox.information(
                self, "Keypoint Mapping", "Could not determine a keypoint scheme.")
            return
        found = sorted({kp for kps in file_sets.values() for kp in kps})

        # Seed the dialog from any saved rename map (inverted to expected->found).
        saved = self._saved_keypoint_aliases()  # {found: canonical}
        initial = {canon: data for data, canon in saved.items() if canon in canonical}

        dlg = KeypointMappingDialog(
            expected_keypoints=canonical,
            found_keypoints=found,
            initial_map=initial,
            expected_label="Project Keypoint",
            found_label="Found in Pose File(s)",
            explainer=(
                "Map each keypoint the project expects to the matching keypoint "
                "in your imported pose files. Keypoints with the same name need no "
                "change. Suggestions are auto-filled — review and correct them."
            ),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        rename = keypoint_mapping.to_rename_map(dlg.result_map)  # {found: canonical}
        if self._project_root:
            from abel.storage.file_store import write_json
            cfg = self._project_root / "config"
            cfg.mkdir(parents=True, exist_ok=True)
            write_json(cfg / "keypoint_aliases.json", rename)
            self._append_log(
                f"Saved keypoint mapping ({len(rename)} rename rule(s)). "
                "It will be applied during feature extraction."
            )
        self._check_keypoint_consistency()

    def _open_body_part_rename(self) -> None:
        """Let the user give body parts new names used by all downstream steps."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import and link sessions first.")
            return
        file_sets = self._pose_keypoint_sets()
        found = sorted({kp for kps in file_sets.values() for kp in kps})
        if not found:
            QMessageBox.information(
                self, "Rename Body Parts", "No readable pose files to inspect.")
            return

        existing = self._saved_keypoint_aliases()  # {original: new}
        dlg = BodyPartRenameDialog(found, initial_map=existing, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        renames = dlg.result_map  # {original: new}, changed parts only

        # The rename map shares config/keypoint_aliases.json with Keypoint
        # Mapping (both are {source_name: target_name}).  Preserve any alias
        # entries for parts not shown here, and replace the rest with the user's
        # choices — dropping parts they reset back to their original name.
        merged = {k: v for k, v in existing.items() if k not in set(found)}
        merged.update(renames)

        if merged == existing:
            self._append_log("Body-part names unchanged — nothing to update.")
            return

        from abel.storage.file_store import write_json
        cfg = self._project_root / "config"
        cfg.mkdir(parents=True, exist_ok=True)
        write_json(cfg / "keypoint_aliases.json", merged)
        self._append_log(
            f"Renamed {len(renames)} body part(s). The new names will be used "
            "by all subsequent processing (feature extraction, models)."
        )

        # The rename changes every keypoint-derived column, so any cached pose/
        # context features are stale.  Mark them for rebuild rather than reusing
        # them, otherwise the next feature extraction would silently keep the
        # old names.
        from abel.services.feature_prep_service import FeaturePrepService
        FeaturePrepService.invalidate_caches(self._project_root)

        fp = self._project_root / "derived" / "pose_features" / "frame_pose.parquet"
        if fp.exists():
            self._append_log(
                "Pose features already exist under the old names — re-run feature "
                "extraction so the renames take effect everywhere."
            )
        self._check_keypoint_consistency()

    def _multi_animal_sessions(self) -> list:
        return [s for s in self._manifest.linked_sessions if getattr(s, "individuals", None)]

    def _selected_multi_session(self):
        """Resolve which multi-animal session to operate on.

        Prefers the table-selected row; falls back to the sole multi-animal
        session.  Returns the LinkedSession or None (with a user message).
        """
        multi = self._multi_animal_sessions()
        if not multi:
            QMessageBox.information(
                self, "No Multi-Animal Sessions",
                "No imported pose file contains multiple tracked individuals "
                "(no DLC 'individuals' header level was detected).\n\nIf you imported "
                "multi-animal files before updating, click Auto Match to re-probe them.",
            )
            return None
        rows = sorted({i.row() for i in self.session_table.selectionModel().selectedRows()})
        if rows:
            sid = self.session_table.item(rows[0], 0).text()
            sess = next((s for s in multi if s.session_id == sid), None)
            if sess is not None:
                return sess
            QMessageBox.information(
                self, "Select a Multi-Animal Session",
                "The selected session is single-animal. Select a multi-animal session "
                "row, or deselect to use the only multi-animal session.",
            )
            return None
        if len(multi) == 1:
            return multi[0]
        QMessageBox.information(
            self, "Select a Session",
            "Select the multi-animal session row you want to assign identities for.",
        )
        return None

    def _open_identity_map(self) -> None:
        """Visual identity assignment + swap correction for one session."""
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import and link sessions first.")
            return
        session = self._selected_multi_session()
        if session is None:
            return

        pose_path = self._import_service.pose_path_for_session(self._manifest, session.session_id)
        if not pose_path:
            QMessageBox.warning(self, "No Pose File", "Could not resolve this session's pose file.")
            return

        # Load cleaned multi-animal pose (raw identities — corrections are applied
        # live inside the dialog for preview, then persisted for extraction).
        try:
            multi = PoseProcessingService().load_and_clean_multi(pose_path)
        except Exception as exc:
            QMessageBox.warning(self, "Pose Load Failed", f"Could not read pose file:\n{exc}")
            return
        swap_info = PoseProcessingService.detect_identity_swaps(multi)

        video_path = self._import_service.video_path_for_session(self._manifest, session.session_id)
        provider, cap = self._make_frame_provider(video_path)
        default_frame = min(max(0, multi.n_frames // 2), max(0, multi.n_frames - 1))
        try:
            dlg = AnimalIdentityDialog(
                session_label=session.subject_id or session.session_id,
                multi=multi,
                frame_provider=provider,
                n_frames=multi.n_frames,
                default_frame=default_frame,
                swap_frames=swap_info.get("frames", []),
                current_map=dict(getattr(session, "individual_subject_map", {}) or {}),
                current_corrections=list(getattr(session, "identity_corrections", []) or []),
                parent=self,
            )
            accepted = dlg.exec() == QDialog.DialogCode.Accepted
        finally:
            if cap is not None:
                cap.release()
        if not accepted:
            return

        self._import_service.update_session_individual_map(
            self._manifest, session.session_id, dlg.result_map
        )
        self._import_service.update_session_identity_corrections(
            self._manifest, session.session_id, dlg.result_corrections
        )
        # Identity/track changes invalidate any cached features for this session.
        if self._project_root:
            from abel.services.feature_prep_service import FeaturePrepService
            FeaturePrepService.invalidate_caches(self._project_root)
        ident = ", ".join(f"{k}→{v}" for k, v in dlg.result_map.items())
        self._append_log(
            f"Session {session.session_id}: identities [{ident}]; "
            f"{len(dlg.result_corrections)} swap correction(s). "
            "Re-run feature extraction to apply."
        )
        self._save_manifest(silent=True)
        self._populate_table(self._manifest)

    @staticmethod
    def _make_frame_provider(video_path):
        """Return ``(provider, cap)`` reading BGR frames by index from a video.

        ``provider`` is a callable ``int -> ndarray|None``; ``cap`` is the open
        OpenCV capture (or None) the caller must release.
        """
        if not video_path:
            return (lambda _i: None), None
        try:
            import cv2  # noqa: PLC0415
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return (lambda _i: None), None

            def _provider(idx: int):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    return frame if ok else None
                except Exception:
                    return None

            return _provider, cap
        except Exception:
            return (lambda _i: None), None

    def _open_pixel_scale_calibrator(self) -> None:
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import and link sessions first.")
            return

        default_session_id = None
        rows = sorted({index.row() for index in self.session_table.selectionModel().selectedRows()})
        if rows:
            # Read the Session ID from column 0 rather than indexing positionally,
            # since the table may be sorted into a different order.
            id_item = self.session_table.item(rows[0], 0)
            if id_item is not None:
                default_session_id = id_item.text()

        dlg = PixelScaleCalibrationDialog(
            import_service=self._import_service,
            manifest=self._manifest,
            default_session_id=default_session_id,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        session_id = str(dlg.result_session_id or "").strip()
        pixels_per_mm = dlg.result_pixels_per_mm
        if not session_id or pixels_per_mm is None or pixels_per_mm <= 0:
            QMessageBox.warning(self, "Calibration", "Calibration did not produce a valid px/mm value.")
            return

        self._import_service.update_session_pixels_per_mm(self._manifest, session_id, pixels_per_mm)
        self._populate_table(self._manifest)
        self._save_manifest(silent=True)
        self._append_log(
            f"Calibrated px/mm for {session_id}: {pixels_per_mm:.6g} (from two-point measurement)."
        )

    def _apply_pxmm_to_all_sessions(self) -> None:
        """Prompt for a px/mm value and apply it to every linked session."""
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import sessions first.")
            return

        from PySide6.QtWidgets import QInputDialog
        current_values = [
            s.pixels_per_mm for s in self._manifest.linked_sessions if s.pixels_per_mm is not None
        ]
        default = current_values[0] if current_values else 1.0
        val, ok = QInputDialog.getDouble(
            self,
            "Apply px/mm to All Sessions",
            "Pixels per mm value to apply to all sessions:",
            default,
            0.0001,
            1_000_000.0,
            6,
        )
        if not ok or val <= 0:
            return

        count = 0
        for session in self._manifest.linked_sessions:
            self._import_service.update_session_pixels_per_mm(self._manifest, session.session_id, val)
            count += 1

        self._populate_table(self._manifest)
        self._save_manifest(silent=True)
        self._append_log(f"Applied px/mm={val:.6g} to all {count} session(s).")

    def _apply_pxmm_to_selected_sessions(self) -> None:
        """Prompt for a px/mm value and apply it to table-selected sessions."""
        if not self._manifest.linked_sessions:
            QMessageBox.information(self, "No Sessions", "Import sessions first.")
            return

        rows = sorted({index.row() for index in self.session_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(
                self,
                "No Selection",
                "Select one or more sessions in the table first, then click this button.",
            )
            return

        session_ids: list[str] = []
        for row in rows:
            item = self.session_table.item(row, 0)
            if item is not None:
                session_ids.append(item.text())
        if not session_ids:
            return

        from PySide6.QtWidgets import QInputDialog
        current_values = [
            s.pixels_per_mm
            for s in self._manifest.linked_sessions
            if s.session_id in set(session_ids) and s.pixels_per_mm is not None
        ]
        default = current_values[0] if current_values else 1.0
        val, ok = QInputDialog.getDouble(
            self,
            "Apply px/mm to Selected Sessions",
            f"Pixels per mm value to apply to {len(session_ids)} selected session(s):",
            default,
            0.0001,
            1_000_000.0,
            6,
        )
        if not ok or val <= 0:
            return

        count = 0
        for sid in session_ids:
            self._import_service.update_session_pixels_per_mm(self._manifest, sid, val)
            count += 1

        self._populate_table(self._manifest)
        self._save_manifest(silent=True)
        self._append_log(f"Applied px/mm={val:.6g} to {count} selected session(s).")

    def _save_manifest(self, silent: bool = False) -> None:
        if not self._project_root:
            if not silent:
                self._append_log("Cannot save manifest: no project loaded.")
            return
        self._import_service.save_manifest(self._project_root, self._manifest)
        self._append_log("Import manifest saved automatically.")
        self._logger.info("Import manifest saved for project %s", self._project_root)

        # Auto-copy when the project is configured for copy mode.
        if self._project_source_mode_is_copy():
            self._auto_copy_if_needed()

    # ------------------------------------------------------------------
    # Copy-to-project helpers
    # ------------------------------------------------------------------

    def _project_source_mode_is_copy(self) -> bool:
        """Return True if the project.yaml specifies copy mode for videos or poses."""
        if not self._project_root:
            return False
        try:
            import yaml
            cfg_path = Path(self._project_root) / "project.yaml"
            if not cfg_path.exists():
                return False
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            return cfg.get("video_source_mode") == "copy" or cfg.get("pose_source_mode") == "copy"
        except Exception:
            return False

    def _auto_copy_if_needed(self) -> None:
        """If any asset still lacks a local_path, trigger a copy."""
        if not self._manifest:
            return
        needs_copy = any(v.local_path is None for v in self._manifest.videos) or \
                     any(p.local_path is None for p in self._manifest.poses)
        if needs_copy:
            self._run_copy(auto=True)

    def _copy_files_to_project(self) -> None:
        """Manual button handler — copy referenced files into raw/."""
        if not self._project_root:
            QMessageBox.warning(self, "No Project", "Load a project first.")
            return
        if not self._manifest:
            QMessageBox.warning(self, "No Manifest", "Import files first.")
            return
        self._run_copy(auto=False)

    def _run_copy(self, *, auto: bool) -> None:
        """Copy external assets into the project folder."""
        import threading

        project_root = self._project_root
        manifest = self._manifest

        # Use signals so all Qt widget updates happen on the GUI thread.
        progress_sig = self._copy_progress_signal
        log_sig = self._copy_log_signal

        def _progress(done: int, total: int, msg: str) -> None:
            progress_sig.emit(f"[{done}/{total}] {msg}")

        def _do_copy():
            try:
                result = self._import_service.copy_assets_to_project(
                    project_root,
                    manifest,
                    progress_cb=_progress,
                )
                vc = result.get("videos_copied", 0)
                pc = result.get("poses_copied", 0)
                summary = f"Copy complete — {vc} videos, {pc} poses copied."
                progress_sig.emit(summary)
                log_sig.emit(summary)
            except Exception as exc:
                msg = f"Copy failed: {exc}"
                progress_sig.emit(msg)
                log_sig.emit(msg)

        label = "Auto-copying" if auto else "Copying"
        self._copy_status_label.setText(f"{label} files to project…")
        self._append_log(f"{label} referenced files into project raw/ folder…")
        t = threading.Thread(target=_do_copy, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Relocate source paths
    # ------------------------------------------------------------------

    def _relocate_sources(self, kind: str) -> None:
        """Let the user pick a folder and update source_path for matching files."""
        if not self._manifest:
            QMessageBox.warning(self, "No Manifest", "Import files first.")
            return

        label = "video" if kind == "video" else "DLC/pose"
        folder = QFileDialog.getExistingDirectory(self, f"Select new {label} folder")
        if not folder:
            return

        folder_path = Path(folder)
        files = [f for f in folder_path.iterdir() if f.is_file()]
        # Build a lookup of available files in the chosen folder.
        available = {f.name: f for f in files}

        # For pose/DLC files, build a fallback index keyed by the DLC match-key
        # (the underlying video stem, with the scorer/model suffix stripped).
        # This lets the same subjects/sessions re-run through a *different* DLC
        # model relocate cleanly even though the full filenames no longer match.
        by_key: dict[str, list[Path]] = {}
        if kind == "pose":
            for f in files:
                if f.suffix.lower() in ImportService.POSE_EXTENSIONS:
                    by_key.setdefault(ImportService._match_key(f), []).append(f)

        assets = self._manifest.videos if kind == "video" else self._manifest.poses
        matched = 0
        lenient = 0
        ambiguous: list[str] = []
        for asset in assets:
            name = Path(asset.source_path).name
            if name in available:
                asset.source_path = str(available[name])
                # Drop any in-project copy so the new source is what's actually
                # read (local_path wins over source_path everywhere downstream).
                asset.local_path = None
                matched += 1
                continue
            if kind == "pose":
                # Lenient fallback: match by DLC stem when the exact filename
                # differs (e.g. a different DLC run of the same recordings).
                asset_path = Path(asset.source_path)
                key = ImportService._match_key(asset_path)
                candidates = by_key.get(key, [])
                if len(candidates) > 1:
                    # A recording usually exports both .csv and .h5 with the
                    # same stem; prefer the candidate matching the asset's own
                    # format so that doesn't read as a genuine ambiguity.
                    same_ext = [
                        c for c in candidates
                        if c.suffix.lower() == asset_path.suffix.lower()
                    ]
                    if len(same_ext) == 1:
                        candidates = same_ext
                if len(candidates) == 1:
                    asset.source_path = str(candidates[0])
                    asset.local_path = None
                    matched += 1
                    lenient += 1
                elif len(candidates) > 1:
                    ambiguous.append(name)

        if matched:
            # Save directly — bypass _save_manifest to avoid triggering auto-copy.
            self._import_service.save_manifest(self._project_root, self._manifest)
            # The relocated files are new data, so any features cached from the
            # old files are stale. Invalidate them so extraction rebuilds from
            # the new source (otherwise the old copies would keep being used).
            if self._project_root and kind == "pose":
                from abel.services.feature_prep_service import FeaturePrepService
                FeaturePrepService.invalidate_caches(self._project_root)
            self._populate_table(self._manifest)
            msg = f"Relocated {matched}/{len(assets)} {label} files to {folder}."
            if lenient:
                msg += (
                    f" {lenient} matched leniently by DLC stem "
                    "(different DLC run of the same recordings)."
                )
            if ambiguous:
                msg += (
                    f" Skipped {len(ambiguous)} with multiple possible matches "
                    f"({', '.join(ambiguous[:3])}{'…' if len(ambiguous) > 3 else ''})."
                )
            if kind == "pose":
                msg += (
                    " Cleared any in-project copies so the new pose files are now "
                    "active — re-run feature extraction to apply them."
                )
            self._append_log(msg)
        else:
            QMessageBox.information(
                self, "No Matches",
                f"No filenames in that folder matched the current {label} assets.",
            )

    def _append_log(self, message: str) -> None:
        self.log_panel.append(message)
