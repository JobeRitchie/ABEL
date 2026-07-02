"""Main application window with startup view and workflow tabs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTabWidget,
    QWidget,
)

from abel.core.exceptions import ProjectError
from abel.models.schemas import ProjectContext
from abel.services.behavior_service import BehaviorService
from abel.services.candidate_service import CandidateGenerationService
from abel.services.dependency_service import DependencyService
from abel.services.export_service import ExportService
from abel.services.import_service import ImportService
from abel.services.logging_service import LoggingService
from abel.services.pose_features_service import PoseFeaturesService
from abel.services.preprocessing_service import ClipExtractionService
from abel.services.project_service import ProjectService
from abel.services.review_service import ReviewService
from abel.services.roi_service import ROIService
from abel.services.seed_service import SeedService
from abel.services.settings_service import SettingsService
from abel.ui.assets import icon_path
from abel.ui.dialogs import ProjectWizardDialog
from abel.ui.startup_widget import StartupWidget
from abel.ui.tabs.active_learning_tab import ActiveLearningTab
from abel.ui.tabs.behavior_tab import BehaviorTab
from abel.ui.tabs.data_import_tab import DataImportTab
from abel.ui.tabs.dependencies_tab import DependenciesTab
from abel.ui.tabs.export_tab import ExportTab
from abel.ui.tabs.feature_audit_tab import FeatureAuditTab
from abel.ui.tabs.help_tab import HelpTab
from abel.ui.tabs.info_tab import InfoTab
from abel.ui.tabs.home_tab import HomeTab
from abel.ui.tabs.logs_tab import LogsTab
from abel.ui.tabs.pose_features_tab import PoseFeaturesTab
from abel.ui.tabs.preprocessing_tab import ClipExtractionTab
from abel.ui.tabs.review_tab import ReviewTab
from abel.ui.tabs.roi_definition_tab import ROIDefinitionTab
from abel.ui.tabs.seed_examples_tab import SeedExamplesTab
from abel.ui.tabs.settings_tab import SettingsTab
from abel.ui.tabs.temporal_refinement_tab import TemporalRefinementTab
from abel.ui.tabs.temporal_review_tab import TemporalReviewTab
from abel.ui.tabs.behavior_analytics_tab import BehaviorAnalyticsTab
from abel.ui.tabs.direct_use_tab import DirectUseTab
from abel.ui.tabs.apply_models_tab import ApplyModelsTab
from abel.ui.tabs.transfer_feedback_tab import TransferFeedbackTab
from abel.ui.tabs.model_refinement_tab import ModelRefinementTab
from abel.ui.tabs.validation_tab import ValidationTab
from abel.services.validation_service import ValidationService
from abel.services.workflow_snapshot_service import WorkflowSnapshotService


class MainWindow(QMainWindow):
    """Top-level window and app flow coordinator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ABEL")
        self.setWindowIcon(QIcon(str(icon_path())))
        self._apply_initial_window_geometry()

        self._logging = LoggingService()
        self._logger = self._logging.configure_app_logging()
        self._settings_service = SettingsService()
        self._project_service = ProjectService()
        self._dependency_service = DependencyService()
        self._import_service = ImportService()
        self._behavior_service = BehaviorService()
        self._seed_service = SeedService()
        self._pose_features_service = PoseFeaturesService()
        self._candidate_service = CandidateGenerationService()
        self._clip_extraction_service = ClipExtractionService()
        self._review_service = ReviewService()
        self._export_service = ExportService()
        self._roi_service = ROIService()
        self._validation_service = ValidationService()
        self._export_service.set_behavior_service(self._behavior_service)
        self._project: ProjectContext | None = None

        self.stack = QStackedWidget()
        self.startup = StartupWidget()
        self.tabs = self._build_tabs()
        self.stack.setMinimumSize(0, 0)
        self.tabs.setMinimumSize(0, 0)

        self.stack.addWidget(self.startup)
        self.stack.addWidget(self.tabs)
        self.setCentralWidget(self.stack)

        self.startup.create_project_requested.connect(self.create_project)
        self.startup.open_project_requested.connect(self.open_project_dialog)
        self.startup.dependencies_requested.connect(self.show_dependencies_from_startup)
        self.startup.recent_project_requested.connect(self.open_recent_project)

        self._connect_home_actions()

        self._refresh_recent_projects()

    def _connect_home_actions(self) -> None:
        self.home_tab.open_folder_btn.clicked.connect(self.open_project_folder)
        self.home_tab.open_outputs_btn.clicked.connect(self.open_outputs_folder)
        self.home_tab.open_models_btn.clicked.connect(self.open_models_folder)
        self.home_tab.create_project_btn.clicked.connect(self.create_project_from_home)
        self.home_tab.open_project_requested.connect(self._open_project_from_home)
        self.home_tab.snapshot_requested.connect(self._create_snapshot_workflow)
        self.home_tab.direct_use_requested.connect(self._show_direct_use_tab)

    def _rebuild_tabs_for_project_switch(self) -> None:
        """Recreate tab widgets to guarantee per-project UI isolation."""
        old_tabs = self.tabs
        self.tabs = self._build_tabs()
        self.tabs.setMinimumSize(0, 0)
        self.stack.insertWidget(1, self.tabs)
        self.stack.removeWidget(old_tabs)
        old_tabs.deleteLater()
        self._connect_home_actions()

    def _reset_project_scoped_services(self) -> None:
        """Recreate services that may hold project-scoped in-memory state."""
        self._import_service = ImportService()
        self._behavior_service = BehaviorService()
        self._seed_service = SeedService()
        self._pose_features_service = PoseFeaturesService()
        self._candidate_service = CandidateGenerationService()
        self._clip_extraction_service = ClipExtractionService()
        self._review_service = ReviewService()
        self._export_service = ExportService()
        self._roi_service = ROIService()
        self._validation_service = ValidationService()
        self._export_service.set_behavior_service(self._behavior_service)

    def _build_app_settings_tab(self) -> QTabWidget:
        """Compose the Application subtab group: Preferences, Dependencies, Logs, Help."""
        app_tabs = QTabWidget()
        app_tabs.setTabPosition(QTabWidget.TabPosition.North)
        app_tabs.addTab(self.settings_tab, "Preferences")
        app_tabs.addTab(self.dependencies_tab, "Dependencies")
        app_tabs.addTab(self.logs_tab, "Logs")
        app_tabs.addTab(self.help_tab, "Help")
        return app_tabs

    def _build_learning_tab(self) -> QTabWidget:
        """Compose the Active Learning subtab group: Seeds, Learning, Clips, Review."""
        learning_tabs = QTabWidget()
        learning_tabs.setTabPosition(QTabWidget.TabPosition.North)
        learning_tabs.addTab(self.seed_tab, "Seeds")
        learning_tabs.addTab(self.active_learning_tab, "Learning")
        learning_tabs.addTab(self.clip_extraction_tab, "Clips")
        learning_tabs.addTab(self.feature_audit_tab, "Feature Audit")
        learning_tabs.addTab(self.review_tab, "Review")
        return learning_tabs

    def _build_temporal_tab(self) -> QTabWidget:
        """Compose the Temporal subtab group: Refinement, Review."""
        temporal_tabs = QTabWidget()
        temporal_tabs.setTabPosition(QTabWidget.TabPosition.North)
        temporal_tabs.addTab(self.temporal_refinement_tab, "Refinement")
        temporal_tabs.addTab(self.temporal_review_tab, "Review")
        return temporal_tabs

    def _build_direct_use_tab(self) -> QTabWidget:
        """Compose the Direct Use subtab group: Run, Apply Models, Transfer Feedback."""
        direct_use_tabs = QTabWidget()
        direct_use_tabs.setTabPosition(QTabWidget.TabPosition.North)
        direct_use_tabs.addTab(self.direct_use_tab, "Run")
        direct_use_tabs.addTab(self.apply_models_tab, "Apply Models")
        direct_use_tabs.addTab(self.transfer_feedback_tab, "Transfer Feedback")
        return direct_use_tabs

    def _apply_initial_window_geometry(self) -> None:
        """Size and place the window within the visible screen area."""
        app = QApplication.instance() or QGuiApplication.instance()
        if app is None:
            self.resize(1280, 800)
            return

        screen = self.screen() or app.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return

        available = screen.availableGeometry()
        target_width = min(1280, max(960, int(available.width() * 0.94)))
        target_height = min(800, max(640, int(available.height() * 0.92)))
        self.resize(target_width, target_height)

        min_width = min(target_width, max(720, int(available.width() * 0.65)))
        min_height = min(target_height, max(520, int(available.height() * 0.6)))
        self.setMinimumSize(min_width, min_height)

        center_x = available.x() + (available.width() - target_width) // 2
        center_y = available.y() + (available.height() - target_height) // 2
        self.move(center_x, center_y)

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        tabs.setMinimumSize(0, 0)

        # --- Instantiate all leaf-level tab widgets ---
        self.home_tab = HomeTab()
        self.dependencies_tab = DependenciesTab(self._dependency_service)
        self.data_import_tab = DataImportTab(self._import_service)
        self.data_import_tab.num_animals_changed.connect(self._on_num_animals_changed)
        self.behavior_tab = BehaviorTab(self._behavior_service)
        self.seed_tab = SeedExamplesTab(
            self._seed_service, self._behavior_service, self._import_service
        )
        self.pose_features_tab = PoseFeaturesTab(
            self._pose_features_service, self._import_service, self._behavior_service,
            roi_service=self._roi_service,
        )
        self.roi_tab = ROIDefinitionTab(self._roi_service, self._import_service)
        self.active_learning_tab = ActiveLearningTab(
            self._import_service,
            self._seed_service,
            self._behavior_service,
            self._candidate_service,
        )
        self.clip_extraction_tab = ClipExtractionTab(
            self._clip_extraction_service,
            self._candidate_service,
            self._import_service,
            self._behavior_service,
        )
        self.feature_audit_tab = FeatureAuditTab()
        self.active_learning_tab.edge_case_candidates_requested.connect(self._load_edge_cases_in_clip_extraction)
        self.active_learning_tab.uncertainty_candidates_updated.connect(self._update_clip_extraction_uncertainty_candidates)
        self.active_learning_tab.uncertainty_candidates_appended.connect(self._append_clip_extraction_uncertainty_candidates)
        self.pose_features_tab.segmentation_completed.connect(self.active_learning_tab._refresh_segment_settings_display)
        self.review_tab = ReviewTab(
            self._review_service,
            self._candidate_service,
            self._import_service,
            self._behavior_service,
        )
        # When the user toggles "Allow co-occurring behaviors" in the Behaviors tab,
        # immediately refresh the co-occurring UI state in the Review tab so the
        # user doesn't need to manually click Refresh.
        self.behavior_tab.co_occurring_changed.connect(self.review_tab._load_co_occurring_setting)
        # When behaviors are added/edited/deleted in the Behaviors tab, refresh the
        # Active Learning target-behavior dropdown so new behaviors appear without
        # requiring a project reload.
        self.behavior_tab.behaviors_changed.connect(
            self.active_learning_tab._refresh_behavior_options
        )
        self.temporal_refinement_tab = TemporalRefinementTab()
        self.temporal_review_tab = TemporalReviewTab()
        self.temporal_review_tab.bout_candidates_requested.connect(self._load_bout_candidates_in_clip_extraction)
        self.temporal_review_tab.bout_candidates_append_requested.connect(self._append_bout_candidates_in_clip_extraction)
        self.behavior_analytics_tab = BehaviorAnalyticsTab()
        self.validation_tab = ValidationTab(
            self._validation_service,
            self._behavior_service,
        )
        self.export_tab = ExportTab(
            self._export_service,
            self._candidate_service,
            self._review_service,
            self._behavior_service,
        )
        self.logs_tab = LogsTab()
        self.settings_tab = SettingsTab(self._settings_service)
        self.help_tab = HelpTab()
        self.info_tab = InfoTab()

        self.direct_use_tab = DirectUseTab()
        self.direct_use_tab.pipeline_complete.connect(self._on_direct_use_complete)
        self.apply_models_tab = ApplyModelsTab()
        self.transfer_feedback_tab = TransferFeedbackTab()
        # After a Direct Use run completes, pre-fill the feedback target.
        self.direct_use_tab.pipeline_complete.connect(
            self.transfer_feedback_tab.set_target_project
        )

        self.model_refinement_tab = ModelRefinementTab()
        self.model_refinement_tab.retrain_requested.connect(
            self._show_active_learning_tab
        )

        # --- Compose grouped tabs ---
        self._learning_group = self._build_learning_tab()
        self._temporal_group = self._build_temporal_tab()
        self._direct_use_group = self._build_direct_use_tab()
        self._app_settings_group = self._build_app_settings_tab()

        # --- Top-level tabs (consolidated) ---
        tabs.addTab(self.home_tab, "Home")
        tabs.addTab(self.data_import_tab, "Data Import")
        tabs.addTab(self.behavior_tab, "Behaviors")
        tabs.addTab(self.roi_tab, "ROI")
        tabs.addTab(self.pose_features_tab, "Features")
        tabs.addTab(self._learning_group, "Active Learning")
        tabs.addTab(self._temporal_group, "Temporal")
        tabs.addTab(self.behavior_analytics_tab, "Analytics")
        tabs.addTab(self.validation_tab, "Validation")
        tabs.addTab(self.export_tab, "Export")
        tabs.addTab(self._direct_use_group, "Direct Use")
        tabs.addTab(self.model_refinement_tab, "Model Refinement")
        tabs.addTab(self._app_settings_group, "Settings")
        tabs.addTab(self.info_tab, "Info")

        for i in range(tabs.count()):
            page = tabs.widget(i)
            if page is not None:
                page.setMinimumSize(0, 0)

        # Wire lazy-init signals — tabs initialize only on first visit
        self._initialized_tabs: set[QWidget] = set()
        tabs.currentChanged.connect(self._on_top_tab_changed)
        self._learning_group.currentChanged.connect(self._on_sub_tab_changed)
        self._temporal_group.currentChanged.connect(self._on_sub_tab_changed)
        self._direct_use_group.currentChanged.connect(self._on_sub_tab_changed)
        self._app_settings_group.currentChanged.connect(self._on_sub_tab_changed)
        return tabs

    def _load_edge_cases_in_clip_extraction(self, candidates: list, source_label: str) -> None:
        """Receive edge-case rows from active learning and load them into clip extraction."""
        if not candidates:
            return
        self.clip_extraction_tab.load_external_candidates(candidates, source_label=source_label)
        self.tabs.setCurrentWidget(self._learning_group)
        self._learning_group.setCurrentWidget(self.clip_extraction_tab)

    def _update_clip_extraction_uncertainty_candidates(self, candidates: list, source_label: str) -> None:
        """Pre-populate clip extraction with uncertainty-ranked candidates after each pipeline run.

        Unlike edge-case candidates this does NOT switch the active tab — the user stays on the
        Active Learning tab to review metrics and can navigate to Clip Extraction when ready.
        The candidates replace any previous AL-uncertainty set so re-runs are clean.
        """
        if not candidates:
            return
        self.clip_extraction_tab.load_external_candidates(
            candidates,
            source_label=source_label,
            clear_source="active_learning_uncertainty",
        )

    def _append_clip_extraction_uncertainty_candidates(self, candidates: list, source_label: str) -> None:
        """Accumulate uncertainty-ranked candidates from a batch run into clip extraction.

        Unlike :meth:`_update_clip_extraction_uncertainty_candidates`, this does NOT clear
        the previous active-learning-uncertainty set, so review clips from each batch run
        (Retrain All / Pipeline All) add up in the Clips tab instead of replacing the
        previous model training's clips.  Upsert is keyed by window_id, so re-running the
        same behavior refreshes rather than duplicates its clips.
        """
        if not candidates:
            return
        self.clip_extraction_tab.load_external_candidates(candidates, source_label=source_label)

    def _load_bout_candidates_in_clip_extraction(self, candidates: list, source_label: str) -> None:
        """Receive bout-review candidates from temporal review and load them into clip extraction."""
        if not candidates:
            return
        self.clip_extraction_tab.load_external_candidates(
            candidates,
            source_label=source_label,
            clear_source="temporal_bout_review",
        )
        self.tabs.setCurrentWidget(self._learning_group)
        self._learning_group.setCurrentWidget(self.clip_extraction_tab)

    def _append_bout_candidates_in_clip_extraction(self, candidates: list, source_label: str) -> None:
        """Append bout-review candidates into clip extraction without clearing existing clips."""
        if not candidates:
            return
        self.clip_extraction_tab.load_external_candidates(
            candidates,
            source_label=source_label,
        )
        self.tabs.setCurrentWidget(self._learning_group)
        self._learning_group.setCurrentWidget(self.clip_extraction_tab)

    def _refresh_recent_projects(self) -> None:
        recent = self._settings_service.load_recent_projects()
        self.startup.set_recent_projects(recent)

    def show_dependencies_from_startup(self) -> None:
        self.stack.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self._app_settings_group)
        self._app_settings_group.setCurrentWidget(self.dependencies_tab)

    def create_project(self) -> None:
        dialog = ProjectWizardDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        root, cfg = dialog.to_project_config()
        if not cfg.project_name:
            self._error("Project name is required.")
            return

        try:
            context = self._project_service.create_project(root, cfg)
            self._set_project(context)
        except ProjectError as exc:
            self._error(str(exc))

    def create_project_from_home(self) -> None:
        """Prompt before replacing the active project context from the Home tab."""
        if self._project is not None:
            response = QMessageBox.question(
                self,
                "Create New Project",
                "A project is currently open. Creating a new project will switch the active project. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        self.create_project()

    def open_project_dialog(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select ABEL project folder")
        if not selected:
            return
        self.open_recent_project(selected)

    def open_recent_project(self, path: str) -> None:
        project_root = Path(path)
        try:
            context = self._project_service.open_project(project_root)
            self._set_project(context)
        except Exception as exc:
            self._error(f"Could not open project: {exc}")

    def _set_project(self, context: ProjectContext) -> None:
        previous_root = self._project.project_root if self._project is not None else None
        if previous_root is not None and previous_root != context.project_root:
            self._reset_project_scoped_services()
            self._rebuild_tabs_for_project_switch()

        self._project = context
        self._logging.attach_project_handler(context.project_root)
        settings = self._settings_service.load_app_settings()
        self._settings_service.add_recent_project(
            context.project_root,
            max_items=settings.max_recent_projects,
        )
        self._refresh_recent_projects()
        self._pose_features_service.set_project(context.project_root)

        # ── Phase 1: fast tabs — window becomes visible now ──────────
        self.home_tab.update_project(context.config.project_name, context.project_root)
        self.logs_tab.set_project(context.project_root)
        self.settings_tab.set_project(context.project_root)
        self.help_tab.set_project(context.project_root)
        self.direct_use_tab.set_project(context.project_root)
        self.apply_models_tab.set_project(context.project_root)
        # Eagerly initialize signal-receiving tabs so they have _project_root set
        # before any pipeline signal (e.g. uncertainty_candidates_updated) fires.
        # Both use deferred I/O internally so there is no startup cost.
        self.active_learning_tab.set_project(context.project_root)
        self.clip_extraction_tab.set_project(context.project_root)

        # Phase 1 tabs are done; remaining tabs initialize on first visit
        self._initialized_tabs = {
            self.home_tab, self.logs_tab, self.settings_tab,
            self.help_tab, self.direct_use_tab,
            self.active_learning_tab, self.clip_extraction_tab,
        }

        self.stack.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.home_tab)

        # Defer only the lightweight stats refresh for the home tab
        root = context.project_root
        QTimer.singleShot(0, lambda: self._update_home_stats(root))

    def _on_num_animals_changed(self, n: int) -> None:
        """Persist a change to the project's animal count (keeps in-memory config in sync)."""
        if self._project is None:
            return
        cfg = self._project.config
        cfg.num_animals = int(n)
        cfg.single_animal = int(n) <= 1
        self._project_service.save_config(self._project.project_root, cfg)
        self._logger.info("num_animals set to %d (single_animal=%s)", n, cfg.single_animal)

    def _update_home_stats(self, project_root: Path) -> None:
        """Deferred: update home-tab stats after the window has painted."""
        if self._project is None or self._project.project_root != project_root:
            return
        self.home_tab.update_stats(self._compute_project_stats(project_root))
        self._logger.info("Project loaded: %s", project_root)

    def _lazy_init_tab(self, widget: QWidget) -> None:
        """Call set_project on *widget* exactly once; no-op if already done."""
        if widget in self._initialized_tabs:
            return
        if self._project is None:
            return
        root = self._project.project_root
        self._initialized_tabs.add(widget)
        # Sync subject order from analytics tab to export service
        if widget is self.export_tab:
            order = self.behavior_analytics_tab.ordered_session_labels()
            self._export_service.set_subject_order(order)
        if widget is self.data_import_tab:
            widget.set_project_root(root)
        elif hasattr(widget, "set_project"):
            widget.set_project(root)

    def _on_top_tab_changed(self, index: int) -> None:
        """Lazy-init the tab that just became visible at the top level."""
        widget = self.tabs.widget(index)
        if widget is None:
            return
        # Always sync subject order when switching to the export tab
        if widget is self.export_tab and self._project is not None:
            order = self.behavior_analytics_tab.ordered_session_labels()
            self._export_service.set_subject_order(order)
        if widget in (self._learning_group, self._temporal_group, self._app_settings_group):
            sub = widget.currentWidget()
            if sub is not None:
                self._lazy_init_tab(sub)
        else:
            self._lazy_init_tab(widget)

    def _on_sub_tab_changed(self, index: int) -> None:
        """Lazy-init the sub-tab that just became visible inside a group."""
        group = self.sender()
        if not isinstance(group, QTabWidget):
            return
        widget = group.widget(index)
        if widget is not None:
            self._lazy_init_tab(widget)

    def _compute_project_stats(self, project_root: Path) -> dict:
        """Fast stats read from persisted files for the home tab."""
        import json  # noqa: PLC0415
        import re    # noqa: PLC0415
        import yaml  # noqa: PLC0415
        stats: dict = {}
        manifest_path = project_root / "derived" / "review_tables" / "import_manifest.json"
        if manifest_path.exists():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                stats["sessions"] = len(raw.get("linked_sessions", []))
            except Exception:
                pass
        behavior_path = project_root / "config" / "behavior_definitions.yaml"
        if behavior_path.exists():
            try:
                raw_b = yaml.safe_load(behavior_path.read_text(encoding="utf-8")) or {}
                behaviors = raw_b.get("behaviors", [])
                stats["behaviors"] = len(behaviors)
                stats["behavior_names"] = [
                    b.get("name", b.get("behavior_id", "?")) for b in behaviors if b.get("active", True)
                ]
            except Exception:
                pass
        seeds_path = project_root / "config" / "seeds.json"
        if seeds_path.exists():
            try:
                raw_s = json.loads(seeds_path.read_text(encoding="utf-8"))
                stats["seeds"] = len(raw_s.get("seeds", []))
            except Exception:
                pass
        clips_path = project_root / "derived" / "review_tables" / "clip_manifest.json"
        if clips_path.exists():
            try:
                raw_c = json.loads(clips_path.read_text(encoding="utf-8"))
                stats["clips"] = len(raw_c.get("clips", []))
            except Exception:
                pass

        # ── Pipeline settings ────────────────────────────────────────────
        pipeline: dict = {}
        snapshot_path = project_root / "derived" / "workflow_snapshot.json"
        if snapshot_path.exists():
            try:
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                fps = float(snap.get("fps") or 30.0)
                win_fr = int(snap.get("segment_window_frames") or 0)
                str_fr = int(snap.get("segment_stride_frames") or 0)
                rs = snap.get("run_settings") or {}
                pipeline = {
                    "window_frames": win_fr,
                    "window_sec": round(win_fr / fps, 3) if fps > 0 and win_fr else None,
                    "stride_frames": str_fr,
                    "stride_sec": round(str_fr / fps, 3) if fps > 0 and str_fr else None,
                    "fps": fps,
                    "model_version": snap.get("model_version") or "—",
                    "query_mode": rs.get("mode") or "—",
                    "classifier": None,
                }
            except Exception:
                pass
        proj_yaml = project_root / "project.yaml"
        if proj_yaml.exists():
            try:
                raw_p = yaml.safe_load(proj_yaml.read_text(encoding="utf-8")) or {}
                bm = raw_p.get("behavior_model") or {}
                pipeline["classifier"] = bm.get("classifier_type") or "—"
                # Fill window/stride from project config if snapshot didn't supply them
                if not pipeline.get("window_frames"):
                    fps = pipeline.get("fps") or float(raw_p.get("default_fps") or 30.0)
                    win_fr = int(bm.get("segment_window_frames") or 0)
                    str_fr = int(bm.get("segment_stride_frames") or 0)
                    pipeline.update({
                        "window_frames": win_fr,
                        "window_sec": round(win_fr / fps, 3) if fps > 0 and win_fr else None,
                        "stride_frames": str_fr,
                        "stride_sec": round(str_fr / fps, 3) if fps > 0 and str_fr else None,
                        "fps": fps,
                    })
            except Exception:
                pass
        if pipeline:
            stats["pipeline"] = pipeline

        # ── Model performance metrics ────────────────────────────────────
        metrics_path = project_root / "derived" / "evaluation" / "model_metrics.json"
        if metrics_path.exists():
            try:
                text = metrics_path.read_text(encoding="utf-8")
                # JSON does not allow bare NaN — replace before parsing
                text = re.sub(r'\bNaN\b', 'null', text)
                raw_m = json.loads(text)

                def _fmt(v: object) -> str:
                    if v is None:
                        return "—"
                    try:
                        return f"{float(v):.3f}"
                    except (TypeError, ValueError):
                        return "—"

                fl = raw_m.get("frame_level") or {}
                sl = raw_m.get("segment_level") or {}
                stats["model_metrics"] = {
                    "frame_f1":        _fmt(fl.get("f1")),
                    "frame_precision": _fmt(fl.get("precision")),
                    "frame_recall":    _fmt(fl.get("recall")),
                    "frame_pr_auc":    _fmt(fl.get("pr_auc")),
                    "segment_f1":      _fmt(sl.get("f1")),
                }
            except Exception:
                pass

        return stats

    def open_project_folder(self) -> None:
        if not self._project:
            self._error("No project loaded.")
            return
        os.startfile(str(self._project.project_root))

    def open_outputs_folder(self) -> None:
        if not self._project:
            self._error("No project loaded.")
            return
        os.startfile(str(self._project.project_root / "exports"))

    def open_models_folder(self) -> None:
        if not self._project:
            self._error("No project loaded.")
            return
        models_dir = self._project.project_root / "derived" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(models_dir))

    def _open_project_from_home(self) -> None:
        """Open another project via the Home tab button."""
        if self._project is not None:
            response = QMessageBox.question(
                self,
                "Open Project",
                "A project is currently open. Opening another project will switch context. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        self.open_project_dialog()

    def _create_snapshot_workflow(self) -> None:
        """Serialize the current project's pipeline settings into a snapshot."""
        if self._project is None:
            self._error("No project is loaded.")
            return
        svc = WorkflowSnapshotService()
        try:
            snapshot = svc.build_from_project(self._project.project_root)
            svc.save(self._project.project_root, snapshot)

            # Build a summary of captured behaviours
            sbm = snapshot.selected_behavior_models or {}
            excluded = set(snapshot.excluded_behavior_ids or [])
            beh_lookup: dict[str, str] = {}
            for b in snapshot.behavior_definitions:
                bid = b.get("behavior_id", b.get("name", ""))
                beh_lookup[bid] = b.get("name", b.get("short_name", bid))

            model_lines: list[str] = []
            for bid, mver in sbm.items():
                name = beh_lookup.get(bid, bid)
                if bid not in excluded:
                    model_lines.append(f"  • {name} → {mver}")

            models_text = "\n".join(model_lines) if model_lines else f"  • {snapshot.model_version}"

            QMessageBox.information(
                self,
                "Snapshot Created",
                f"Workflow snapshot saved to:\n"
                f"{self._project.project_root / 'derived' / 'workflow_snapshot.json'}\n\n"
                f"Behavior Models:\n{models_text}\n\n"
                "This snapshot can be used in a Direct Use workflow.",
            )
        except Exception as exc:
            self._error(f"Failed to create workflow snapshot:\n{exc}")

    def _show_direct_use_tab(self) -> None:
        """Switch to the Direct Use tab, pre-seeding the source from the current project."""
        if self._project is not None:
            self.direct_use_tab.set_source_from_current(self._project.project_root)
        self.tabs.setCurrentWidget(self._direct_use_group)
        self._direct_use_group.setCurrentWidget(self.direct_use_tab)

    def _show_active_learning_tab(self) -> None:
        """Switch to the Active Learning group (Learning sub-tab) to retrain."""
        self._lazy_init_tab(self.active_learning_tab)
        self.tabs.setCurrentWidget(self._learning_group)
        self._learning_group.setCurrentWidget(self.active_learning_tab)

    def _on_direct_use_complete(self, target_root: Path) -> None:
        """Handle pipeline completion — offer to switch to analytics."""
        reply = QMessageBox.question(
            self,
            "Pipeline Complete",
            "The Direct Use pipeline has finished.\n\n"
            "Would you like to open the target project and view the Analytics tab?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.open_recent_project(str(target_root))
            self.tabs.setCurrentWidget(self.behavior_analytics_tab)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Gracefully drain background workers before the process exits.

        Without this, QThreadPool workers that are executing numpy/scipy
        code hold Intel MKL internal threads alive.  When Qt destroys the
        window those MKL threads receive a WM_CLOSE message and call
        ``forrtl: error (200): program aborting due to window-CLOSE event``
        which terminates the process abnormally and leaves file handles
        (including the launcher's 2>> log redirect) unreleased.
        """
        # 1. Stop MKL/OpenMP from creating new threads; request idle exit.
        _shutdown_mkl_threads()

        # 2. Give running QThreadPool workers up to 4 s to complete.
        pool = QThreadPool.globalInstance()
        pool.waitForDone(4000)

        # 3. Flush and close all logging handlers to release file handles.
        logging.shutdown()

        super().closeEvent(event)

    def _error(self, message: str) -> None:
        self._logger.error(message)
        QMessageBox.critical(self, "ABEL", message)


def build_window() -> QWidget:
    return MainWindow()


def _shutdown_mkl_threads() -> None:
    """Tell numpy/MKL/OpenMP to stop spinning background threads.

    Intel MKL worker threads run a Windows message pump on Windows.
    If they receive a WM_CLOSE message while still active they call
    ``forrtl: error (200): program aborting due to window-CLOSE event``
    which crashes the process without releasing file handles.
    Setting the thread count to 1 makes them exit before the main
    thread's event loop ends.
    """
    try:
        import os as _os
        _os.environ["OMP_NUM_THREADS"] = "1"
        _os.environ["MKL_NUM_THREADS"] = "1"
        _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except Exception:
        pass
    try:
        import numpy as _np
        _np.__config__  # noqa: B018 – force attribute access to ensure module is loaded
    except Exception:
        pass
