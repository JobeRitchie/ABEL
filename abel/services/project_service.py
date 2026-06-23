"""Project lifecycle and persistence orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from abel.core.constants import (
    PROJECT_CONFIG_FILES,
    PROJECT_DIRS,
    PROJECT_SCHEMA_VERSION,
)
from abel.core.exceptions import ProjectError
from abel.models.schemas import ProjectConfig, ProjectContext, ProjectState
from abel.storage.file_store import backup_file, read_json, read_yaml, write_json, write_yaml


class ProjectService:
    """Creates, opens, and updates ABEL projects on disk."""

    def create_project(
        self,
        root_dir: Path,
        config: ProjectConfig,
    ) -> ProjectContext:
        project_root = root_dir / config.project_name
        if project_root.exists():
            raise ProjectError(f"Project folder already exists: {project_root}")

        project_root.mkdir(parents=True, exist_ok=False)
        for rel in PROJECT_DIRS:
            (project_root / rel).mkdir(parents=True, exist_ok=True)

        config.schema_version = PROJECT_SCHEMA_VERSION
        config.created_at = datetime.utcnow()
        config.updated_at = datetime.utcnow()

        state = ProjectState(schema_version=PROJECT_SCHEMA_VERSION)

        write_yaml(project_root / "project.yaml", config.model_dump(mode="json"))
        write_json(project_root / "project_state.json", state.model_dump(mode="json"))

        for key, rel in PROJECT_CONFIG_FILES.items():
            target = project_root / rel
            if key == "experiment":
                write_yaml(target, {"behavior_model": config.behavior_model.model_dump(mode="json")})
            elif key == "environment_rois":
                write_yaml(
                    target,
                    {
                        "schema_version": "0.3.0",
                        "project_rois": {
                            "target_zone": {"x": 0, "y": 0, "w": 0, "h": 0},
                            "subject_crop": {"x": 0, "y": 0, "w": 0, "h": 0},
                        },
                        "subject_rois": {},
                        "motion": {
                            "local_radius_px": 36,
                        },
                    },
                )
            elif key == "behavior_adaptive_settings":
                write_yaml(
                    target,
                    {
                        "phase1": {
                            "enabled": False,
                            "enable_modality_benchmarking": True,
                            "enable_multiscale_benchmarking": True,
                            "enable_confound_analysis": True,
                            "diagnostics_enabled": True,
                            "cache_features": True,
                            "regenerate_diagnostics": False,
                            "export_high_resolution": True,
                            "save_artifacts": True,
                            "scales_sec": [0.1, 0.2, 0.25, 0.5, 1.0, 2.0],
                            "primary_metric": "ap",
                            "min_examples_per_class": 12,
                            "min_examples_for_learned_weights": 75,
                            "use_gpu_if_available": True,
                            "quick_feature_test": False,
                            "compare_all_scales": False,
                            "subset_max_sessions": 6,
                            "subset_max_segments_per_scale": 25000,
                            "cpu_parallel_workers": 0,
                            "cpu_use_process_pool": True,
                        },
                        "behavior_overrides": {},
                    },
                )
            else:
                write_yaml(target, {})

        readme_text = (
            f"ABEL Project: {config.project_name}\n"
            f"Created: {config.created_at.isoformat()}\n"
            "All project data, logs, derived artifacts, and exports live in this folder.\n"
        )
        (project_root / "README_project.txt").write_text(readme_text, encoding="utf-8")

        return ProjectContext(project_root=project_root, config=config, state=state)

    def open_project(self, project_root: Path) -> ProjectContext:
        project_file = project_root / "project.yaml"
        state_file = project_root / "project_state.json"
        if not project_file.exists() or not state_file.exists():
            raise ProjectError("Invalid project folder: missing project.yaml or project_state.json")

        config = ProjectConfig.model_validate(read_yaml(project_file, {}))
        state = ProjectState.model_validate(read_json(state_file, {}))
        state.last_opened_at = datetime.utcnow()
        self.save_state(project_root, state)
        return ProjectContext(project_root=project_root, config=config, state=state)

    def save_config(self, project_root: Path, config: ProjectConfig) -> None:
        project_file = project_root / "project.yaml"
        backup_file(project_file, project_root / "backups")
        config.updated_at = datetime.utcnow()
        write_yaml(project_file, config.model_dump(mode="json"))

    def save_state(self, project_root: Path, state: ProjectState) -> None:
        state_file = project_root / "project_state.json"
        write_json(state_file, state.model_dump(mode="json"))
