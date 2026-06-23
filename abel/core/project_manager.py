"""Project manager entry points for temporal refinement stage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from abel.temporal_refinement.temporal_refinement_service import (
    TemporalRefinementConfig,
    TemporalRefinementService,
)


class ProjectManager:
    """High-level orchestration wrapper for temporal refinement operations."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._temporal = TemporalRefinementService()
        self._temporal.set_project(project_root)

    def run_temporal_refinement_training(
        self,
        concept_id: str,
        config: TemporalRefinementConfig | None = None,
        model_name: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict:
        return self._temporal.run_temporal_refinement_training(
            concept_id=concept_id,
            config=config,
            model_name=model_name,
            force=False,
            progress_cb=progress_cb,
        )

    def list_temporal_training_runs(self, concept_id: str) -> list[dict]:
        return self._temporal.list_temporal_training_runs(concept_id=concept_id)

    def set_active_temporal_training_run(self, concept_id: str, training_dir: str) -> dict:
        return self._temporal.set_active_temporal_training_run(concept_id=concept_id, training_dir=training_dir)

    def run_temporal_refinement_inference(
        self,
        concept_id: str,
        sessions: list[str] | None = None,
        mode: str = "dense",
        config: TemporalRefinementConfig | None = None,
        max_sessions: int | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict:
        return self._temporal.run_temporal_refinement_inference(
            concept_id=concept_id,
            sessions=sessions,
            mode=mode,
            config=config,
            force=max_sessions is not None,
            max_sessions=max_sessions,
            progress_cb=progress_cb,
        )

    def run_temporal_refinement_postprocess(
        self,
        concept_id: str,
        sessions: list[str] | None = None,
        config: TemporalRefinementConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict:
        return self._temporal.run_temporal_refinement_postprocess(
            concept_id=concept_id,
            sessions=sessions,
            config=config,
            force=False,
            progress_cb=progress_cb,
        )

    def load_temporal_feedback(self, concept_id: str) -> dict:
        return self._temporal.load_temporal_feedback(concept_id=concept_id)

    def add_temporal_feedback_interval(
        self,
        concept_id: str,
        session_id: str,
        start_frame: int,
        end_frame: int,
        feedback_type: str,
    ) -> dict:
        return self._temporal.add_temporal_feedback_interval(
            concept_id=concept_id,
            session_id=session_id,
            start_frame=start_frame,
            end_frame=end_frame,
            feedback_type=feedback_type,
        )

    def remove_temporal_feedback_interval(
        self,
        concept_id: str,
        session_id: str,
        start_frame: int,
        end_frame: int,
        feedback_type: str,
    ) -> dict:
        return self._temporal.remove_temporal_feedback_interval(
            concept_id=concept_id,
            session_id=session_id,
            start_frame=start_frame,
            end_frame=end_frame,
            feedback_type=feedback_type,
        )

    def clear_temporal_refinement_cache(
        self,
        concept_id: str | None = None,
        clear_run_artifacts: bool = False,
    ) -> dict:
        return self._temporal.clear_temporal_tab_cache(
            concept_id=concept_id,
            clear_run_artifacts=clear_run_artifacts,
        )
