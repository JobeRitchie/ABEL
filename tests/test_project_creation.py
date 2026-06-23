from pathlib import Path

from abel.models.schemas import ProjectConfig
from abel.services.project_service import ProjectService


def test_create_and_open_project(tmp_path: Path) -> None:
    service = ProjectService()
    config = ProjectConfig(project_name="demo_project")
    context = service.create_project(tmp_path, config)

    assert (context.project_root / "project.yaml").exists()
    assert (context.project_root / "project_state.json").exists()
    assert (context.project_root / "raw" / "videos").exists()

    reopened = service.open_project(context.project_root)
    assert reopened.config.project_name == "demo_project"
