"""Saved validation sessions: the setup a run came from, and where it is filed.

A session is the audit trail for a figure — which projects were loaded, which
behaviors were checked, and every rename applied on top.  The load-bearing
invariants are that a reload reproduces the setup exactly (including which
behaviors were *un*checked), that renames survive the round trip without ever
being written back to the project on disk, and that anything that moved or
disappeared since the save is reported rather than silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from abel.validation.datamodel import ProjectRef
from abel.validation.workspace import (
    SessionRecord, SessionStore, slugify, workspace_root,
)


def _make_project(tmp_path: Path, name: str, behaviors: dict[str, str]) -> Path:
    root = tmp_path / name
    (root / "config").mkdir(parents=True)
    (root / "derived" / "training_sets").mkdir(parents=True)
    (root / "derived" / "training_sets" / "training_set.parquet").write_bytes(b"")
    (root / "project.yaml").write_text(yaml.safe_dump({"project_name": name}), encoding="utf-8")
    (root / "config" / "behavior_definitions.yaml").write_text(
        yaml.safe_dump({"behaviors": [{"behavior_id": bid, "name": nm}
                                      for bid, nm in behaviors.items()]}),
        encoding="utf-8")
    return root


@pytest.fixture
def project(tmp_path: Path) -> ProjectRef:
    root = _make_project(tmp_path, "DG_EPM", {"b1": "Grooming", "b2": "Freeze", "b3": "Rear"})
    return ProjectRef.load(root)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


# ── home location ───────────────────────────────────────────────────────────

def test_workspace_root_lives_in_the_user_home_and_honours_the_override(monkeypatch) -> None:
    monkeypatch.delenv("ABEL_VALIDATION_HOME", raising=False)
    assert workspace_root() == Path.home() / "ABEL Validation"

    monkeypatch.setenv("ABEL_VALIDATION_HOME", str(Path("D:/elsewhere")))
    assert workspace_root() == Path("D:/elsewhere")


def test_slugify_makes_a_folder_safe_stem() -> None:
    assert slugify("Manuscript — main / v2") == "manuscript-main-v2"
    assert slugify("   ") == "session"


# ── capture + round trip ────────────────────────────────────────────────────

def test_capture_records_checked_and_unchecked_behaviors(project: ProjectRef) -> None:
    rec = SessionRecord.capture("main", {project.project_id: project},
                                {project.project_id: {"b1", "b3"}})
    entry = rec.projects[0]
    assert {b.behavior_id: b.checked for b in entry.behaviors} == {
        "b1": True, "b2": False, "b3": True}


def test_round_trip_restores_projects_selection_and_both_kinds_of_rename(
    project: ProjectRef, store: SessionStore
) -> None:
    project.rename("EPM (cohort A)")
    project.set_behavior_alias("b1", "Groom")
    rec = SessionRecord.capture("main", {project.project_id: project},
                                {project.project_id: {"b1"}},
                                holdout={"holdout_seed": 7})
    store.save(rec)

    restored = store.load("main").restore()
    assert list(restored.projects) == ["EPM (cohort A)"]
    proj = restored.projects["EPM (cohort A)"]
    assert proj.original_name == "DG_EPM"          # the project on disk is untouched
    assert proj.behavior_label("b1") == "Groom"
    assert proj.behavior_disk_name("b1") == "Grooming"
    assert restored.selected == {"EPM (cohort A)": {"b1"}}
    assert store.load("main").holdout == {"holdout_seed": 7}
    assert not restored.missing_projects and not restored.missing_behaviors


def test_restore_reports_a_project_whose_folder_moved(
    project: ProjectRef, store: SessionStore, tmp_path: Path
) -> None:
    rec = SessionRecord.capture("main", {project.project_id: project},
                                {project.project_id: {"b1"}})
    store.save(rec)
    project.root.rename(tmp_path / "moved_away")

    restored = store.load("main").restore()
    assert not restored.projects
    assert restored.missing_projects and "DG_EPM" in restored.missing_projects[0]
    assert [e.project_id for e in restored.unavailable] == ["DG_EPM"]


def test_an_offline_project_survives_a_reload_and_resave(
    project: ProjectRef, store: SessionStore, tmp_path: Path
) -> None:
    """An unmounted drive must not quietly erase a project from the record.

    Reload while the drive is offline, save again, reconnect: the setup — including
    which behaviors were checked — has to still be there.
    """
    other = ProjectRef.load(_make_project(tmp_path, "OFT", {"c1": "Rear"}))
    store.save(SessionRecord.capture(
        "main", {project.project_id: project, other.project_id: other},
        {project.project_id: {"b1"}, other.project_id: {"c1"}}))

    offline = tmp_path / "offline_dg_epm"
    project.root.rename(offline)
    restored = store.load("main").restore()
    assert list(restored.projects) == ["OFT"]

    # The user carries on and saves — DG_EPM must not be dropped.
    store.save(SessionRecord.capture(
        "main", restored.projects, restored.selected,
        keep_entries=restored.unavailable))
    offline.rename(project.root)

    reloaded = store.load("main").restore()
    assert set(reloaded.projects) == {"DG_EPM", "OFT"}
    assert reloaded.selected["DG_EPM"] == {"b1"}
    assert not reloaded.unavailable


def test_a_reachable_project_with_no_training_set_is_reported_not_loaded(
    project: ProjectRef, store: SessionStore
) -> None:
    store.save(SessionRecord.capture("main", {project.project_id: project},
                                     {project.project_id: {"b1"}}))
    (project.root / "derived" / "training_sets" / "training_set.parquet").unlink()

    restored = store.load("main").restore()
    assert not restored.projects
    assert "training_set.parquet" in restored.missing_projects[0]
    assert [e.project_id for e in restored.unavailable] == ["DG_EPM"]


def test_two_projects_restoring_to_one_name_do_not_silently_merge(
    project: ProjectRef, store: SessionStore, tmp_path: Path
) -> None:
    """project_id is what every figure groups by — a collision would merge results."""
    clash = ProjectRef.load(_make_project(tmp_path, "clash", {"c1": "Rear"}))
    clash.rename("DG_EPM")
    store.save(SessionRecord.capture(
        "main", [project, clash],
        {project.project_id: {"b1"}, "DG_EPM": {"c1"}}))

    restored = store.load("main").restore()
    assert len(restored.projects) == 1
    assert "collides" in restored.missing_projects[0]
    assert len(restored.unavailable) == 1


def test_restore_reports_a_behavior_deleted_since_the_save(
    project: ProjectRef, store: SessionStore
) -> None:
    store.save(SessionRecord.capture("main", {project.project_id: project},
                                     {project.project_id: {"b1", "b2"}}))
    bd = project.root / "config" / "behavior_definitions.yaml"
    bd.write_text(yaml.safe_dump({"behaviors": [{"behavior_id": "b1", "name": "Grooming"}]}),
                  encoding="utf-8")

    restored = store.load("main").restore()
    assert restored.selected == {"DG_EPM": {"b1"}}
    assert len(restored.missing_behaviors) == 2      # b2 and b3


def test_restore_picks_up_a_behavior_added_since_the_save(
    project: ProjectRef, store: SessionStore
) -> None:
    """Reloading must not hide new labelling work — it just leaves it unchecked."""
    store.save(SessionRecord.capture("main", {project.project_id: project},
                                     {project.project_id: {"b1"}}))
    bd = project.root / "config" / "behavior_definitions.yaml"
    bd.write_text(yaml.safe_dump({"behaviors": [
        {"behavior_id": bid, "name": nm} for bid, nm in
        [("b1", "Grooming"), ("b2", "Freeze"), ("b3", "Rear"), ("b4", "Wet dog shake")]]}),
        encoding="utf-8")

    restored = store.load("main").restore()
    proj = restored.projects["DG_EPM"]
    assert "b4" in proj.behavior_names
    assert restored.selected["DG_EPM"] == {"b1"}


# ── layout: runs live inside their session ──────────────────────────────────

def test_runs_are_filed_under_the_session(project: ProjectRef, store: SessionStore) -> None:
    rec = SessionRecord.capture("Manuscript main", {project.project_id: project},
                                {project.project_id: {"b1"}})
    store.save(rec)
    runs = store.runs_dir("Manuscript main")
    assert runs == store.root / "manuscript-main" / "runs"
    assert runs.is_dir()
    assert (store.root / "manuscript-main" / "session.json").exists()
    assert (store.root / "manuscript-main" / "SETUP.md").exists()


def test_attach_to_run_freezes_the_setup_inside_the_run_dir(
    project: ProjectRef, store: SessionStore
) -> None:
    rec = SessionRecord.capture("main", {project.project_id: project},
                                {project.project_id: {"b1"}})
    store.save(rec)
    run_dir = store.runs_dir("main") / "run_2026-07-21_101500"
    run_dir.mkdir()
    store.attach_to_run(rec, run_dir)

    # Later edits to the session must not rewrite a finished run's setup.
    project.set_behavior_alias("b1", "Groom")
    store.save(SessionRecord.capture("main", {project.project_id: project},
                                     {project.project_id: {"b1", "b2"}}))

    frozen = SessionRecord.from_dict(
        json.loads((run_dir / "session.json").read_text(encoding="utf-8")))
    beh = {b.behavior_id: b for b in frozen.projects[0].behaviors}
    assert beh["b1"].display_name == "Grooming"
    assert not beh["b2"].checked


def test_list_sessions_summarizes_each_setup(project: ProjectRef, store: SessionStore) -> None:
    store.save(SessionRecord.capture("alpha", {project.project_id: project},
                                     {project.project_id: {"b1", "b2"}}))
    store.save(SessionRecord.capture("beta", {}, {}))
    (store.runs_dir("alpha") / "run_2026-07-21_101500").mkdir()

    infos = {i.name: i for i in store.list_sessions()}
    assert set(infos) == {"alpha", "beta"}
    assert infos["alpha"].n_projects == 1
    assert infos["alpha"].n_behaviors == 2
    assert infos["alpha"].n_runs == 1
    assert infos["beta"].n_projects == 0


def test_delete_removes_the_setup_but_never_the_results(
    project: ProjectRef, store: SessionStore
) -> None:
    store.save(SessionRecord.capture("main", {project.project_id: project},
                                     {project.project_id: {"b1"}}))
    run_dir = store.runs_dir("main") / "run_2026-07-21_101500"
    run_dir.mkdir()
    store.delete("main")
    assert not store.exists("main")
    assert run_dir.exists()


def test_setup_markdown_spells_out_inclusions_and_renames(project: ProjectRef) -> None:
    project.rename("EPM cohort A")
    project.set_behavior_alias("b1", "Groom")
    md = SessionRecord.capture("main", {project.project_id: project},
                               {project.project_id: {"b1"}}).to_markdown()
    assert "renamed from 'DG_EPM'" in md
    assert "Groom" in md and "renamed from 'Grooming'" in md
    assert "**Included**" in md and "**Excluded**" in md
