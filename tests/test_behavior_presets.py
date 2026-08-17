"""Behavior presets: built-in assay sets + saving the current set as a preset.

The Behavior tab offers named starting sets of behavior definitions (Standard
Rodent, Elevated Plus Maze, Fear Conditioning, Novelty Suppressed Feeding) and
lets a user save the current project's behaviors as a reusable preset stored
outside the project. These tests pin the merge semantics — applying a preset
never duplicates or edits an existing behavior — and the user-preset round trip.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from abel.models.schemas import BehaviorDefinition
from abel.services import behavior_service as bs
from abel.services.behavior_service import NO_BEHAVIOR_ID, BehaviorService


@pytest.fixture
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BehaviorService:
    """A project-bound service whose user presets live in a temp file."""
    monkeypatch.setattr(bs, "USER_PRESETS_PATH", tmp_path / "presets" / "behavior_presets.yaml")
    service = BehaviorService()
    service.set_project(tmp_path / "project")
    return service


def _names(service: BehaviorService) -> list[str]:
    return [b.name for b in service.behaviors]


def test_builtin_presets_have_the_documented_behaviors(svc: BehaviorService) -> None:
    expected = {
        "Standard Rodent": ["Rear", "Groom", "Walk"],
        "Elevated Plus Maze": [
            "Rear", "Groom", "Walk", "Head Dip", "Protected Head Dip", "Stretch Attend",
        ],
        "Fear Conditioning": ["Rear", "Groom", "Walk", "Freeze", "Dart"],
        "Novelty Suppressed Feeding": ["Rear", "Groom", "Walk", "Approach", "Eat"],
    }
    assert svc.preset_names() == list(expected)
    for preset, names in expected.items():
        assert [d["name"] for d in svc.preset_definitions(preset)] == names


def test_apply_preset_adds_behaviors_with_definitions(svc: BehaviorService) -> None:
    added = svc.apply_preset("Fear Conditioning")

    assert added == 5
    assert _names(svc) == ["No Behavior", "Rear", "Groom", "Walk", "Freeze", "Dart"]
    freeze = next(b for b in svc.behaviors if b.name == "Freeze")
    assert freeze.operational_definition
    assert freeze.min_duration_sec == 1.0
    assert freeze.keyboard_shortcut == "z"

    # Persisted, not just in memory.
    reloaded = BehaviorService()
    reloaded.set_project(svc.project_root)
    assert _names(reloaded) == _names(svc)


def test_applying_a_second_preset_does_not_duplicate_shared_behaviors(svc: BehaviorService) -> None:
    svc.apply_preset("Standard Rodent")
    added = svc.apply_preset("Elevated Plus Maze")

    assert added == 3  # only the EPM-specific behaviors
    assert _names(svc) == [
        "No Behavior", "Rear", "Groom", "Walk",
        "Head Dip", "Protected Head Dip", "Stretch Attend",
    ]


def test_apply_preset_leaves_an_existing_behavior_untouched(svc: BehaviorService) -> None:
    original = svc.add(
        BehaviorDefinition(
            behavior_id=str(uuid.uuid4()),
            name="Groom",
            short_name="mygroom",
            description="Lab-specific definition.",
            min_duration_sec=2.5,
        )
    )
    svc.apply_preset("Standard Rodent")

    kept = svc.get(original.behavior_id)
    assert kept is not None
    assert kept.short_name == "mygroom"
    assert kept.min_duration_sec == 2.5
    assert [b.name for b in svc.behaviors].count("Groom") == 1


def test_preset_shortcut_is_dropped_when_already_claimed(svc: BehaviorService) -> None:
    svc.add(
        BehaviorDefinition(
            behavior_id=str(uuid.uuid4()),
            name="Wall Rear",
            short_name="wrear",
            keyboard_shortcut="r",
        )
    )
    svc.apply_preset("Standard Rodent")

    rear = next(b for b in svc.behaviors if b.name == "Rear")
    assert rear.keyboard_shortcut is None
    groom = next(b for b in svc.behaviors if b.name == "Groom")
    assert groom.keyboard_shortcut == "g"


def test_save_current_as_preset_round_trips_into_a_new_project(svc: BehaviorService, tmp_path: Path) -> None:
    svc.apply_preset("Standard Rodent")
    svc.add(
        BehaviorDefinition(
            behavior_id=str(uuid.uuid4()),
            name="Wet Dog Shake",
            short_name="wds",
            description="Rapid axial shake.",
            min_duration_sec=0.3,
        )
    )

    count = svc.save_preset("My Assay")
    assert count == 4  # No Behavior is excluded
    assert "My Assay" in svc.preset_names()

    other = BehaviorService()
    other.set_project(tmp_path / "project2")
    assert other.apply_preset("My Assay") == 4
    shake = next(b for b in other.behaviors if b.name == "Wet Dog Shake")
    assert shake.description == "Rapid axial shake."
    assert shake.min_duration_sec == 0.3
    # A saved preset must not carry ids from the project it came from.
    assert shake.behavior_id != next(
        b.behavior_id for b in svc.behaviors if b.name == "Wet Dog Shake"
    )


def test_saved_preset_excludes_the_system_no_behavior_label(svc: BehaviorService) -> None:
    svc.apply_preset("Standard Rodent")
    svc.save_preset("Trio")

    assert NO_BEHAVIOR_ID not in str(svc.preset_definitions("Trio"))
    assert [d["name"] for d in svc.preset_definitions("Trio")] == ["Rear", "Groom", "Walk"]


def test_save_preset_rejects_duplicate_and_builtin_names(svc: BehaviorService) -> None:
    svc.apply_preset("Standard Rodent")
    svc.save_preset("Mine")

    with pytest.raises(ValueError):
        svc.save_preset("Mine")
    with pytest.raises(ValueError):
        svc.save_preset("Standard Rodent")
    with pytest.raises(ValueError):
        svc.save_preset("   ")

    assert svc.save_preset("Mine", overwrite=True) == 3


def test_delete_preset_removes_only_user_presets(svc: BehaviorService) -> None:
    svc.apply_preset("Standard Rodent")
    svc.save_preset("Mine")

    assert svc.delete_preset("Standard Rodent") is False
    assert "Standard Rodent" in svc.preset_names()
    assert svc.delete_preset("Mine") is True
    assert "Mine" not in svc.preset_names()
    assert svc.delete_preset("Mine") is False
