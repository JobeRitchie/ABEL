"""A keyboard shortcut belongs to at most one behavior.

The Review tab binds one key to one behavior, so a second behavior on the same
key is unreachable from the keyboard. The service rejects the duplicate on add
and edit, and import drops a conflicting key rather than duplicating it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abel.models.schemas import BehaviorDefinition
from abel.services.behavior_service import BehaviorService, DuplicateShortcutError
from abel.storage.file_store import write_yaml


@pytest.fixture
def svc(tmp_path: Path) -> BehaviorService:
    service = BehaviorService()
    service.set_project(tmp_path / "project")
    return service


def _beh(name: str, key: str | None) -> BehaviorDefinition:
    return BehaviorDefinition(behavior_id="", name=name, short_name=name.lower(), keyboard_shortcut=key)


def test_add_rejects_taken_key_case_insensitively(svc: BehaviorService) -> None:
    svc.add(_beh("Rear", "r"))
    with pytest.raises(DuplicateShortcutError, match="Rear"):
        svc.add(_beh("Run", "R"))
    assert [b.name for b in svc.behaviors if b.keyboard_shortcut] == ["No Behavior", "Rear"]


def test_add_rejects_the_no_behavior_key(svc: BehaviorService) -> None:
    with pytest.raises(DuplicateShortcutError):
        svc.add(_beh("Nap", "n"))


def test_update_rejects_taking_another_behaviors_key(svc: BehaviorService) -> None:
    rear = svc.add(_beh("Rear", "r"))
    groom = svc.add(_beh("Groom", "g"))
    with pytest.raises(DuplicateShortcutError):
        svc.update(groom.behavior_id, groom.model_copy(update={"keyboard_shortcut": "r"}))
    assert svc.get(groom.behavior_id).keyboard_shortcut == "g"
    assert svc.get(rear.behavior_id).keyboard_shortcut == "r"


def test_update_keeping_own_key_and_blank_keys_are_allowed(svc: BehaviorService) -> None:
    rear = svc.add(_beh("Rear", "r"))
    assert svc.update(rear.behavior_id, rear.model_copy(update={"description": "up"}))
    svc.add(_beh("Walk", None))
    svc.add(_beh("Sniff", None))
    assert svc.shortcut_owner("") is None


def test_import_drops_conflicting_keys(svc: BehaviorService, tmp_path: Path) -> None:
    svc.add(_beh("Rear", "r"))
    src = tmp_path / "defs.yaml"
    write_yaml(src, {"behaviors": [
        _beh("Run", "r").model_dump(mode="json") | {"behavior_id": "a"},
        _beh("Dart", "d").model_dump(mode="json") | {"behavior_id": "b"},
        _beh("Dig", "D").model_dump(mode="json") | {"behavior_id": "c"},
    ]})
    assert svc.import_definitions(src) == 3
    keys = {b.name: b.keyboard_shortcut for b in svc.behaviors}
    assert keys["Run"] is None
    assert keys["Dart"] == "d"
    assert keys["Dig"] is None
