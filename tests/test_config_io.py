from pathlib import Path

from abel.models.schemas import AppSettings
from abel.storage.file_store import read_json, read_yaml, write_json, write_yaml


def test_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    settings = AppSettings(theme="dark")
    write_yaml(path, settings.model_dump(mode="json"))
    loaded = read_yaml(path, {})
    assert loaded["theme"] == "dark"


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {"a": 1, "b": "x"}
    write_json(path, payload)
    loaded = read_json(path, {})
    assert loaded == payload
