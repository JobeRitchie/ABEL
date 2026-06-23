from pathlib import Path

from abel.services.provenance_service import ProvenanceService


def test_config_hash_reproducible_for_same_payload() -> None:
    svc = ProvenanceService()
    cfg_a = {"a": 1, "b": {"x": 2, "y": 3}}
    cfg_b = {"b": {"y": 3, "x": 2}, "a": 1}

    assert svc.config_hash(cfg_a) == svc.config_hash(cfg_b)


def test_provenance_contains_required_fields(tmp_path: Path) -> None:
    svc = ProvenanceService()
    prov = svc.make_provenance(
        project_root=tmp_path,
        model_version="m1",
        feature_version="f1",
        config={"z": 1},
    )

    assert prov.app_version
    assert prov.model_version == "m1"
    assert prov.feature_version == "f1"
    assert prov.config_hash
    assert prov.timestamp is not None
