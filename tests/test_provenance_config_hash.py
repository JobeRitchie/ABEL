from pathlib import Path

from abel.services.provenance_service import ProvenanceService


def test_config_hash_handles_frozenset_and_is_deterministic() -> None:
    cfg = {
        "sessions": frozenset({"session_b", "session_a"}),
        "path": Path("C:/tmp/project"),
        "nested": {"labels": set(["b", "a"]), "items": (1, 2, 3)},
    }

    h1 = ProvenanceService.config_hash(cfg)
    h2 = ProvenanceService.config_hash(cfg)

    assert isinstance(h1, str)
    assert len(h1) == 16
    assert h1 == h2


def test_config_hash_order_independent_for_sets() -> None:
    cfg_a = {"s": frozenset(["x", "y", "z"])}
    cfg_b = {"s": frozenset(["z", "x", "y"])}

    assert ProvenanceService.config_hash(cfg_a) == ProvenanceService.config_hash(cfg_b)
