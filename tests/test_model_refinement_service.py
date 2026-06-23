"""Tests for cross-project example import (model refinement)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from abel.services.model_refinement_service import ModelRefinementService


def _write_behaviors(root: Path, behaviors: list[dict]) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "behavior_definitions.yaml").write_text(
        yaml.dump({"behaviors": behaviors}), encoding="utf-8",
    )


def _feature_frame(segment_ids: list[str], feat_cols: list[str], sessions: list[str]) -> pd.DataFrame:
    data: dict = {
        "segment_id": segment_ids,
        "session_id": sessions,
        "animal_id": ["a"] * len(segment_ids),
        "start_frame": list(range(len(segment_ids))),
        "end_frame": [i + 14 for i in range(len(segment_ids))],
    }
    for i, c in enumerate(feat_cols):
        data[c] = [float(i + j) for j in range(len(segment_ids))]
    return pd.DataFrame(data)


def _make_host(tmp_path: Path, feat_cols: list[str]) -> Path:
    """Host project with an existing training set + behaviour defs."""
    root = tmp_path / "host"
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "host-walk", "name": "Walk"},
        {"behavior_id": "host-rear", "name": "Rear"},
    ])
    ts_dir = root / "derived" / "training_sets"
    ts_dir.mkdir(parents=True, exist_ok=True)
    df = _feature_frame(["seg_h1", "seg_h2"], feat_cols, ["hs1", "hs1"])
    df["label"] = ["host-walk", "no_behavior"]
    df["label_source"] = "review"
    df["reviewer_confidence"] = 1.0
    df.to_parquet(ts_dir / "training_set.parquet", index=False)
    return root


def _make_source(tmp_path: Path, feat_cols: list[str], name: str = "src") -> Path:
    """Source project with segment_features + reviewer labels."""
    root = tmp_path / name
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-walk", "name": "Walk"},     # matches host "Walk"
        {"behavior_id": "src-jump", "name": "Jump"},      # no host match
    ])
    rep = root / "derived" / "representations"
    rep.mkdir(parents=True, exist_ok=True)
    seg_ids = ["seg_s1", "seg_s2", "seg_s3", "seg_s4"]
    feats = _feature_frame(seg_ids, feat_cols, ["ss1"] * 4)
    feats.to_parquet(rep / "segment_features.parquet", index=False)

    lab_dir = root / "derived" / "review_labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({
        "segment_id": seg_ids,
        "review_label": ["src-walk", "src-walk", "no_behavior", "src-jump"],
        "reviewer_id": ["r"] * 4,
        "confidence": [1.0] * 4,
        "timestamp": ["2026-01-01T00:00:0" + str(i) for i in range(4)],
    })
    labels.to_parquet(lab_dir / "reviewer_labels.parquet", index=False)
    return root


FEATS = [f"feat_{i}" for i in range(20)]


def test_preview_compatible_same_schema(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS)
    pv = ModelRefinementService().preview(host, src)
    assert pv.compatible, pv.reason
    assert pv.coverage == 1.0
    # Walk (2) + no_behavior (1) are importable; Jump (1) is not.
    assert pv.importable_labeled == 3
    matched_names = {m.host_name for m in pv.matched_behaviors}
    assert "Walk" in matched_names and "No Behavior" in matched_names
    assert any(m.source_name == "Jump" for m in pv.unmatched_behaviors)


def test_preview_blocks_incompatible_schema(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    # Source shares only 2 of 20 host feature columns.
    src = _make_source(tmp_path, ["feat_0", "feat_1"] + [f"other_{i}" for i in range(18)])
    pv = ModelRefinementService().preview(host, src)
    assert not pv.compatible
    assert "Incompatible feature schemas" in pv.reason
    assert pv.coverage < 0.95


def test_preview_requires_host_training_set(tmp_path: Path) -> None:
    root = tmp_path / "host"
    _write_behaviors(root, [{"behavior_id": "no_behavior", "name": "No Behavior"}])
    src = _make_source(tmp_path, FEATS)
    pv = ModelRefinementService().preview(root, src)
    assert not pv.compatible
    assert "no training set" in pv.reason.lower()


def test_import_merges_namespaced_rows(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="DonorA")
    svc = ModelRefinementService()
    result = svc.import_examples(host, src)
    assert result["status"] == "success", result.get("error")
    # 3 importable labels (2 Walk + 1 no_behavior); Jump dropped.
    assert result["imported_rows"] == 3

    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    # 2 original host rows + 3 imported.
    assert len(ts) == 5
    imported = ts[ts["label_source"] == "imported:DonorA"]
    assert len(imported) == 3
    # Labels were remapped to host behaviour ids.
    assert set(imported["label"]) == {"host-walk", "no_behavior"}
    # segment_ids are namespaced to avoid collisions.
    assert all(sid.startswith("DonorA__") for sid in imported["segment_id"])
    # Column schema is preserved exactly.
    assert list(ts.columns) == list(
        pd.read_parquet(
            host / "derived" / "training_sets" / "snapshots"
            / Path(result["snapshot_path"]).name
        ).columns
    )


def _make_source_with_training_set(
    tmp_path: Path, feat_cols: list[str], name: str = "srcTS",
) -> Path:
    """Source whose assembled training set holds the project's full labeled set,
    far more than its sparse Review-tab log (mirrors real projects, where the
    review log is a tiny subset of what the model actually trained on)."""
    root = tmp_path / name
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-walk", "name": "Walk"},   # matches host "Walk"
        {"behavior_id": "src-rear", "name": "Rear"},   # matches host "Rear"
        {"behavior_id": "src-jump", "name": "Jump"},   # no host match
    ])
    # Sparse review log: a single reviewed segment (all the OLD importer saw).
    lab_dir = root / "derived" / "review_labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "segment_id": ["seg_s1"],
        "review_label": ["no_behavior"],
        "reviewer_id": ["r"],
        "confidence": [1.0],
        "timestamp": ["2026-01-01T00:00:00"],
    }).to_parquet(lab_dir / "reviewer_labels.parquet", index=False)
    # The real labeled set: 6 segments across reviewer/seed/feedback sources.
    seg_ids = [f"seg_s{i}" for i in range(1, 7)]
    ts = _feature_frame(seg_ids, feat_cols, ["ss1"] * 6)
    ts["label"] = ["src-walk", "src-walk", "src-rear", "no_behavior", "src-jump", "src-walk"]
    ts["label_source"] = ["reviewer", "seed", "temporal_feedback", "reviewer", "reviewer", "seed"]
    ts["reviewer_confidence"] = 1.0
    ts_dir = root / "derived" / "training_sets"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(ts_dir / "training_set.parquet", index=False)
    return root


def test_import_pulls_full_training_set_not_just_review_log(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)  # behaviours: No Behavior, Walk, Rear
    src = _make_source_with_training_set(tmp_path, FEATS, name="DonorTS")
    svc = ModelRefinementService()

    pv = svc.preview(host, src)
    assert pv.compatible, pv.reason
    # All 6 training-set labels are considered, not the 1-row review log.
    assert pv.total_labeled == 6
    # Walk(3) + Rear(1) + no_behavior(1) map to host; Jump(1) does not.
    assert pv.importable_labeled == 5
    assert any(m.source_name == "Jump" for m in pv.unmatched_behaviors)

    result = svc.import_examples(host, src)
    assert result["status"] == "success", result.get("error")
    # 5 imported is impossible from the 1-row review log — proves the training
    # set is the label source now.
    assert result["imported_rows"] == 5
    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    imported = ts[ts["label_source"] == "imported:DonorTS"]
    assert len(imported) == 5
    # Labels were remapped onto host behaviour ids (Walk/Rear/no_behavior).
    assert set(imported["label"]) == {"host-walk", "host-rear", "no_behavior"}
    assert all(sid.startswith("DonorTS__") for sid in imported["segment_id"])
    # Source features landed in real columns, not NaN-filled.
    assert imported[FEATS[0]].notna().all()


def test_load_labels_falls_back_to_review_log_without_training_set(tmp_path: Path) -> None:
    """Sources with no assembled training set still import via the review log."""
    src = _make_source(tmp_path, FEATS, name="NoTS")  # only segment_features + log
    labels = ModelRefinementService()._load_labels(src)
    assert labels is not None
    assert len(labels) == 4  # the 4 reviewer-log rows
    assert set(labels.columns) >= {"segment_id", "review_label", "confidence"}


def _make_source_renamed(tmp_path: Path, feat_cols: list[str], name: str = "src") -> Path:
    """Source whose behaviour is named 'Head Dip' (host calls it 'Dip')."""
    root = tmp_path / name
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "src-dip", "name": "Head Dip"},
    ])
    rep = root / "derived" / "representations"
    rep.mkdir(parents=True, exist_ok=True)
    seg_ids = ["seg_s1", "seg_s2", "seg_s3"]
    feats = _feature_frame(seg_ids, feat_cols, ["ss1"] * 3)
    feats.to_parquet(rep / "segment_features.parquet", index=False)
    lab_dir = root / "derived" / "review_labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({
        "segment_id": seg_ids,
        "review_label": ["src-dip", "src-dip", "no_behavior"],
        "reviewer_id": ["r"] * 3,
        "confidence": [1.0] * 3,
        "timestamp": ["2026-01-01T00:00:0" + str(i) for i in range(3)],
    })
    labels.to_parquet(lab_dir / "reviewer_labels.parquet", index=False)
    return root


def _make_host_with_dip(tmp_path: Path, feat_cols: list[str]) -> Path:
    root = tmp_path / "host"
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": "host-dip", "name": "Dip"},
    ])
    ts_dir = root / "derived" / "training_sets"
    ts_dir.mkdir(parents=True, exist_ok=True)
    df = _feature_frame(["seg_h1", "seg_h2"], feat_cols, ["hs1", "hs1"])
    df["label"] = ["host-dip", "no_behavior"]
    df["label_source"] = "review"
    df["reviewer_confidence"] = 1.0
    df.to_parquet(ts_dir / "training_set.parquet", index=False)
    return root


def test_remap_alias_makes_behavior_importable(tmp_path: Path) -> None:
    host = _make_host_with_dip(tmp_path, FEATS)
    src = _make_source_renamed(tmp_path, FEATS)
    svc = ModelRefinementService()

    # Without an alias, 'Head Dip' does not match host 'Dip'; only
    # no_behavior (1) is importable, Head Dip (2) is skipped.
    pv = svc.preview(host, src, name_overrides={})
    assert pv.importable_labeled == 1
    assert any(m.source_name == "Head Dip" for m in pv.unmatched_behaviors)
    assert not any(m.remapped for m in pv.matched_behaviors)

    # With the alias, 'Head Dip' -> 'Dip' becomes importable.
    overrides = {"head dip": "Dip"}
    pv2 = svc.preview(host, src, name_overrides=overrides)
    assert pv2.compatible, pv2.reason
    assert pv2.importable_labeled == 3
    remapped = [m for m in pv2.matched_behaviors if m.remapped]
    assert any(m.host_name == "Dip" for m in remapped)


def test_import_applies_alias(tmp_path: Path) -> None:
    host = _make_host_with_dip(tmp_path, FEATS)
    src = _make_source_renamed(tmp_path, FEATS, name="DonorB")
    svc = ModelRefinementService()
    result = svc.import_examples(host, src, name_overrides={"head dip": "Dip"})
    assert result["status"] == "success", result.get("error")
    assert result["imported_rows"] == 3
    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    imported = ts[ts["label_source"] == "imported:DonorB"]
    # Head Dip rows were remapped onto the host 'Dip' behaviour id.
    assert set(imported["label"]) == {"host-dip", "no_behavior"}


def test_aliases_persist_and_autoload(tmp_path: Path) -> None:
    host = _make_host_with_dip(tmp_path, FEATS)
    src = _make_source_renamed(tmp_path, FEATS)
    svc = ModelRefinementService()
    svc.save_aliases(host, {"Head Dip": "Dip"})
    assert svc.load_aliases(host) == {"head dip": "Dip"}
    # preview with no explicit overrides should pick up the saved table.
    pv = svc.preview(host, src)
    assert pv.compatible
    assert pv.importable_labeled == 3


def test_suggest_host_match() -> None:
    svc = ModelRefinementService()
    host_names = ["Dip", "Rear", "Groom"]
    assert svc.suggest_host_match("Head Dip", host_names) == "Dip"
    assert svc.suggest_host_match("rearing", host_names) == "Rear"
    assert svc.suggest_host_match("xyzzy", host_names) == ""


def test_list_host_behaviors_excludes_no_behavior(tmp_path: Path) -> None:
    host = _make_host_with_dip(tmp_path, FEATS)
    names = [n for _, n in ModelRefinementService().list_host_behaviors(host)]
    assert names == ["Dip"]


# Two projects that track the SAME keypoints under different names: the host
# calls them center_body/left_ear, the source back_mid/ear_left.  Every derived
# column (per-keypoint kinematics + pairwise distances) is therefore named
# differently, even though the schemes are identical.
_HOST_KP_FEATS = [
    "dist_center_body_to_left_ear", "dist_center_body_to_nose",
    "center_body_speed", "left_ear_speed", "nose_speed",
]
_SRC_KP_FEATS = [
    "dist_back_mid_to_ear_left", "dist_back_mid_to_nose",
    "back_mid_speed", "ear_left_speed", "nose_speed",
]


def test_preview_remaps_keypoint_scheme(tmp_path: Path) -> None:
    host = _make_host(tmp_path, _HOST_KP_FEATS)
    src = _make_source(tmp_path, _SRC_KP_FEATS)
    pv = ModelRefinementService().preview(host, src)
    # Differently-named-but-identical keypoint schemes line up after remap.
    assert pv.compatible, pv.reason
    assert pv.coverage == 1.0
    assert pv.keypoint_renames == {"back_mid": "center_body", "ear_left": "left_ear"}


def test_import_remaps_keypoint_scheme(tmp_path: Path) -> None:
    host = _make_host(tmp_path, _HOST_KP_FEATS)
    src = _make_source(tmp_path, _SRC_KP_FEATS, name="DonorKP")
    result = ModelRefinementService().import_examples(host, src)
    assert result["status"] == "success", result.get("error")
    assert result["imported_rows"] == 3
    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    imported = ts[ts["label_source"] == "imported:DonorKP"]
    # Source features landed in the host-named columns (not NaN-filled), proving
    # the columns were renamed onto the host scheme rather than treated missing.
    assert imported["center_body_speed"].notna().all()
    assert imported["dist_center_body_to_left_ear"].notna().all()


def _write_registry(root: Path, ppm: float, net: str) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    entries = {
        f"s{i}": {"pixels_per_mm": ppm, "pose_filename": f"SUBJ{i}{net}_snapshot_100.csv"}
        for i in range(3)
    }
    (cfg / "session_registry.json").write_text(
        json.dumps({"schema_version": "0.2.0", "entries": entries}), encoding="utf-8",
    )


def _write_experiment(root: Path, window: int = 15) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "experiment.yaml").write_text(yaml.dump({"behavior_model": {
        "segment_window_frames": window,
        "segment_stride_frames": 15,
        "use_video_features": False,
        "invariant_features": {
            "enable_egocentric_kinematics": True,
            "enable_body_length_normalization": True,
            "enable_relative_geometry": True,
            "enable_head_direction": True,
            "enable_joint_angles": True,
            "enable_spine_curvature": True,
        },
    }}), encoding="utf-8")


def _write_project_yaml(
    root: Path, use_video: bool, window: int = 15, stride: int = 15,
) -> None:
    """Live project.yaml: the checkbox writes use_video to feature_extraction."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.yaml").write_text(yaml.dump({
        "behavior_model": {
            "segment_window_frames": window,
            "segment_stride_frames": stride,
            "use_video_features": use_video,
        },
        "feature_extraction": {"use_video_features": use_video},
    }), encoding="utf-8")


def test_diagnostics_use_video_reads_project_yaml_over_stale_experiment(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="Donor")
    for root in (host, src):
        _write_registry(root, 0.80, "DLC_Resnet50_Sameshuffle1")
        _write_experiment(root, window=15)        # stale: use_video_features False
        _write_project_yaml(root, use_video=True)  # live checkbox: True for both
    d = ModelRefinementService().preview(host, src).diagnostics
    # No false "True vs False": both resolve to True from the live project.yaml.
    assert not any("use_video_features" in m for m in d.config_mismatches)


def test_diagnostics_use_video_mismatch_taken_from_project_yaml(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="Donor2")
    for root in (host, src):
        _write_registry(root, 0.80, "DLC_Resnet50_Sameshuffle1")
        _write_experiment(root, window=15)  # both stale-False; the real values...
    _write_project_yaml(host, use_video=True)   # ...differ in project.yaml
    _write_project_yaml(src, use_video=False)
    d = ModelRefinementService().preview(host, src).diagnostics
    assert any("use_video_features" in m for m in d.config_mismatches)


def test_diagnostics_flags_value_level_differences(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="Donor")
    _write_registry(host, 0.72, "DLC_Resnet50_Ashuffle1")
    _write_registry(src, 0.82, "DLC_Resnet50_Bshuffle2")
    _write_experiment(host, window=15)
    _write_experiment(src, window=30)  # extraction-setting mismatch

    d = ModelRefinementService().preview(host, src).diagnostics
    assert d is not None
    # Calibration difference is surfaced.
    assert d.px_per_mm_pct_diff is not None and d.px_per_mm_pct_diff > 10
    # Different DLC networks are detected.
    assert not d.pose_models_match
    assert d.host_pose_models == ["DLC_Resnet50_Ashuffle1"]
    # Extraction-config mismatch is reported.
    assert any("segment_window_frames" in m for m in d.config_mismatches)


def test_diagnostics_clean_when_projects_match(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="Twin")
    for root in (host, src):
        _write_registry(root, 0.80, "DLC_Resnet50_Sameshuffle1")
        _write_experiment(root, window=15)
    d = ModelRefinementService().preview(host, src).diagnostics
    assert d.pose_models_match
    assert d.config_mismatches == []
    assert d.px_per_mm_pct_diff == 0


def _big_project(
    root: Path, feat_cols: list[str], n: int, *, is_host: bool, shift: float, seed: int,
) -> None:
    """A project with ``n`` rows of Gaussian features (mean per column offset by
    ``shift``) so feature-distribution shift is statistically measurable."""
    import numpy as np  # local import keeps the rest of the suite numpy-free
    _write_behaviors(root, [
        {"behavior_id": "no_behavior", "name": "No Behavior"},
        {"behavior_id": ("host-walk" if is_host else "src-walk"), "name": "Walk"},
    ])
    rng = np.random.default_rng(seed)
    data: dict = {
        "segment_id": [f"seg{i}" for i in range(n)],
        "session_id": ["s"] * n,
        "animal_id": ["a"] * n,
        "start_frame": list(range(n)),
        "end_frame": [i + 14 for i in range(n)],
    }
    for j, c in enumerate(feat_cols):
        data[c] = rng.normal(j + shift, 1.0, n)
    df = pd.DataFrame(data)
    labels = ["no_behavior" if i % 2 else None for i in range(n)]
    if is_host:
        ts = root / "derived" / "training_sets"
        ts.mkdir(parents=True, exist_ok=True)
        df["label"] = ["host-walk" if i % 2 else "no_behavior" for i in range(n)]
        df["label_source"] = "review"
        df["reviewer_confidence"] = 1.0
        df.to_parquet(ts / "training_set.parquet", index=False)
    else:
        rep = root / "derived" / "representations"
        rep.mkdir(parents=True, exist_ok=True)
        df.to_parquet(rep / "segment_features.parquet", index=False)
        lab = root / "derived" / "review_labels"
        lab.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "segment_id": data["segment_id"],
            "review_label": ["src-walk" if i % 2 else "no_behavior" for i in range(n)],
            "reviewer_id": ["r"] * n,
            "confidence": [1.0] * n,
            "timestamp": [f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}" for i in range(n)],
        }).to_parquet(lab / "reviewer_labels.parquet", index=False)


def test_diagnostics_feature_shift_vs_baseline(tmp_path: Path) -> None:
    feats = [f"feat_{i}" for i in range(25)]
    host = tmp_path / "host"
    src = tmp_path / "src"
    # Source feature means are offset by 1.0 (~1 IQR), host is its own reference.
    _big_project(host, feats, 200, is_host=True, shift=0.0, seed=1)
    _big_project(src, feats, 200, is_host=False, shift=1.0, seed=2)

    d = ModelRefinementService().preview(host, src).diagnostics
    assert d.feature_shift_median is not None
    assert d.within_host_shift_median is not None
    # Within-project shift is ~0 (sampling noise); cross-project shift is large.
    assert d.within_host_shift_median < 0.1
    assert d.feature_shift_median > d.within_host_shift_median
    assert d.feature_shift_median > 0.3


def test_diagnostics_skipped_for_import_path(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS)
    pv = ModelRefinementService().preview(host, src, compute_diagnostics=False)
    assert pv.diagnostics is None


def _write_source_clips(root: Path, seg_ids: list[str], session: str = "ss1") -> None:
    cdir = root / "derived" / "clips" / session
    cdir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(seg_ids):
        (cdir / f"{s}_{i:06x}ab.mp4").write_bytes(b"FAKEMP4")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_import_registers_reviewed_clips_with_source(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="DonorR")
    seg_ids = ["seg_s1", "seg_s2", "seg_s3", "seg_s4"]  # s4 = Jump (no host match)
    _write_source_clips(src, seg_ids)

    result = ModelRefinementService().import_examples(host, src)
    assert result["status"] == "success", result.get("error")
    # Walk (2) + no_behavior (1) register; Jump (1) is skipped.
    assert result["review_registered"] == 3

    tables = host / "derived" / "review_tables"
    cands = _read_json(tables / "external_window_candidates.json")["candidates"]
    assert len(cands) == 3
    assert all(c["source"] == "DonorR" for c in cands)
    assert all(c["clip_path"] for c in cands)  # clips were copied

    decisions = _read_json(tables / "review_decisions.json")["decisions"]
    assert len(decisions) == 3
    assert all(d["new_status"] == "reviewed" and d["decision"] == "accept" for d in decisions)
    # The candidate<->decision join key lines up (window_id == clip_id).
    assert {c["window_id"] for c in cands} == {d["clip_id"] for d in decisions}

    # Clips physically copied into this project (Jump's clip is not).
    copied = list((host / "derived" / "clips").rglob("*.mp4"))
    assert len(copied) == 3


def test_review_registration_is_idempotent(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="DonorR")
    _write_source_clips(src, ["seg_s1", "seg_s2", "seg_s3", "seg_s4"])
    svc = ModelRefinementService()
    svc.import_examples(host, src)
    # Re-register (e.g. backfill) must not duplicate candidates or decisions.
    out = svc.register_imported_for_review(host, src)
    assert out["status"] == "success"
    tables = host / "derived" / "review_tables"
    cands = _read_json(tables / "external_window_candidates.json")["candidates"]
    decisions = _read_json(tables / "review_decisions.json")["decisions"]
    assert len(cands) == 3
    assert len([d for d in decisions if d["clip_id"].startswith("DonorR__")]) == 3


def test_import_records_manifest_and_lists(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="DonorM")
    svc = ModelRefinementService()
    svc.import_examples(host, src)

    imports = svc.list_imports(host)
    assert len(imports) == 1
    rec = imports[0]
    assert rec.tag == "DonorM"
    assert rec.imported_rows == 3
    assert rec.source_root == str(src)
    assert rec.behaviors  # host behaviour name -> count


def test_remove_import_cleans_everything(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, FEATS, name="DonorX")
    _write_source_clips(src, ["seg_s1", "seg_s2", "seg_s3", "seg_s4"])
    svc = ModelRefinementService()
    svc.import_examples(host, src)

    ts_path = host / "derived" / "training_sets" / "training_set.parquet"
    before = pd.read_parquet(ts_path)
    assert (before["label_source"].astype(str) == "imported:DonorX").sum() == 3

    res = svc.remove_import(host, "DonorX")
    assert res["status"] == "success"
    assert res["removed_rows"] == 3
    assert res["removed_clips"] == 3
    assert res["removed_decisions"] == 3

    # Training rows gone, host's own rows untouched.
    after = pd.read_parquet(ts_path)
    assert (after["label_source"].astype(str) == "imported:DonorX").sum() == 0
    assert len(after) == len(before) - 3
    # Manifest, external candidates, decisions, and copied clips all cleared.
    assert svc.list_imports(host) == []
    tables = host / "derived" / "review_tables"
    assert _read_json(tables / "external_window_candidates.json")["candidates"] == []
    assert _read_json(tables / "review_decisions.json")["decisions"] == []
    assert not list((host / "derived" / "clips").glob("DonorX__*"))


def test_remove_import_leaves_other_sources(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    a = _make_source(tmp_path, FEATS, name="SrcA")
    b = _make_source(tmp_path, FEATS, name="SrcB")
    svc = ModelRefinementService()
    svc.import_examples(host, a)
    svc.import_examples(host, b)
    svc.remove_import(host, "SrcA")

    tags = {r.tag for r in svc.list_imports(host)}
    assert tags == {"SrcB"}
    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    assert (ts["label_source"].astype(str) == "imported:SrcA").sum() == 0
    assert (ts["label_source"].astype(str) == "imported:SrcB").sum() == 3


def test_import_blocked_when_incompatible(tmp_path: Path) -> None:
    host = _make_host(tmp_path, FEATS)
    src = _make_source(tmp_path, ["feat_0"] + [f"x_{i}" for i in range(19)])
    result = ModelRefinementService().import_examples(host, src)
    assert result["status"] == "error"
    assert "Incompatible" in result["error"]
    # Host training set is untouched.
    ts = pd.read_parquet(host / "derived" / "training_sets" / "training_set.parquet")
    assert len(ts) == 2
