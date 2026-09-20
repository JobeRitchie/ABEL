"""Auto Hunter presets: portability, ranking direction and graceful absence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.clip_metrics_service import ClipMetricsService, rich_metric_id
from abel.services.hunter_presets import (
    PRESETS,
    SOCIAL_PRESET,
    fit_preset,
    preset_by_id,
)

_SOCIAL_COLS = [
    "social_min_keypoint_dist_nearest_norm_norm_median",
    "social_dist_centroid_to_centroid_nearest_norm_norm_median",
    "social_dist_nose_to_nose_nearest_norm_norm_median",
    "social_dist_nose_to_tail_base_nearest_norm_norm_median",
    "social_in_contact_mean",
    "social_bbox_overlap_nearest_mean",
    "social_facing_angle_nearest_median",
]


def _write_features(root, frame: pd.DataFrame) -> None:
    path = root / "derived" / "representations" / "segment_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _synthetic_pool(n: int = 600, scale: float = 1.0, offset: float = 0.0) -> pd.DataFrame:
    """A pool where the last 60 windows are unambiguously 'social'.

    ``scale``/``offset`` simulate a different rig: a project whose distances are
    recorded in other units entirely.  A portable preset must rank the same
    windows first either way.
    """
    rng = np.random.default_rng(0)
    far = rng.normal(3.0, 0.5, n)
    out = pd.DataFrame(
        {
            "segment_id": [f"w{i:04d}" for i in range(n)],
            "social_min_keypoint_dist_nearest_norm_norm_median": far,
            "social_dist_centroid_to_centroid_nearest_norm_norm_median": far + 1.0,
            "social_dist_nose_to_nose_nearest_norm_norm_median": far + 0.5,
            "social_dist_nose_to_tail_base_nearest_norm_norm_median": far + 0.5,
            "social_in_contact_mean": np.zeros(n),
            "social_bbox_overlap_nearest_mean": np.zeros(n),
            "social_facing_angle_nearest_median": rng.uniform(1.0, 3.0, n),
        }
    )
    soc = slice(n - 60, n)
    for col in _SOCIAL_COLS[:4]:
        out.loc[soc, col] = rng.normal(0.3, 0.05, 60)
    out.loc[soc, "social_in_contact_mean"] = rng.uniform(0.6, 1.0, 60)
    out.loc[soc, "social_bbox_overlap_nearest_mean"] = rng.uniform(0.3, 0.7, 60)
    out.loc[soc, "social_facing_angle_nearest_median"] = rng.uniform(0.0, 0.4, 60)
    for col in _SOCIAL_COLS:
        if col.startswith("social_dist") or col.startswith("social_min"):
            out[col] = out[col] * scale + offset
    return out


def _service(tmp_path, frame: pd.DataFrame) -> ClipMetricsService:
    _write_features(tmp_path, frame)
    svc = ClipMetricsService()
    svc.set_project(tmp_path)
    return svc


def test_social_preset_is_registered():
    assert preset_by_id("social") is SOCIAL_PRESET
    assert SOCIAL_PRESET in PRESETS


def test_fit_ranks_social_windows_first(tmp_path):
    frame = _synthetic_pool()
    svc = _service(tmp_path, frame)
    fit = fit_preset(svc, SOCIAL_PRESET)
    assert fit.usable()
    assert len(fit.used) == len(_SOCIAL_COLS)
    assert not fit.missing

    pool = svc.load_rich_features(metric_ids=fit.used)
    scores = fit.scorer.score(pool)
    top = set(scores.sort_values(ascending=False).head(60).index)
    truly_social = {f"w{i:04d}" for i in range(540, 600)}
    # The ranker is what the hunt actually uses, so it has to put the planted
    # social windows at the top rather than merely inside the descriptive box.
    assert len(top & truly_social) >= 55


def test_ranking_survives_a_change_of_units(tmp_path):
    """The same windows rank first in a rig recorded on a different scale."""
    base = _synthetic_pool()
    rescaled = _synthetic_pool(scale=12.5, offset=400.0)

    def _top(frame, sub):
        svc = _service(sub, frame)
        fit = fit_preset(svc, SOCIAL_PRESET)
        pool = svc.load_rich_features(metric_ids=fit.used)
        return list(fit.scorer.score(pool).sort_values(ascending=False).head(60).index)

    assert _top(base, tmp_path / "a") == _top(rescaled, tmp_path / "b")


def test_descriptive_box_widens_with_breadth(tmp_path):
    svc = _service(tmp_path, _synthetic_pool())
    tight = fit_preset(svc, SOCIAL_PRESET, breadth=0.25)
    broad = fit_preset(svc, SOCIAL_PRESET, breadth=0.65)
    assert tight.criteria and len(tight.criteria) == len(broad.criteria)
    by_id = {c.metric_id: c for c in tight.criteria}
    for c in broad.criteria:
        t = by_id[c.metric_id]
        if c.high is not None:
            assert c.high >= t.high
        else:
            assert c.low <= t.low


def test_missing_social_features_is_reported_not_crashed(tmp_path):
    """A single-animal project has no social columns: say so, don't half-run."""
    frame = pd.DataFrame(
        {"segment_id": ["w0", "w1"], "nose_speed_mean": [1.0, 2.0]}
    )
    svc = _service(tmp_path, frame)
    fit = fit_preset(svc, SOCIAL_PRESET)
    assert not fit.usable()
    assert fit.scorer is None
    assert "social" in fit.note.lower()


def test_no_feature_table_is_reported(tmp_path):
    svc = ClipMetricsService()
    svc.set_project(tmp_path)
    fit = fit_preset(svc, SOCIAL_PRESET)
    assert not fit.usable()
    assert "Feature Extraction" in fit.note


def test_alternate_norm_spelling_still_resolves(tmp_path):
    """The pre-``_norm_norm`` schema spelling must not silently drop features."""
    frame = _synthetic_pool().rename(
        columns={c: c.replace("_norm_norm_", "_norm_") for c in _SOCIAL_COLS}
    )
    svc = _service(tmp_path, frame)
    fit = fit_preset(svc, SOCIAL_PRESET)
    assert fit.usable()
    assert not fit.missing
    assert rich_metric_id("social_min_keypoint_dist_nearest_norm_median") in fit.used


def test_scorer_tolerates_a_missing_column(tmp_path):
    svc = _service(tmp_path, _synthetic_pool())
    fit = fit_preset(svc, SOCIAL_PRESET)
    pool = svc.load_rich_features(metric_ids=fit.used)
    trimmed = pool.drop(columns=[fit.used[0]])
    scores = fit.scorer.score(trimmed)
    assert len(scores) == len(pool)
    assert np.isfinite(scores.to_numpy()).all()


@pytest.mark.parametrize("preset", PRESETS, ids=[p.id for p in PRESETS])
def test_every_preset_is_well_formed(preset):
    assert preset.features and preset.min_features <= len(preset.features)
    assert preset.name and preset.tagline and preset.description and preset.requires
    for f in preset.features:
        assert f.columns and all(isinstance(c, str) for c in f.columns)
        assert f.weight != 0.0
        assert f.box in (None, "low", "high")


# --- dialog wiring -----------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.ui.clip_mining_dialog import ClipMiningDialog  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - headless/Qt unavailable
        pytest.skip(f"Qt unavailable: {exc}")
    yield app


def _dialog(tmp_path, frame):
    _write_features(tmp_path, frame)
    return ClipMiningDialog(tmp_path, lambda: [], "test scope", lambda refs, sc: None)


def test_dialog_has_both_tabs(tmp_path, _app):
    dlg = _dialog(tmp_path, _synthetic_pool())
    assert [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())] == [
        "Criteria",
        "Auto Hunter",
    ]


def test_arming_a_preset_fills_the_criteria_rows(tmp_path, _app):
    dlg = _dialog(tmp_path, _synthetic_pool())
    dlg._arm_preset()
    # The preset's definition has to be visible and editable, not hidden in the
    # ranker: every boxed feature comes back as a criterion row.
    assert dlg._rows
    assert all(r.metric.currentData().startswith("feat:social_") for r in dlg._rows)
    assert dlg._essence_scorer is not None
    assert dlg._prefer_ranked is True
    assert "Social interaction" in dlg._preset_status.text()


def test_choosing_a_hard_filter_overrides_the_preset_preference(tmp_path, _app):
    dlg = _dialog(tmp_path, _synthetic_pool())
    dlg._arm_preset()
    dlg._match_all.click()
    assert dlg._prefer_ranked is False
    # A later re-grade must leave the user's choice alone.
    dlg._refresh_rank_scores()
    assert not dlg._match_ranked.isChecked()


def test_preset_hunt_ranks_without_a_pose_pass(tmp_path, _app):
    """Preset features are all precomputed, so mining never touches the pose drive."""
    dlg = _dialog(tmp_path, _synthetic_pool())
    dlg._arm_preset()
    assert dlg._feature_only_criteria() is True
