"""Tests for cross-session feature coverage auditing.

The failure this guards against is a column that pools to a healthy-looking
feature while being entirely absent for a subset of sessions — those sessions
then get scored off a constant input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abel.services.feature_coverage_service import (
    CoverageSplit,
    audit_session_coverage,
    format_coverage_report,
)
from abel.temporal_refinement.temporal_refinement_service import (
    TemporalRefinementService,
)

RNG = np.random.default_rng(0)


def _session(n=50, **cols) -> pd.DataFrame:
    return pd.DataFrame({k: np.asarray(v) for k, v in cols.items()}, index=range(n))


def _varying(n=50):
    return RNG.normal(size=n)


# ── the audit ───────────────────────────────────────────────────────────────

def test_no_split_when_every_session_is_populated():
    frames = {
        "s1": _session(speed=_varying(), roi_dist=_varying()),
        "s2": _session(speed=_varying(), roi_dist=_varying()),
    }
    assert audit_session_coverage(frames) == []


def test_all_nan_in_one_session_is_reported():
    frames = {
        "s1": _session(speed=_varying(), nose_to_target_dist=_varying()),
        "s2": _session(speed=_varying(), nose_to_target_dist=np.full(50, np.nan)),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert splits[0].columns == ["nose_to_target_dist"]
    assert splits[0].sessions_absent == ["s2"]
    assert splits[0].sessions_present == ["s1"]
    assert splits[0].reason == "all-NaN"
    assert splits[0].family == "ROI / target zone"


def test_zero_filled_session_is_reported_as_constant():
    """A partial R3D backfill zero-fills rather than NaN-fills, by whole family."""
    real = {f"r3d_{i:03d}": _varying() for i in range(12)}
    filled = {f"r3d_{i:03d}": np.zeros(50) for i in range(12)}
    frames = {"s1": _session(**real), "s2": _session(**filled)}
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert splits[0].reason == "constant"
    assert splits[0].family == "R3D video embedding"
    assert splits[0].sessions_absent == ["s2"]
    assert len(splits[0].columns) == 12


def test_isolated_constant_column_is_not_reported():
    """An occupancy flag for a zone one animal never entered goes flat alone."""
    frames = {
        "s1": _session(speed=_varying(), in_roi_1_nose=_varying()),
        "s2": _session(speed=_varying(), in_roi_1_nose=np.zeros(50)),
    }
    assert audit_session_coverage(frames) == []


def test_constant_group_threshold_is_tunable():
    frames = {
        "s1": _session(speed=_varying(), roi=_varying()),
        "s2": _session(speed=_varying(), roi=np.zeros(50)),
    }
    assert audit_session_coverage(frames) == []
    assert len(audit_session_coverage(frames, min_constant_group=1)) == 1


def test_all_nan_needs_no_group_threshold():
    """Nothing legitimately blanks a single column for one session only."""
    frames = {
        "s1": _session(speed=_varying(), roi=_varying()),
        "s2": _session(speed=_varying(), roi=np.full(50, np.nan)),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert splits[0].reason == "all-NaN"


def test_zero_fill_caught_across_several_absent_sessions():
    real = {f"r3d_{i:03d}": _varying() for i in range(12)}
    filled = {f"r3d_{i:03d}": np.zeros(50) for i in range(12)}
    frames = {
        "s1": _session(**real),
        "s2": _session(**filled),
        "s3": _session(**filled),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert splits[0].sessions_absent == ["s2", "s3"]


def test_uniformly_dead_column_is_not_reported():
    """A column dead everywhere is the pooled audit's job, not this one."""
    frames = {
        "s1": _session(speed=_varying(), dead=np.full(50, np.nan)),
        "s2": _session(speed=_varying(), dead=np.full(50, np.nan)),
    }
    assert audit_session_coverage(frames) == []


def test_uniformly_constant_column_is_not_reported():
    frames = {
        "s1": _session(speed=_varying(), flat=np.ones(50)),
        "s2": _session(speed=_varying(), flat=np.ones(50)),
    }
    assert audit_session_coverage(frames) == []


def test_columns_missing_together_collapse_into_one_finding():
    """334 ROI columns absent for the same 16 sessions must read as 1 finding."""
    roi_cols = [f"nose_to_roi_1_dist_{i}" for i in range(30)]
    present = {c: _varying() for c in roi_cols}
    absent = {c: np.full(50, np.nan) for c in roi_cols}
    frames = {
        "s1": _session(speed=_varying(), **present),
        "s2": _session(speed=_varying(), **present),
        "s3": _session(speed=_varying(), **absent),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert len(splits[0].columns) == 30
    assert splits[0].sessions_absent == ["s3"]


def test_different_absent_sets_stay_separate_findings():
    frames = {
        "s1": _session(a=_varying(), b=_varying()),
        "s2": _session(a=np.full(50, np.nan), b=_varying()),
        "s3": _session(a=_varying(), b=np.full(50, np.nan)),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 2
    assert {tuple(s.sessions_absent) for s in splits} == {("s2",), ("s3",)}


def test_columns_filter_restricts_the_audit():
    frames = {
        "s1": _session(used=_varying(), unused=_varying()),
        "s2": _session(used=_varying(), unused=np.full(50, np.nan)),
    }
    assert audit_session_coverage(frames, columns=["used"]) == []
    assert len(audit_session_coverage(frames, columns=["used", "unused"])) == 1


def test_single_session_has_nothing_to_compare_against():
    frames = {"s1": _session(a=np.full(50, np.nan))}
    assert audit_session_coverage(frames) == []


def test_missing_column_in_one_session_counts_as_absent():
    """A session whose frame table lacks the column entirely."""
    frames = {
        "s1": _session(speed=_varying(), extra=_varying()),
        "s2": _session(speed=_varying()),
    }
    splits = audit_session_coverage(frames, columns=["speed", "extra"])
    assert len(splits) == 1
    assert splits[0].columns == ["extra"]
    assert splits[0].sessions_absent == ["s2"]


def test_empty_session_frame_is_absent_everywhere():
    frames = {
        "s1": _session(speed=_varying()),
        "s2": pd.DataFrame({"speed": []}),
    }
    splits = audit_session_coverage(frames, columns=["speed"])
    assert len(splits) == 1
    assert splits[0].sessions_absent == ["s2"]


def test_identifier_columns_are_never_audited():
    frames = {
        "s1": _session(speed=_varying(), animal_id=np.arange(50)),
        "s2": _session(speed=_varying(), animal_id=np.full(50, np.nan)),
    }
    assert audit_session_coverage(frames) == []


def test_string_columns_are_skipped_not_fatal():
    """Real feature tables carry provenance/label columns the name list misses."""
    frames = {
        "s1": _session(speed=_varying(), source=np.array(["s1"] * 50)),
        "s2": _session(speed=_varying(), source=np.array(["s2"] * 50)),
    }
    assert audit_session_coverage(frames) == []


def test_string_column_does_not_mask_a_real_split():
    frames = {
        "s1": _session(roi=_varying(), source=np.array(["s1"] * 50)),
        "s2": _session(roi=np.full(50, np.nan), source=np.array(["s2"] * 50)),
    }
    splits = audit_session_coverage(frames)
    assert len(splits) == 1
    assert splits[0].columns == ["roi"]


def test_findings_are_ordered_by_blast_radius():
    wide = {f"w{i}": _varying() for i in range(5)}
    wide_nan = {f"w{i}": np.full(50, np.nan) for i in range(5)}
    frames = {
        "s1": _session(narrow=_varying(), **wide),
        "s2": _session(narrow=np.full(50, np.nan), **wide),
        "s3": _session(narrow=_varying(), **wide_nan),
    }
    splits = audit_session_coverage(frames)
    assert len(splits[0].columns) == 5  # the wider split leads
    assert splits[1].columns == ["narrow"]


def test_report_is_rendered_without_dumping_every_column():
    split = CoverageSplit(
        family="ROI / target zone",
        columns=[f"c{i}" for i in range(334)],
        sessions_present=["s1"],
        sessions_absent=[f"n{i}" for i in range(16)],
    )
    text = format_coverage_report([split])
    assert "334 ROI / target zone feature(s)" in text
    assert "+331 more" in text
    assert "+8 more" in text
    assert "c333" not in text


# ── the preflight ───────────────────────────────────────────────────────────

def _service() -> TemporalRefinementService:
    return TemporalRefinementService.__new__(TemporalRefinementService)


def test_preflight_raises_on_a_split_in_a_model_feature():
    frames = {
        "s1": _session(speed=_varying(), nose_to_target_dist=_varying()),
        "s2": _session(speed=_varying(), nose_to_target_dist=np.full(50, np.nan)),
    }
    payloads = {"bid": {"feature_cols": ["speed", "nose_to_target_dist"]}}
    with pytest.raises(ValueError) as err:
        _service()._preflight_feature_coverage(payloads, ["s1", "s2"], frames, None)
    assert "Feature coverage is inconsistent" in str(err.value)
    assert "nose_to_target_dist" in str(err.value)


def test_preflight_ignores_splits_in_features_no_model_reads():
    frames = {
        "s1": _session(speed=_varying(), unused=_varying()),
        "s2": _session(speed=_varying(), unused=np.full(50, np.nan)),
    }
    payloads = {"bid": {"feature_cols": ["speed"]}}
    _service()._preflight_feature_coverage(payloads, ["s1", "s2"], frames, None)


def test_preflight_ignores_sessions_not_selected():
    frames = {
        "s1": _session(speed=_varying(), roi=_varying()),
        "s2": _session(speed=_varying(), roi=_varying()),
        "s3": _session(speed=_varying(), roi=np.full(50, np.nan)),
    }
    payloads = {"bid": {"feature_cols": ["speed", "roi"]}}
    _service()._preflight_feature_coverage(payloads, ["s1", "s2"], frames, None)
    with pytest.raises(ValueError):
        _service()._preflight_feature_coverage(
            payloads, ["s1", "s2", "s3"], frames, None
        )


def test_preflight_passes_when_models_declare_no_features():
    _service()._preflight_feature_coverage({"bid": {}}, ["s1"], {}, None)
