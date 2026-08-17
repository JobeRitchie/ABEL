"""Cross-session feature coverage auditing.

:class:`~abel.services.feature_audit_service.FeatureAuditService` judges a
column by pooling every row in the project, which answers "is this feature
alive?" but not "is it alive *everywhere*?".  Those differ in the case that
matters most: a column populated for some sessions and all-NaN (or all-zero)
for the rest pools to a perfectly healthy-looking feature, while the sessions
missing it get scored off a constant input.

That has bitten this project repeatedly and always the same way — an ROI drawn
for one cohort but not another, an R3D backfill that zero-filled the sessions it
could not decode, two spellings of a distance column splitting one feature into
two half-populated ones.  In every case the model still returns a probability,
so the failure surfaces as a suspiciously flat or suspiciously eager trace
rather than an error.

This module compares sessions against each other and reports the columns whose
availability *splits*, grouped so one missing feature family is one finding
instead of several hundred.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from abel.services.r3d_feature_service import is_r3d_column
from abel.services.roi_service import is_roi_column

# Per-session rows sampled when deciding whether a column is populated.  All-NaN
# and all-constant are properties a sample detects reliably, and a full scan of
# every frame × ~1900 columns would cost more than the check is worth.
_SAMPLE_ROWS = 2000

# How many columns must go constant together before that counts as a fill rather
# than a rare event.  Fills arrive by whole feature family — 512 R3D dimensions,
# hundreds of ROI columns — so anything in the low tens separates the two cleanly.
_MIN_CONSTANT_GROUP = 8

_NON_FEATURE_COLS = frozenset({
    "segment_id", "start_frame", "end_frame", "frame", "animal_id",
    "session_id", "subject_id", "label", "behavior_id", "chunk_index",
})


def _family_of(col: str) -> str:
    if is_r3d_column(col):
        return "R3D video embedding"
    if is_roi_column(col):
        return "ROI / target zone"
    return "pose / kinematics"


@dataclass
class CoverageSplit:
    """One group of columns that is populated for some sessions and not others.

    Columns are grouped by the exact set of sessions missing them, so a whole
    feature family that drops out together reads as a single finding.
    """

    family: str
    columns: list[str]
    sessions_present: list[str] = field(default_factory=list)
    sessions_absent: list[str] = field(default_factory=list)
    reason: str = "all-NaN"  # or "constant"

    def describe(self, limit: int = 8) -> str:
        cols = ", ".join(self.columns[:3])
        if len(self.columns) > 3:
            cols += f", … (+{len(self.columns) - 3} more)"
        sess = ", ".join(self.sessions_absent[:limit])
        if len(self.sessions_absent) > limit:
            sess += f", … (+{len(self.sessions_absent) - limit} more)"
        return (
            f"{len(self.columns)} {self.family} feature(s) are {self.reason} for "
            f"{len(self.sessions_absent)} of "
            f"{len(self.sessions_absent) + len(self.sessions_present)} session(s) "
            f"but populated for the rest — e.g. {cols}.\n"
            f"    Affected sessions: {sess}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "reason": self.reason,
            "n_columns": len(self.columns),
            "columns": self.columns,
            "sessions_absent": self.sessions_absent,
            "sessions_present": self.sessions_present,
        }


def _session_column_state(
    df: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column ``(all_nan, constant)`` masks for one session."""
    if len(df) > _SAMPLE_ROWS:
        idx = np.linspace(0, len(df) - 1, _SAMPLE_ROWS).astype(int)
        df = df.iloc[idx]
    block = df.reindex(columns=cols).to_numpy(dtype="float64", na_value=np.nan)
    all_nan = np.all(np.isnan(block), axis=0)
    # nanstd over an all-NaN column is a documented NaN, not a problem; those
    # columns are already classified by all_nan.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(block, axis=0)
    constant = ~all_nan & (np.nan_to_num(spread, nan=0.0) < 1e-12)
    return all_nan, constant


def audit_session_coverage(
    frames_by_session: dict[str, pd.DataFrame],
    *,
    columns: Iterable[str] | None = None,
    min_constant_group: int = _MIN_CONSTANT_GROUP,
) -> list[CoverageSplit]:
    """Report feature columns whose availability splits across sessions.

    *frames_by_session* maps session id to that session's feature rows.  Pass
    *columns* to restrict the audit to the features a consumer actually uses —
    a column no model reads is not worth reporting on.

    A column is "absent" for a session when it is entirely NaN there, or
    entirely constant while varying elsewhere (the signature of a zero-fill).
    Only columns with both a present and an absent session are returned; a
    column missing everywhere is uniformly dead, which the pooled feature audit
    already covers.
    """
    sessions = sorted(frames_by_session)
    if len(sessions) < 2:
        return []  # Nothing to compare against.

    if columns is not None:
        cols = [c for c in dict.fromkeys(columns) if c not in _NON_FEATURE_COLS]
    else:
        seen: dict[str, None] = {}
        for df in frames_by_session.values():
            for c in df.columns:
                if c not in _NON_FEATURE_COLS:
                    seen.setdefault(c, None)
        cols = list(seen)

    # Feature tables carry identifier and provenance columns beyond the names
    # above (session labels, source tags, timestamps).  Coverage is only defined
    # for numeric features, and a string column would abort the whole audit at
    # the float conversion, so select by dtype rather than by name.
    numeric = {
        c for df in frames_by_session.values()
        for c in df.columns[[pd.api.types.is_numeric_dtype(d) for d in df.dtypes]]
    }
    cols = [c for c in cols if c in numeric]
    if not cols:
        return []

    nan_mask = np.zeros((len(sessions), len(cols)), dtype=bool)
    const_mask = np.zeros_like(nan_mask)
    for i, sid in enumerate(sessions):
        df = frames_by_session[sid]
        if df is None or len(df) == 0:
            nan_mask[i, :] = True
            continue
        nan_mask[i], const_mask[i] = _session_column_state(df, cols)

    # Group by (reason, exact set of absent sessions) so a family that drops out
    # together collapses into one finding.
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for j, col in enumerate(cols):
        for reason, mask in (("all-NaN", nan_mask), ("constant", const_mask)):
            absent = mask[:, j]
            if not absent.any():
                continue
            # "Present" means genuinely varying somewhere — a column that is
            # all-NaN in one session and constant in the rest never counts as
            # populated, and belongs to the uniformly-dead case.
            present = ~nan_mask[:, j] & ~const_mask[:, j]
            if not present.any():
                continue
            key = (reason, tuple(s for s, a in zip(sessions, absent) if a))
            groups.setdefault(key, []).append(col)

    # A rare-event feature — an occupancy flag for a zone the animal never
    # entered — is legitimately constant for a whole session, and on the z-scored
    # representation cache it is not distinguishable from a zero-fill by its
    # values alone.  What separates them is breadth: a fill takes out an entire
    # feature family at once (512 R3D dimensions, hundreds of ROI columns), while
    # a rare event goes flat in isolation.  All-NaN needs no such guard — nothing
    # legitimately blanks a column for one session only.
    groups = {
        key: cols_in_group for key, cols_in_group in groups.items()
        if key[0] != "constant" or len(cols_in_group) >= min_constant_group
    }

    splits = [
        CoverageSplit(
            family=_family_of(cols_in_group[0]),
            columns=sorted(cols_in_group),
            sessions_absent=list(absent_sids),
            sessions_present=[s for s in sessions if s not in set(absent_sids)],
            reason=reason,
        )
        for (reason, absent_sids), cols_in_group in groups.items()
    ]
    # Widest blast radius first: most columns, then most sessions affected.
    splits.sort(key=lambda s: (-len(s.columns), -len(s.sessions_absent)))
    return splits


def format_coverage_report(splits: list[CoverageSplit], limit: int = 5) -> str:
    """Render *splits* as an operator-facing message."""
    shown = splits[:limit]
    lines = [f"  - {s.describe()}" for s in shown]
    if len(splits) > limit:
        lines.append(f"  - … and {len(splits) - limit} further coverage split(s).")
    return "\n".join(lines)
