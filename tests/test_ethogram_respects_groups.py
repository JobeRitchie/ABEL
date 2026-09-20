"""The Graphs-tab ethogram must follow the Group-by selection.

It used to draw every checked subject regardless of the facet controls, so
pinning a factor to one level (or viewing "By Group") still showed sessions
from other groups.  The plot, its session checklist and both exports now share
one roster built from the Group-by selection.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from abel.ui.tabs.behavior_analytics_tab import (  # noqa: E402
    FACET_COMBINE,
    FACET_SPLIT,
    _GraphsWidget,
    facet_session_labels,
)

FPS = 30.0
LABEL_BY_SESSION = {"s1": "F1", "s2": "F2", "s3": "M1", "s4": "M2", "s5": "U1"}
FACTORS = {
    "F1": {"Sex": "Female", "Drug": "Saline"},
    "F2": {"Sex": "Female", "Drug": "Fentanyl"},
    "M1": {"Sex": "Male", "Drug": "Saline"},
    "M2": {"Sex": "Male", "Drug": "Fentanyl"},
    "U1": {},  # no factor levels assigned
}
REAR = "b_rear"


def _make_widget(controls: dict[str, str], mode: str, ticked: set[str] | None = None):
    stub = SimpleNamespace()
    for name in (
        "_export_sessions", "_ethogram_roster", "_ethogram_sessions", "_ethogram",
    ):
        setattr(stub, name, getattr(_GraphsWidget, name).__get__(stub))

    groups, _split = facet_session_labels(FACTORS, ["Sex", "Drug"], controls)
    raw = {REAR: pd.DataFrame([
        {"session_id": sid, "behavior": "Rear", "start_frame": 0, "end_frame": 29}
        for sid in LABEL_BY_SESSION
    ])}
    labels = list(LABEL_BY_SESSION.values())
    stub._host = SimpleNamespace(
        _facet_controls=dict(controls),
        _session_factors=FACTORS,
        _session_groups=groups,
        _raw_bouts=raw,
        _project_fps=lambda: FPS,
        _session_label_by_session=dict(LABEL_BY_SESSION),
        ordered_session_labels=lambda: sorted(labels),
        _ordered_group_list=lambda names, *a: sorted({n for n in names if n}),
        _summary_tab=SimpleNamespace(_checked_subjects=lambda: set(labels)),
        _behaviors=SimpleNamespace(behaviors=[
            SimpleNamespace(behavior_id=REAR, name="Rear", color="#ff0000"),
        ]),
        _selected_behavior_ids=lambda: {REAR},
    )
    stub._get_mode = lambda: mode
    stub._checked_groups = lambda: {g for g in groups.values() if g}
    stub._checked_ethogram_sessions = lambda: set(ticked or ())
    stub._get_first_n_bouts = lambda: 0
    stub._cutoff_frames_for_until_behavior = lambda: {}
    stub._get_data_range_seconds = lambda: (None, None)
    stub._ethogram_cache = None
    stub._ethogram_cache_key = None
    stub._figure = plt.figure()
    stub._apply = lambda ax, **kw: None
    stub._gs = lambda: {"legend_fontsize": 8, "legend_loc": "best"}
    return stub


def _drawn_sessions(w) -> list[str]:
    w._ethogram(pd.DataFrame())
    ax = w._figure.axes[0]
    return [t.get_text() for t in ax.get_yticklabels()]


def test_level_filter_applies_in_individual_view():
    w = _make_widget({"Sex": "Female", "Drug": FACET_COMBINE}, mode="individual")
    assert w._ethogram_roster() == ["F1", "F2"]
    assert _drawn_sessions(w) == ["F1", "F2"]


def test_level_filter_applies_in_group_view():
    w = _make_widget({"Sex": "Male", "Drug": FACET_SPLIT}, mode="group")
    assert w._ethogram_roster() == ["M2", "M1"]  # Fentanyl before Saline
    assert _drawn_sessions(w) == ["M2 (Fentanyl)", "M1 (Saline)"]


def test_group_view_drops_ungrouped_and_orders_by_group():
    w = _make_widget({"Sex": FACET_SPLIT, "Drug": FACET_COMBINE}, mode="group")
    assert w._ethogram_roster() == ["F1", "F2", "M1", "M2"]
    assert _drawn_sessions(w) == [
        "F1 (Female)", "F2 (Female)", "M1 (Male)", "M2 (Male)",
    ]


def test_no_filter_individual_view_keeps_everyone():
    w = _make_widget({"Sex": FACET_SPLIT, "Drug": FACET_COMBINE}, mode="individual")
    assert _drawn_sessions(w) == ["F1", "F2", "M1", "M2", "U1"]


def test_session_ticks_intersect_roster():
    w = _make_widget({"Sex": "Female", "Drug": FACET_COMBINE}, mode="individual",
                     ticked={"F2", "M1"})
    assert w._ethogram_sessions() == ["F2"]
