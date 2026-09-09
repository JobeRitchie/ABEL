"""Many-section presets must not freeze the Session Sections chart.

The LPT presets define 101 sections (a 3-minute baseline plus 50 tone/ITI
trials).  The bar and line draw routines used to read each value with a pair of
full-column comparisons per (session, section) and to add one Rectangle patch
per bar, so a 101-section preset across 80 sessions spent ~25 s on the UI thread
per redraw — long enough for Windows to mark the window unresponsive.

The value lookup is now one pivot per behaviour and the individual-mode bars are
one PolyCollection per session.  These tests pin both: the pivot must return
exactly what the per-section scan returned, and the collection must produce the
same geometry, limits and legend entry as ``ax.bar``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from abel.ui.tabs.behavior_analytics_tab import (  # noqa: E402
    _SessionSectionsWidget,
)


def _section_df(sessions: int, sections: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        for idx, name in enumerate(sections):
            rows.append({
                "session_label": f"m{s}",
                "section_name": name,
                "section_idx": idx,
                "n_bouts": int(rng.integers(0, 5)),
                "duration_s": float(rng.random() * 10),
            })
    return pd.DataFrame(rows)


def _naive_individual(df, metric, section_names, session_label):
    """The per-section scan the draw routines used before the pivot."""
    sess_df = df[df["session_label"] == session_label]
    return np.array([
        float(sess_df[sess_df["section_name"] == sn][metric].sum())
        if sn in sess_df["section_name"].values else 0.0
        for sn in section_names
    ])


def _naive_group(df, metric, section_name):
    return (
        df[df["section_name"] == section_name]
        .groupby("session_label")[metric]
        .sum()
        .to_numpy(float)
    )


class TestSectionPivot:
    """The pivot must reproduce the per-section scan exactly."""

    def test_individual_values_match_naive_scan(self):
        names = ["Baseline"] + [
            f"{kind} {i}" for i in range(1, 51) for kind in ("Tone", "ITI")
        ]
        df = _section_df(6, names)
        piv = _SessionSectionsWidget._section_pivot(df, "n_bouts", names)
        for sess in df["session_label"].unique():
            got = np.nan_to_num(
                piv.loc[sess].reindex(names).to_numpy(dtype=float), nan=0.0
            )
            assert np.allclose(got, _naive_individual(df, "n_bouts", names, sess))

    def test_group_values_match_naive_scan(self):
        names = ["Baseline", "Tone 1", "ITI 1", "Tone 2", "ITI 2"]
        df = _section_df(5, names, seed=3)
        piv = _SessionSectionsWidget._section_pivot(df, "duration_s", names)
        for sn in names:
            got = np.sort(piv[sn].dropna().to_numpy(dtype=float))
            assert np.allclose(got, np.sort(_naive_group(df, "duration_s", sn)))

    def test_missing_pairs_stay_nan_not_zero(self):
        """A session with no bouts in a section must be dropped from the group
        mean, not averaged in as a zero."""
        names = ["A", "B"]
        df = _section_df(3, names)
        df = df[~((df["session_label"] == "m1") & (df["section_name"] == "B"))]
        piv = _SessionSectionsWidget._section_pivot(df, "n_bouts", names)
        assert np.isnan(piv.loc["m1", "B"])
        assert len(piv["B"].dropna()) == 2

    def test_duplicate_section_names_stay_one_dimensional(self):
        names = ["Tone 1", "Tone 1", "ITI 1"]
        df = _section_df(2, names)
        piv = _SessionSectionsWidget._section_pivot(df, "n_bouts", names)
        assert list(piv.columns) == ["Tone 1", "ITI 1"]
        assert piv["Tone 1"].ndim == 1
        # The per-section order still expands back to every defined section.
        assert len(piv.loc["m0"].reindex(names)) == 3

    def test_empty_frame_returns_empty_pivot(self):
        piv = _SessionSectionsWidget._section_pivot(
            pd.DataFrame(columns=["session_label", "section_name", "n_bouts"]),
            "n_bouts", ["A", "B"],
        )
        assert piv.empty
        assert list(piv.columns) == ["A", "B"]


class TestFastBars:
    """The collection-drawn bars must look exactly like ax.bar's."""

    @staticmethod
    def _render(vals: np.ndarray, fast: bool):
        n_sess, n_sec = vals.shape
        x = np.arange(n_sec)
        bar_w = 0.8 / n_sess
        fig = Figure(figsize=(8, 4), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for i in range(n_sess):
            offset = (i - n_sess / 2 + 0.5) * bar_w
            if fast:
                _SessionSectionsWidget._fast_bars(
                    ax, x + offset, vals[i], bar_w * 0.9, "#1f77b4", 0.85, f"m{i}"
                )
            else:
                ax.bar(
                    x + offset, vals[i], bar_w * 0.9,
                    color="#1f77b4", alpha=0.85, label=f"m{i}",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in x], rotation=30, ha="right")
        fig.canvas.draw()
        return fig, ax, np.asarray(fig.canvas.buffer_rgba()).copy()

    @pytest.mark.parametrize("vals", [
        np.arange(1, 13, dtype=float).reshape(3, 4),
        np.zeros((3, 4)),
        np.array([[0.0, 5.0, 0.0], [3.0, 0.0, 1.0]]),
        np.array([[1.0]]),
    ])
    def test_pixels_match_ax_bar(self, vals):
        _f1, ax_bar, img_bar = self._render(vals, fast=False)
        _f2, ax_fast, img_fast = self._render(vals, fast=True)
        assert ax_fast.get_xlim() == pytest.approx(ax_bar.get_xlim())
        assert ax_fast.get_ylim() == pytest.approx(ax_bar.get_ylim())
        assert np.array_equal(img_bar, img_fast)

    def test_one_artist_per_series_not_per_bar(self):
        vals = np.ones((4, 101))
        _fig, ax, _img = self._render(vals, fast=True)
        assert len(ax.collections) == 4
        assert not ax.patches

    def test_series_label_reaches_the_legend(self):
        vals = np.ones((3, 5))
        _fig, ax, _img = self._render(vals, fast=True)
        _handles, labels = ax.get_legend_handles_labels()
        assert labels == ["m0", "m1", "m2"]


class TestLegendPlacement:
    """'best' placement scans every artist, so it is dropped for long legends."""

    def test_best_kept_for_short_legends(self):
        assert _SessionSectionsWidget._legend_loc_for(6, "best") == "best"

    def test_best_dropped_for_long_legends(self):
        assert _SessionSectionsWidget._legend_loc_for(80, "best") == "upper right"

    def test_explicit_choice_is_never_overridden(self):
        assert _SessionSectionsWidget._legend_loc_for(80, "lower left") == "lower left"


class TestLptPresets:
    """The shipped LPT presets are what the timing above assumes."""

    @pytest.mark.parametrize("name,shock_trials", [
        ("LPT Conditioning 1", {4, 13, 26, 31, 45}),
        ("LPT Conditioning 2", {3, 11, 21, 35, 42}),
        ("LPT Extinction", set()),
    ])
    def test_trial_structure(self, name, shock_trials):
        preset = next(
            p for p in _SessionSectionsWidget._BUILTIN_PRESETS if p["name"] == name
        )
        secs = preset["sections"]
        assert secs[0] == {"name": "Baseline", "duration": 180}
        assert len(secs) == 1 + 50 * 2
        # 3-min baseline + 50 trials of 15 s tone + 60 s ITI.
        assert sum(s["duration"] for s in secs) == 180 + 50 * 75
        for trial in range(1, 51):
            tone, iti = secs[trial * 2 - 1], secs[trial * 2]
            assert tone["duration"] == 15
            assert iti == {"name": f"ITI {trial}", "duration": 60}
            expected = f"Tone {trial}" + (
                " (Shock)" if trial in shock_trials else ""
            )
            assert tone["name"] == expected
