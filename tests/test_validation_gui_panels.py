"""Every validation analysis must actually put its figures on its GUI tab.

The discrimination tab shipped broken twice over, in two independent ways, and both
failures were invisible: the run succeeded, the PNGs were written, and only the GUI's
lookup was wrong, so the tab just said "No figures produced yet."

    1. `_ResultPanel` only built a view dropdown when constructed with `views=`.
       Discrimination and Generalization were constructed without one but populated
       via `set_views()`, so their images were filed under keys the panel never read.
    2. The runner writes one matrix per feature family
       (`…__separability_matrix__pose_video.png`); the GUI globbed for
       `*__separability_matrix.png`, which matches none of them.

So this test builds a run directory with the *exact* filenames
:mod:`abel.validation.runner` writes (see its plots/csvs section), feeds it to the
real `ValidationWindow._on_finished`, and asserts each tab ends up with rendered
thumbnails — not merely with a populated dict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
# The validation suite is not version-controlled (see .gitignore); skip rather than
# fail collection in a checkout that does not carry it.
pytest.importorskip("abel.validation.runner")

from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from abel.validation.plots import LEARNING_CURVE_VIEWS  # noqa: E402
from abel.validation.runner import (  # noqa: E402
    ANALYSIS_ABLATION, ANALYSIS_AL_CURVE, ANALYSIS_DISCRIMINATION,
    ANALYSIS_GENERALIZATION, ANALYSIS_LEARNING_CURVE, ANALYSIS_RARE_DISCOVERY,
    RunOutputs,
)

ALL_ANALYSES = [
    ANALYSIS_LEARNING_CURVE, ANALYSIS_ABLATION, ANALYSIS_DISCRIMINATION,
    ANALYSIS_GENERALIZATION, ANALYSIS_AL_CURVE, ANALYSIS_RARE_DISCOVERY,
]

PROJ = "proj1"          # runner._tag() of a project id
BEH = "Freeze"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _png(path: Path) -> None:
    """A real (loadable) PNG — QPixmap skips anything it cannot decode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = QPixmap(120, 90)
    pix.fill(QColor("#89b4fa"))
    assert pix.save(str(path), "PNG")


def _csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a,b\n1,2\n", encoding="utf-8")


def _fake_run_dir(root: Path) -> Path:
    """Mirror of what runner.run_validation writes for every analysis."""
    run = root / "run_20260712_000000"

    # learning_curves: f"{stem}__{view}.png" per view, + the 0_AVERAGE curves
    for view in LEARNING_CURVE_VIEWS:
        _png(run / "learning_curves" / f"{PROJ}__{BEH}__{view}.png")
        _png(run / "learning_curves" / f"0_AVERAGE__{view}.png")
    _csv(run / "learning_curves" / "learning_curve_points.csv")

    # ablation: f"feature_impact__{idx}_{budget_label}.png"
    _png(run / "ablation" / "feature_impact__0_50clips.png")
    _png(run / "ablation" / "feature_impact__1_all.png")
    _csv(run / "ablation" / "ablation_results.csv")

    # discrimination: one matrix per ADD-ON feature family + one gain plot
    _png(run / "discrimination" / f"{PROJ}__separability_matrix__pose_context.png")
    _png(run / "discrimination" / f"{PROJ}__separability_matrix__pose_video.png")
    _png(run / "discrimination" / f"{PROJ}__feature_gain_by_pair.png")
    _csv(run / "discrimination" / "confusable_pairs.csv")
    _csv(run / "discrimination" / "discrimination_results.csv")

    # generalization (+ the two analyses it emits from the same predictions)
    _png(run / "generalization" / "model_vs_human_kappa.png")
    _csv(run / "generalization" / "agreement.csv")
    _png(run / "time_budget" / f"0_AGREEMENT_FOREST__{PROJ}.png")
    _png(run / "time_budget" / f"{PROJ}__{BEH}.png")
    _csv(run / "time_budget" / "time_budget_agreement.csv")
    _csv(run / "time_budget" / "time_budget_points.csv")
    _png(run / "calibration" / f"{PROJ}__{BEH}.png")
    _csv(run / "calibration" / "calibration.csv")

    # active learning
    _png(run / "active_learning" / f"{PROJ}__{BEH}.png")
    _csv(run / "active_learning" / "al_vs_random_points.csv")
    _csv(run / "active_learning" / "al_vs_random_summary.csv")

    # rare discovery
    _png(run / "rare_discovery" / f"{PROJ}__{BEH}.png")
    _csv(run / "rare_discovery" / "discovery.csv")
    _csv(run / "rare_discovery" / "rarity_scaling.csv")
    _csv(run / "prism" / "prism_behavior_rarity.csv")
    _csv(run / "prism" / "prism_discovery_reviewed.csv")
    _csv(run / "prism" / "prism_effort_reviewed.csv")
    _csv(run / "prism" / "prism_discovery_fullpool.csv")
    _csv(run / "prism" / "prism_rarity_scaling.csv")

    # cross-project (written on every run)
    _png(run / "cross_project" / "0_forest_by_behavior.png")
    _png(run / "cross_project" / "accuracy_bars.png")
    _csv(run / "cross_project" / "dashboard.csv")
    _csv(run / "cross_project" / "training_speed.csv")
    return run


@pytest.fixture()
def populated_window(qapp, tmp_path):
    import pandas as pd

    from abel.validation.gui import ValidationWindow

    run_dir = _fake_run_dir(tmp_path)
    win = ValidationWindow()
    win._on_finished(
        RunOutputs(run_dir=run_dir, cells=pd.DataFrame(),
                   report_path=run_dir / "report.html",
                   manifest_path=run_dir / "manifest.json"),
        ALL_ANALYSES,
    )
    yield win
    win.close()


@pytest.mark.parametrize(
    "panel_attr, tab",
    [
        ("_lc_panel", "Learning Curves"),
        ("_abl_panel", "Ablation"),
        ("_disc_panel", "Discrimination"),
        ("_gen_panel", "Generalization"),
        ("_al_panel", "Active Learning"),
        ("_rare_panel", "Rare Discovery"),
        ("_cross_panel", "Cross-Project"),
    ],
)
def test_tab_shows_figures_after_run(populated_window, panel_attr, tab):
    """Each analysis tab renders at least one thumbnail once the run completes."""
    panel = getattr(populated_window, panel_attr)
    assert panel._current_images(), f"{tab} tab: no figures selected for display"
    assert panel._strip._labels, f"{tab} tab: figures found but no thumbnail rendered"


def test_every_view_of_every_panel_renders(populated_window):
    """Switching to any view in a panel's dropdown shows figures — no dead entries."""
    for attr in ("_lc_panel", "_disc_panel", "_gen_panel"):
        panel = getattr(populated_window, attr)
        combo = panel._view_combo
        assert combo.count() > 0, f"{attr}: view dropdown is empty"
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            assert panel._strip._labels, (
                f"{attr}: view '{combo.itemText(i)}' rendered no figures")


def test_discrimination_views_cover_each_feature_family(populated_window):
    """Both per-family separability matrices and the gain plot are reachable."""
    combo = populated_window._disc_panel._view_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert any("Context" in t for t in labels), labels
    assert any("Video" in t for t in labels), labels
    assert any("gain" in t.lower() for t in labels), labels


def test_data_export_targets_resolve(populated_window):
    """Every tab also has a CSV wired up, so Export/Copy Data is not dead."""
    for attr in ("_lc_panel", "_abl_panel", "_disc_panel", "_gen_panel",
                 "_al_panel", "_rare_panel", "_cross_panel"):
        panel = getattr(populated_window, attr)
        table = panel._current_table()
        assert table is not None and Path(table).exists(), f"{attr}: no exportable data"
        assert panel._csv_btn.isEnabled() and panel._copy_btn.isEnabled()


def test_rare_discovery_panel_exposes_per_figure_prism_exports(populated_window):
    """The rare-discovery tab should offer a separate export for each figure."""
    labels = [populated_window._rare_panel._data_combo.itemText(i)
              for i in range(populated_window._rare_panel._data_combo.count())]
    assert "Behavior rarity" in labels
    assert "Discovery curve (reviewed)" in labels
    assert "Effort-to-N (reviewed)" in labels
    assert "Discovery curve (full pool)" in labels
    assert "Rarity scaling" in labels


@pytest.mark.parametrize(
    "handler, panel_attr, tab",
    [
        ("_on_behaviorscape_finished", "_bscape_panel", "Behaviorscape"),
        ("_on_video_value_finished", "_vv_panel", "Video Features"),
        ("_on_benchmark_finished", "_bench_panel", "Throughput"),
    ],
)
def test_worker_driven_tabs_show_figures(qapp, tmp_path, handler, panel_attr, tab):
    """The tabs fed straight from a worker (not from a run dir) also render."""
    from abel.validation.gui import ValidationWindow

    img = tmp_path / f"{panel_attr}.png"
    _png(img)
    csv = tmp_path / f"{panel_attr}.csv"
    _csv(csv)

    win = ValidationWindow()
    getattr(win, handler)({"images": [img], "tables": {tab: csv}, "summary": "ok"})
    panel = getattr(win, panel_attr)
    assert panel._current_images(), f"{tab} tab: no figures selected for display"
    assert panel._strip._labels, f"{tab} tab: figures found but no thumbnail rendered"
    win.close()


def test_missing_figure_is_skipped_not_raised(qapp, tmp_path):
    """A plot helper that bailed out (None / missing file) must not crash the tab."""
    from abel.validation.gui import ValidationWindow

    good = tmp_path / "good.png"
    _png(good)
    win = ValidationWindow()
    win._bench_panel.set_simple([None, tmp_path / "never_written.png", good], None)
    assert len(win._bench_panel._strip._labels) == 1
    win.close()


def test_empty_run_leaves_no_dead_dropdown(qapp, tmp_path):
    """A run that produced nothing must not leave stale views selected."""
    import pandas as pd

    from abel.validation.gui import ValidationWindow

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    win = ValidationWindow()
    win._on_finished(
        RunOutputs(run_dir=run_dir, cells=pd.DataFrame(),
                   report_path=run_dir / "report.html",
                   manifest_path=run_dir / "manifest.json"),
        ALL_ANALYSES,
    )
    assert win._disc_panel._current_images() == []
    assert win._disc_panel._view_combo.count() == 0
    assert not win._disc_panel._png_btn.isEnabled()
    win.close()
