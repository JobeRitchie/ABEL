"""Tests for the LOSO publication figure + CSV export."""

from __future__ import annotations

import csv

import pytest

from abel.validation import loso_plot


def _results() -> list[dict]:
    return [
        {
            "behavior_name": "Approach",
            "n_subjects": 4,
            "fold_prauc_mean": 0.82, "fold_prauc_sem": 0.05,
            "fold_f1_mean": 0.71, "fold_f1_sem": 0.06,
            "refined_f1": 0.68,
            "folds": [
                {"subject": "MS1", "f1": 0.7, "pr_auc": 0.80},
                {"subject": "MS2", "f1": 0.72, "pr_auc": 0.84},
                {"subject": "MS3", "skipped": "no target positives in holdout"},
            ],
        },
        {"behavior_name": "Rear", "error": "need >=2 subjects, found 1"},
    ]


def test_bar_chart_returns_figure_or_skips() -> None:
    if not loso_plot._HAS_MPL:
        pytest.skip("matplotlib not installed")
    fig = loso_plot.loso_bar_chart(_results())
    assert fig is not None
    ax = fig.axes[0]
    # One group per scorable behavior (the errored one is dropped).
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Approach"]
    assert ax.get_ylim()[1] == pytest.approx(1.05)


def test_bar_chart_empty_results_returns_none() -> None:
    assert loso_plot.loso_bar_chart([]) is None
    assert loso_plot.loso_bar_chart([{"error": "x"}]) is None


def test_csv_export_writes_summary_and_fold_rows(tmp_path) -> None:
    out = tmp_path / "loso.csv"
    loso_plot.loso_results_to_csv(_results(), out)
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    summaries = [r for r in rows if r["row_type"] == "summary"]
    folds = [r for r in rows if r["row_type"] == "fold"]
    # One summary per behavior (incl. the errored one), two scorable folds.
    assert len(summaries) == 2
    assert len(folds) == 2
    approach = next(r for r in summaries if r["behavior"] == "Approach")
    assert approach["prauc_mean"] == "0.820000"
    assert approach["n_subjects"] == "4"
    errored = next(r for r in summaries if r["behavior"] == "Rear")
    assert "need >=2" in errored["error"]
    # The skipped fold (no f1/pr_auc) is not written as a fold row.
    assert all(r["subject"] in {"MS1", "MS2"} for r in folds)
