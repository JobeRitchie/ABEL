"""Auto-tuned onset thresholds come from held-out window predictions only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel.storage.file_store import write_json
from abel.temporal_refinement.onset_autotune import suggest_for_predictions, suggest_onset_thresholds


def _preds(pos_probs, neg_probs, target_index=0):
    y = [target_index] * len(pos_probs) + [1 - target_index] * len(neg_probs)
    return pd.DataFrame({
        "label_true": y,
        "prediction_prob": list(pos_probs) + list(neg_probs),
        "target_index": target_index,
    })


def test_squashed_probabilities_get_a_low_threshold() -> None:
    # Ranking is perfect but every positive sits below 0.1: the 0.3 default finds nothing.
    rng = np.random.default_rng(0)
    preds = _preds(rng.uniform(0.06, 0.09, 40), rng.uniform(0.0, 0.04, 400))
    s = suggest_for_predictions("b", "Allogroom", 0.3, preds)
    assert s.is_change
    # Window midpoint ~0.05, shrunk toward the 0.1 prior.
    assert 0.08 <= s.suggested <= 0.095
    assert s.f1_current == 0.0


def test_threshold_at_the_suggestion_is_left_alone() -> None:
    preds = _preds(np.full(40, 0.9), np.full(400, 0.1))
    first = suggest_for_predictions("b", "Dig", 0.5, preds)
    # Flat plateau 0.1-0.9, midpoint ~0.5, shrunk to ~0.2.
    assert abs(first.suggested - 0.2) < 0.01
    s = suggest_for_predictions("b", "Dig", first.suggested, preds)
    assert not s.is_change
    assert "already at the suggestion" in s.note


def test_too_few_positives_is_not_tuned() -> None:
    s = suggest_for_predictions("b", "Chase", 0.3, _preds([0.9, 0.8], np.full(100, 0.1)))
    assert not s.is_change
    assert s.val_positives == 2


def test_weak_model_is_flagged_but_still_shrunk() -> None:
    # Positives indistinguishable from negatives: best F1 flags nearly every window.
    rng = np.random.default_rng(1)
    preds = _preds(rng.uniform(0.0, 1.0, 20), rng.uniform(0.0, 1.0, 400))
    s = suggest_for_predictions("b", "Dominate", 0.9, preds)
    assert s.is_change and s.suggested <= 0.32
    assert "Weak model" in s.note


def test_model_found_by_recorded_target_not_folder_name(tmp_path: Path) -> None:
    md = tmp_path / "derived" / "models" / "behavior_model_Custom_Name"
    md.mkdir(parents=True)
    write_json(md / "run_settings.json", {"target_behavior": "uuid-1"})
    _preds(np.full(40, 0.08), np.full(400, 0.01)).to_parquet(md / "validation_predictions.parquet")
    out = suggest_onset_thresholds(tmp_path, [("uuid-1", "Groom"), ("uuid-2", "Rear")], {"uuid-1": 0.3})
    assert out[0].is_change
    assert not out[1].is_change and "No trained model" in out[1].note


def test_bout_cleanup_scales_with_fps() -> None:
    from abel.temporal_refinement.onset_autotune import suggest_bout_cleanup
    assert suggest_bout_cleanup(30.0) == (12, 6)
    assert suggest_bout_cleanup(60.0) == (24, 12)


def test_flat_plateau_picks_its_middle_not_its_top_edge() -> None:
    # Dense traces average overlapping windows, so short bouts peak below their window
    # probability; the top of a flat plateau loses them.
    from abel.temporal_refinement.onset_autotune import _GRID, _plateau_midpoint

    f1 = np.where((_GRID >= 0.15) & (_GRID <= 0.55), 0.76, 0.5)
    f1[np.argmin(np.abs(_GRID - 0.5))] = 0.765  # the argmax sits near the top edge
    assert abs(_plateau_midpoint(f1) - 0.35) <= 0.01


def test_threshold_at_plateau_edge_moves_even_at_equal_f1() -> None:
    # The old rule demanded an F1 gain, so a threshold on the flat top edge (Sniff
    # Anogenital at 0.56) was never moved and kept losing short trace bouts.
    preds = _preds(np.full(40, 0.9), np.full(400, 0.1))
    s = suggest_for_predictions("b", "Sniff Anogenital", 0.85, preds)
    assert s.is_change
    assert s.suggested < 0.85
    assert s.f1_suggested == s.f1_current


def test_recall_beta_lowers_the_threshold() -> None:
    # Overlapping scores: F-beta 1.5 trades a few false windows for caught positives.
    rng = np.random.default_rng(3)
    preds = _preds(rng.normal(0.6, 0.15, 60).clip(0, 1), rng.normal(0.3, 0.15, 600).clip(0, 1))
    f1 = suggest_for_predictions("b", "Attack", 0.9, preds)
    recall = suggest_for_predictions("b", "Attack", 0.9, preds, recall_beta=1.5)
    assert recall.suggested < f1.suggested


def test_shrunk_suggestions_stay_in_a_narrow_band() -> None:
    from abel.temporal_refinement.onset_autotune import shrink_threshold
    assert shrink_threshold(0.01) >= 0.075
    assert shrink_threshold(0.95) <= 0.32
