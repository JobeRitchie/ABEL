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
    assert 0.04 <= s.suggested <= 0.06
    assert s.f1_current == 0.0
    assert s.f1_suggested == 1.0


def test_near_optimal_threshold_is_left_alone() -> None:
    preds = _preds(np.full(40, 0.9), np.full(400, 0.1))
    s = suggest_for_predictions("b", "Dig", 0.5, preds)
    assert not s.is_change
    assert "already at the suggestion" in s.note


def test_too_few_positives_is_not_tuned() -> None:
    s = suggest_for_predictions("b", "Chase", 0.3, _preds([0.9, 0.8], np.full(100, 0.1)))
    assert not s.is_change
    assert s.val_positives == 2


def test_accept_everything_optimum_is_rejected() -> None:
    # Positives indistinguishable from negatives: best F1 flags nearly every window.
    rng = np.random.default_rng(1)
    preds = _preds(rng.uniform(0.0, 1.0, 20), rng.uniform(0.0, 1.0, 400))
    s = suggest_for_predictions("b", "Dominate", 0.9, preds)
    assert not s.is_change
    assert "too weak" in s.note


def test_model_found_by_recorded_target_not_folder_name(tmp_path: Path) -> None:
    md = tmp_path / "derived" / "models" / "behavior_model_Custom_Name"
    md.mkdir(parents=True)
    write_json(md / "run_settings.json", {"target_behavior": "uuid-1"})
    _preds(np.full(40, 0.08), np.full(400, 0.01)).to_parquet(md / "validation_predictions.parquet")
    out = suggest_onset_thresholds(tmp_path, [("uuid-1", "Groom"), ("uuid-2", "Rear")], {"uuid-1": 0.3})
    assert out[0].is_change
    assert not out[1].is_change and "No trained model" in out[1].note


def test_threshold_below_deployed_trace_baseline_is_rejected() -> None:
    # Held-out windows favor 0.06, but the deployed trace idles at 0.11: 0.06 would mark every frame.
    rng = np.random.default_rng(0)
    preds = _preds(rng.uniform(0.06, 0.09, 40), rng.uniform(0.0, 0.04, 400))
    trace = np.full(1000, 0.11)
    s = suggest_for_predictions("b", "Sniff Body", 0.3, preds, trace_values=trace)
    assert not s.is_change
    assert "baseline" in s.note


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
    assert "fewer short bouts" in s.note


def test_recall_beta_lowers_the_threshold() -> None:
    # Overlapping scores: F-beta 1.5 trades a few false windows for caught positives.
    rng = np.random.default_rng(3)
    preds = _preds(rng.normal(0.6, 0.15, 60).clip(0, 1), rng.normal(0.3, 0.15, 600).clip(0, 1))
    f1 = suggest_for_predictions("b", "Attack", 0.9, preds)
    recall = suggest_for_predictions("b", "Attack", 0.9, preds, recall_beta=1.5)
    assert recall.suggested < f1.suggested


def test_threshold_is_capped_below_the_peaks_of_low_bouts() -> None:
    # Windows separate cleanly at ~0.5, but a fifth of the labeled bouts peak at ~0.3 on
    # the averaged trace, so the held-out best would leave them uncounted.
    rng = np.random.default_rng(4)
    preds = _preds(rng.uniform(0.6, 0.9, 50), rng.uniform(0.0, 0.4, 500))
    peaks = np.r_[rng.uniform(0.28, 0.32, 20), rng.uniform(0.7, 0.95, 80)]
    plain = suggest_for_predictions("b", "Sniff Anogenital", 0.56, preds)
    capped = suggest_for_predictions("b", "Sniff Anogenital", 0.56, preds, bout_peaks=peaks)
    assert plain.suggested is None or plain.suggested > 0.4
    assert capped.suggested <= np.quantile(peaks, 0.1)
    assert np.mean(peaks >= capped.suggested) >= 0.9
    assert "never reach the current threshold" in capped.note


def test_cap_needs_enough_labeled_bouts() -> None:
    rng = np.random.default_rng(5)
    preds = _preds(rng.uniform(0.6, 0.9, 50), rng.uniform(0.0, 0.4, 500))
    few = suggest_for_predictions("b", "Rare", 0.3, preds, bout_peaks=np.full(5, 0.1))
    assert few.suggested is None or few.suggested > 0.4


def test_labeled_bout_peaks_read_the_deployed_trace(tmp_path: Path) -> None:
    from abel.temporal_refinement.onset_autotune import _labeled_bout_peaks

    traces = tmp_path / "derived" / "temporal_refinement" / "target_behavior" / "inference_x" / "animal_probability_traces"
    traces.mkdir(parents=True)
    x = np.zeros(300, dtype=np.float32)
    x[40:60] = 0.3   # low bout
    x[200:240] = 0.9  # strong bout
    pd.DataFrame({"frame": np.arange(300), "prob_b": x}).to_parquet(traces / "s1__track_0_trace.parquet")
    labels = pd.DataFrame({
        "session_id": ["s1"] * 4, "animal_id": ["track_0"] * 4,
        "start_frame": [40, 50, 205, 120], "end_frame": [55, 60, 220, 135],
        "label": ["b", "b|a", "b", "a"],
    })
    peaks = np.sort(_labeled_bout_peaks(tmp_path, "b", labels))
    assert len(peaks) == 2  # the two overlapping low windows are one bout; "a" alone is not b
    assert abs(peaks[0] - 0.3) < 0.01 and abs(peaks[1] - 0.9) < 0.01
