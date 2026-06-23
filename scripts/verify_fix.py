"""Verify that the analytics bout recomputation fix works on real data.

This simulates what the fixed _load_from_target_behavior_tr() and
_load_raw_bouts() now do: recompute bouts from probability traces using
per-behavior thresholds from temporal_review_settings.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the abel package to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abel.temporal_refinement.bout_postprocess import (
    smooth_probabilities,
    threshold_probabilities,
    merge_close_bouts,
    remove_short_bouts,
    binary_trace_to_intervals,
)

PROJECT = Path(r"C:\Users\jober\TMT_smallwindow")
FPS = 30.0

# --- Load behavior definitions ---
beh_path = PROJECT / "config" / "behavior_definitions.yaml"
import yaml
with open(beh_path, encoding="utf-8") as f:
    beh_data = yaml.safe_load(f)
bid_name_map = {}
for b in beh_data.get("behaviors", []):
    bid = str(b.get("behavior_id", ""))
    name = str(b.get("name", bid))
    if bid and bid != "no_behavior":
        bid_name_map[bid] = name
print("Behaviors:", bid_name_map)

# --- Load temporal review settings (per-behavior thresholds) ---
settings_path = PROJECT / "config" / "temporal_review_settings.json"
with open(settings_path, encoding="utf-8") as f:
    review_settings = json.load(f)
defaults = review_settings.get("__all__", {})
by_behavior = review_settings.get("by_behavior", {})
thresholds = {}
for key, vals in by_behavior.items():
    if key == "target_behavior":
        continue
    thresholds[key] = {
        "onset_threshold": float(vals.get("onset_threshold", defaults.get("onset_threshold", 0.5))),
        "min_bout_duration_frames": int(vals.get("min_bout_duration_frames", defaults.get("min_bout_duration_frames", 6))),
        "merge_gap_frames": int(vals.get("merge_gap_frames", defaults.get("merge_gap_frames", 3))),
    }
print("\nPer-behavior thresholds:")
for bid, params in thresholds.items():
    name = bid_name_map.get(bid, bid)
    print(f"  {name} ({bid}): onset={params['onset_threshold']}, "
          f"min_bout={params['min_bout_duration_frames']}, merge_gap={params['merge_gap_frames']}")

# --- Load trace paths and smoothing params ---
latest_path = PROJECT / "derived" / "temporal_refinement" / "target_behavior" / "latest.json"
with open(latest_path, encoding="utf-8") as f:
    latest = json.load(f)
post_dir = latest.get("postprocess_dir", "")
inf_dir = latest.get("inference_dir", "")

# Smoothing from postprocess manifest
pm_path = Path(post_dir) / "postprocess_manifest.json"
with open(pm_path, encoding="utf-8") as f:
    pm = json.load(f)
smoothing_method = pm.get("smoothing_method", "moving_average")
smoothing_window = int(pm.get("smoothing_window", 5))
print(f"\nSmoothing: method={smoothing_method}, window={smoothing_window}")

# Trace paths
inf_path = Path(inf_dir) / "inference_manifest.json"
with open(inf_path, encoding="utf-8") as f:
    inf_manifest = json.load(f)
trace_paths = {str(k): str(v) for k, v in inf_manifest.get("trace_paths", {}).items()}

# --- Session registry for subject names ---
reg_path = PROJECT / "config" / "session_registry.json"
with open(reg_path, encoding="utf-8") as f:
    registry = json.load(f)
subject_by_session = {}
entries = registry.get("entries", {}) if isinstance(registry, dict) else {}
for sid, entry in entries.items():
    subj = str(entry.get("subject_id", sid))
    subject_by_session[sid] = subj

# --- Target sessions ---
target_sessions = {
    "session_741f3824": "TMT11",
    "session_c3f4e39f": "TMT16",
}

# --- Recompute bouts ---
print("\n" + "=" * 70)
print("RECOMPUTING BOUTS FROM TRACES (simulating fixed analytics)")
print("=" * 70)

for sid, label in target_sessions.items():
    tp_str = trace_paths.get(sid, "")
    if not tp_str or not Path(tp_str).exists():
        print(f"\n{label} ({sid}): NO TRACE FILE")
        continue

    trace_df = pd.read_parquet(tp_str)
    frame_arr = trace_df["frame"].to_numpy(dtype=int)
    print(f"\n{'=' * 60}")
    print(f"{label} ({sid}): {len(trace_df)} frames")
    print(f"{'=' * 60}")

    for bid, bname in bid_name_map.items():
        prob_col = f"prob_{bid}"
        if prob_col not in trace_df.columns:
            print(f"  {bname}: column {prob_col} not found")
            continue

        raw = pd.to_numeric(trace_df[prob_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        smoothed = smooth_probabilities(raw, method=smoothing_method, window=smoothing_window)

        params = thresholds.get(bid, {
            "onset_threshold": float(defaults.get("onset_threshold", 0.5)),
            "min_bout_duration_frames": int(defaults.get("min_bout_duration_frames", 6)),
            "merge_gap_frames": int(defaults.get("merge_gap_frames", 3)),
        })
        onset = params["onset_threshold"]
        min_bout = int(params["min_bout_duration_frames"])
        merge_gap = int(params["merge_gap_frames"])

        binary = threshold_probabilities(smoothed, onset_thresh=onset, offset_thresh=onset)
        binary = merge_close_bouts(binary, max_gap_frames=merge_gap)
        binary = remove_short_bouts(binary, min_duration_frames=min_bout)
        intervals = binary_trace_to_intervals(binary)

        mapped_intervals = []
        for s_idx, e_idx in intervals:
            sf = int(frame_arr[s_idx]) if s_idx < len(frame_arr) else s_idx
            ef = int(frame_arr[min(e_idx, len(frame_arr) - 1)])
            mapped_intervals.append((sf, ef))

        n_bouts = len(mapped_intervals)
        total_frames = sum(ef - sf + 1 for sf, ef in mapped_intervals)
        time_s = total_frames / FPS

        print(f"  {bname}: {n_bouts} bouts, {total_frames} frames, {time_s:.1f}s "
              f"(onset={onset}, min_bout={min_bout}, merge_gap={merge_gap})")

# --- Compare against what the OLD code would have loaded ---
print("\n" + "=" * 70)
print("COMPARISON: What the OLD code loaded (stored postprocess bouts)")
print("=" * 70)

bout_paths = {str(k): str(v) for k, v in pm.get("bout_paths", {}).items()}
for sid, label in target_sessions.items():
    bp_str = bout_paths.get(sid, "")
    if not bp_str or not Path(bp_str).exists():
        print(f"  {label}: NO BOUT FILE")
        continue
    bout_df = pd.read_parquet(bp_str)
    print(f"  {label}: {len(bout_df)} bouts in stored file (onset_threshold was 0.98)")

# --- Also show what behavior_bouts/ eval files had ---
print("\n" + "=" * 70)
print("COMPARISON: behavior_bouts/ eval pipeline files")
print("=" * 70)

bouts_dir = PROJECT / "derived" / "behavior_bouts"
for bid, bname in bid_name_map.items():
    bout_path = bouts_dir / f"{bid}_bouts.parquet"
    if not bout_path.exists():
        continue
    df = pd.read_parquet(bout_path)
    for sid, label in target_sessions.items():
        grp = df[df["session_id"] == sid]
        if grp.empty:
            print(f"  {label} {bname}: 0 bouts")
        else:
            n = len(grp)
            if "duration_frames" in grp.columns:
                dur = float(grp["duration_frames"].sum())
            else:
                dur = float((grp["end_frame"] - grp["start_frame"] + 1).sum())
            print(f"  {label} {bname}: {n} bouts, {dur/FPS:.1f}s")

print("\nDone.")
