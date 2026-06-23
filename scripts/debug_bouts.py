"""Diagnostic script: trace the full bout data pipeline for TMT11 and TMT16."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

root = Path(r"C:\Users\jober\TMT_smallwindow")
post_dir = root / "derived/temporal_refinement/target_behavior/postprocess_2cc31577abc4b3fd"
inf_dir = root / "derived/temporal_refinement/target_behavior/inference_2e73bbf0bde9acbe"

pm = json.loads((post_dir / "postprocess_manifest.json").read_text())
im = json.loads((inf_dir / "inference_manifest.json").read_text())

bout_paths = pm.get("bout_paths", {})
trace_paths = im.get("trace_paths", {})

reg = json.loads((root / "config/session_registry.json").read_text())
entries = reg.get("entries", {})

# Build subject map
subject_map = {}
for sid, info in entries.items():
    subject_map[sid] = info.get("subject_id", sid)

print("=== Postprocess parameters ===")
for k, v in pm.get("postprocess", {}).items():
    print(f"  {k}: {v}")

# Check behavior definitions
beh_path = root / "config" / "behavior_definitions.yaml"
if beh_path.exists():
    import yaml
    beh_defs = yaml.safe_load(beh_path.read_text())
    print("\n=== Behavior definitions ===")
    for b in beh_defs.get("behaviors", []):
        print(f"  {b.get('behavior_id', '?')}: {b.get('name', '?')}")

# Examine specific sessions
for name, sid in [("TMT11", "session_741f3824"), ("TMT16", "session_c3f4e39f")]:
    print(f"\n{'='*60}")
    print(f"=== {name} ({sid}) ===")
    print(f"{'='*60}")
    
    bp = bout_paths.get(sid)
    tp = trace_paths.get(sid)
    
    # BOUT FILE
    print(f"\n  Bout path in manifest: {bp}")
    if bp and Path(bp).exists():
        bdf = pd.read_parquet(bp)
        print(f"  Bout file rows: {len(bdf)}")
        if not bdf.empty:
            print(f"  Columns: {list(bdf.columns)}")
            print(f"  First 5 bouts:")
            print(bdf.head(5).to_string(index=False))
            total_frames = (bdf["end_frame"] - bdf["start_frame"] + 1).sum()
            print(f"  Total duration (frames, inclusive): {total_frames}")
            print(f"  Total duration at 30fps: {total_frames/30:.1f}s")
        else:
            print("  *** BOUT FILE IS EMPTY ***")
    elif bp:
        print(f"  *** BOUT FILE DOES NOT EXIST: {bp} ***")
    else:
        print("  *** NO BOUT PATH IN MANIFEST ***")
    
    # TRACE FILE
    print(f"\n  Trace path in manifest: {tp}")
    if tp and Path(tp).exists():
        tdf = pd.read_parquet(tp)
        print(f"  Trace file: {len(tdf)} frames")
        cols = list(tdf.columns)
        print(f"  Columns: {cols}")
        
        if "predicted_behavior" in tdf.columns:
            vc = dict(tdf["predicted_behavior"].value_counts())
            print(f"  Predicted behavior counts: {vc}")
        
        if "probability" in tdf.columns:
            prob = tdf["probability"].to_numpy()
            print(f"  Probability stats: min={prob.min():.3f}, max={prob.max():.3f}, mean={prob.mean():.3f}")
            print(f"  Frames with prob >= 0.30: {(prob >= 0.30).sum()}")
            print(f"  Frames with prob >= 0.50: {(prob >= 0.50).sum()}")
            print(f"  Frames with prob >= 0.98: {(prob >= 0.98).sum()}")
        
        # Check prob_* columns for Dig specifically
        dig_cols = [c for c in cols if "dig" in c.lower() or "Dig" in c]
        for dc in dig_cols:
            vals = tdf[dc].to_numpy()
            print(f"  Column '{dc}': min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
            print(f"    Frames >= 0.30: {(vals >= 0.30).sum()}, >= 0.50: {(vals >= 0.50).sum()}, >= 0.98: {(vals >= 0.98).sum()}")
        
        # Now simulate what _load_raw_bouts target_behavior section does:
        # It reads bout file, then for each bout, checks trace predicted_behavior
        print(f"\n  --- Simulating _load_raw_bouts behavior resolution ---")
        if bp and Path(bp).exists() and not bdf.empty:
            resolved = {}
            for _, bout in bdf.iterrows():
                start = int(bout["start_frame"])
                end = int(bout["end_frame"])
                sl = tdf[(tdf["frame"] >= start) & (tdf["frame"] <= end)]
                if not sl.empty and "predicted_behavior" in sl.columns:
                    mode_val = sl["predicted_behavior"].mode()
                    if len(mode_val) > 0:
                        bid = str(mode_val.iloc[0])
                        resolved.setdefault(bid, []).append((start, end))
            for bid, intervals in sorted(resolved.items()):
                total = sum(e - s + 1 for s, e in intervals)
                print(f"    {bid}: {len(intervals)} bouts, {total} frames, {total/30:.1f}s")
        
        # Now simulate what SHOULD happen: recompute bouts from trace at threshold 0.30
        print(f"\n  --- Recomputing bouts from trace at threshold 0.30 ---")
        from abel.temporal_refinement.bout_postprocess import (
            smooth_probabilities, threshold_probabilities,
            merge_close_bouts, remove_short_bouts, binary_trace_to_intervals
        )
        
        # Use the Dig-specific column if available, else probability
        dig_col = None
        for c in cols:
            if c.lower().startswith("prob_") and "dig" in c.lower():
                dig_col = c
                break
        if dig_col is None and "probability" in cols:
            dig_col = "probability"
        
        if dig_col:
            raw = tdf[dig_col].to_numpy(dtype=np.float32)
            smoothed = smooth_probabilities(raw, "moving_average", 5)
            binary = threshold_probabilities(smoothed, onset_thresh=0.30, offset_thresh=0.30)
            binary = merge_close_bouts(binary, max_gap_frames=45)
            binary = remove_short_bouts(binary, min_duration_frames=45)
            intervals = binary_trace_to_intervals(binary)
            total_frames = sum(e - s + 1 for s, e in intervals)
            print(f"    Using column: {dig_col}")
            print(f"    Recomputed bouts: {len(intervals)}")
            print(f"    Total duration: {total_frames} frames = {total_frames/30:.1f}s")
            if intervals:
                print(f"    First 5 intervals: {intervals[:5]}")

# Also check what behavior_bouts/ files contain
print(f"\n{'='*60}")
print("=== behavior_bouts/ files ===")
print(f"{'='*60}")
bouts_dir = root / "derived" / "behavior_bouts"
for f in sorted(bouts_dir.glob("*.parquet")):
    df = pd.read_parquet(f)
    print(f"\n  {f.name}: {len(df)} rows")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        if "session_id" in df.columns:
            for sid in ["session_741f3824", "session_c3f4e39f"]:
                subj = subject_map.get(sid, sid)
                sdf = df[df["session_id"] == sid]
                if not sdf.empty:
                    if "duration_frames" in sdf.columns:
                        total = sdf["duration_frames"].sum()
                    elif {"start_frame", "end_frame"}.issubset(sdf.columns):
                        total = (sdf["end_frame"] - sdf["start_frame"] + 1).sum()
                    else:
                        total = 0
                    print(f"    {subj} ({sid}): {len(sdf)} bouts, {total} frames, {total/30:.1f}s")
