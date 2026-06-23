"""Re-extract frame-level pose features for all sessions after _keypoint_xy fix.

Reads session_registry.json, resolves pose file paths, re-extracts all sessions,
consolidates into frame_pose.parquet, and verifies forepaw features are alive.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\jober\Desktop\ABEL realism")

from abel.services.pose_processing_service import PoseProcessingService

PROJECT = Path(r"C:\Users\jober\TMT_2.0")
REG_PATH = PROJECT / "config" / "session_registry.json"
PROJ_YAML = PROJECT / "project.yaml"

pose_svc = PoseProcessingService()

# Read session registry
reg = json.loads(REG_PATH.read_text())
entries = reg.get("entries", {})

# Read default fps from project.yaml
from abel.storage.file_store import read_yaml
proj = read_yaml(PROJ_YAML, {})
default_fps = proj.get("default_fps", 30.0)

print(f"Project: {PROJECT}")
print(f"Sessions: {len(entries)}")
print(f"Default FPS: {default_fps}")
print()

t0 = time.monotonic()
ok_count = 0
fail_count = 0

for i, (sid, sess) in enumerate(entries.items()):
    video_path = sess.get("video_path", "")
    pose_filename = sess.get("pose_filename", "")
    subject_id = sess.get("subject_id", "unknown")

    # Resolve pose file: same directory as the video
    video_dir = Path(video_path).parent if video_path else None
    pose_path = video_dir / pose_filename if video_dir else None

    if not pose_path or not pose_path.exists():
        print(f"  [{i+1}/{len(entries)}] SKIP {sid} — pose not found: {pose_path}")
        fail_count += 1
        continue

    try:
        df = pose_svc.extract_and_save_frame_pose_features(
            project_root=PROJECT,
            pose_path=pose_path,
            fps=default_fps,
            animal_id=subject_id,
            session_id=sid,
            video_id=sid,
        )
        fp_std = df["forepaw_speed"].std() if "forepaw_speed" in df.columns else -1
        print(f"  [{i+1}/{len(entries)}] OK  {sid}  frames={len(df):>6d}  forepaw_speed_std={fp_std:.2f}")
        ok_count += 1
    except Exception as e:
        print(f"  [{i+1}/{len(entries)}] ERR {sid}: {e}")
        fail_count += 1

elapsed = time.monotonic() - t0
print(f"\nExtracted {ok_count}/{len(entries)} sessions in {elapsed:.1f}s  ({fail_count} failed)")

# Consolidate
print("\nConsolidating into frame_pose.parquet...")
out = pose_svc.consolidate_session_files(PROJECT)
print(f"  Written: {out}")

# Verify
import numpy as np
import pandas as pd

df = pd.read_parquet(out)
print(f"\nConsolidated: {len(df)} rows, {len(df.columns)} columns")

forepaw_cols = [c for c in df.columns if "forepaw" in c or "oscillation_energy" in c]
print("\nForepaw / oscillation columns:")
for c in sorted(forepaw_cols):
    vals = df[c].dropna()
    nz = np.count_nonzero(vals)
    print(f"  {c:40s}  std={vals.std():>12.4f}  nonzero={nz}/{len(vals)}  {'ALIVE' if vals.std() > 1e-9 else 'DEAD'}")
