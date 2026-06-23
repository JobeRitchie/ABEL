"""Quick diagnostic: check dead features in the audit report and inspect frame_pose.parquet."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r"C:\Users\jober\TMT_2.0")
AUDIT = PROJECT / "derived" / "analysis" / "feature_audit_report.json"
FRAME_POSE = PROJECT / "derived" / "pose_features" / "frame_pose.parquet"

# 1. Show dead features from audit
print("=" * 60)
print("DEAD FEATURES FROM AUDIT REPORT")
print("=" * 60)
with open(AUDIT) as f:
    report = json.load(f)

features = report.get("features", [])
dead = [e for e in features if isinstance(e, dict) and e.get("status") == "dead"]
print(f"Total dead: {len(dead)}")
for d in dead:
    name = d.get("name", d.get("feature", str(d)))
    print(f"  {name}")

# 2. Inspect frame_pose.parquet for forepaw columns
print()
print("=" * 60)
print("FOREPAW COLUMNS IN frame_pose.parquet")
print("=" * 60)
df = pd.read_parquet(FRAME_POSE)
forepaw_cols = [c for c in df.columns if "forepaw" in c.lower() or "paw" in c.lower()]
for c in forepaw_cols:
    vals = df[c].dropna()
    print(f"  {c:40s}  std={vals.std():.6f}  nonzero={np.count_nonzero(vals)}/{len(vals)}")

# 3. Check front_leg columns
print()
print("=" * 60)
print("FRONT_LEG / FRONTLEG COLUMNS IN frame_pose.parquet")
print("=" * 60)
fl_cols = [c for c in df.columns if "front_leg" in c.lower() or "frontleg" in c.lower()]
for c in fl_cols:
    vals = df[c].dropna()
    print(f"  {c:40s}  std={vals.std():.6f}  nonzero={np.count_nonzero(vals)}/{len(vals)}")

# 4. Show all column names for reference
print()
print("=" * 60)
print(f"ALL COLUMNS ({len(df.columns)} total)")
print("=" * 60)
for c in sorted(df.columns):
    print(f"  {c}")
