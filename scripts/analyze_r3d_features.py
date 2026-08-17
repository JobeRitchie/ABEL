import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd


PRJ = Path(r"c:/Users/jober/Models For Manuscript/OpenField_summer")
SEG = PRJ / "derived" / "representations" / "segment_features.parquet"
MAN = PRJ / "derived" / "representations" / "representations.manifest.json"

if not SEG.exists():
    print("segment_features.parquet not found at", SEG)
    sys.exit(2)

print("Loading segment features... this may take a moment")
df = pd.read_parquet(SEG)
cols = df.columns.tolist()
print("Total rows:", len(df), "Total cols:", len(cols))

# R3D columns
r3d_cols = [c for c in cols if c.startswith("r3d_")]
print("R3D cols found:", len(r3d_cols))
if r3d_cols:
    na_frac = df[r3d_cols].isna().mean().sort_values()
    print("NaN fraction (lowest 10):")
    print(na_frac.head(10).to_string())
    print("NaN fraction (highest 10):")
    print(na_frac.tail(10).to_string())
    variances = df[r3d_cols].var(ddof=0)
    print("R3D mean variance:", float(variances.mean()))
    allnan = df[r3d_cols].isna().all(axis=1).sum()
    print("Rows with all-R3D NaN:", int(allnan))
else:
    print("No r3d_* columns present in segment_features.parquet")

# Check for train/split columns
train_cols = [c for c in cols if re.search(r"train|fold|split|is_train", c, re.I)]
print("Potential train/split columns:", train_cols)
for c in train_cols:
    print(c, "unique values:", df[c].dropna().unique()[:20])

# Check representations manifest to see which feature_columns were recorded
if MAN.exists():
    import json
    man = json.loads(MAN.read_text())
    fcols = man.get("feature_columns", [])
    print("feature_columns in manifest: {} columns".format(len(fcols)))
    missing = [c for c in r3d_cols if c not in fcols]
    print("R3D columns missing from manifest feature_columns:", len(missing))
    if missing:
        print("Examples:", missing[:10])

print("Done.")
