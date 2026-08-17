import json
import os
from pathlib import Path

P = Path("c:/Users/jober/Models For Manuscript/OpenField_summer/config/session_registry.json")
data = json.loads(P.read_text())
entries = data.get("entries", {})
miss = []
for sid, v in entries.items():
    path = v.get("video_path")
    if not path or not os.path.exists(path):
        miss.append((sid, path))

print(f"checked {len(entries)} sessions; missing: {len(miss)}")
for sid, path in miss:
    print(sid, path)
