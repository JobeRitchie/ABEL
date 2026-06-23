#!/usr/bin/env python3
from pathlib import Path
files = [
    'abel/services/preprocessing_service.py',
    'abel/ui/tabs/preprocessing_tab.py',
    'abel/ui/tabs/temporal_refinement_tab.py',
]
for p in files:
    pth = Path(p)
    if not pth.exists():
        print('missing', p)
        continue
    b = pth.read_bytes()
    try:
        s = b.decode('utf-8-sig')
    except Exception:
        s = b.decode('utf-8', errors='replace')
    pth.write_text(s, encoding='utf-8')
    print('rewrote', p)
print('done')
