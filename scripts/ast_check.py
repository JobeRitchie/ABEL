#!/usr/bin/env python3
import ast
from pathlib import Path
errors = []
for p in Path('.').rglob('*.py'):
    try:
        s = p.read_text(encoding='utf-8')
        ast.parse(s, filename=str(p))
    except Exception as e:
        errors.append((p, e))
if not errors:
    print('AST parse: OK')
else:
    for p,e in errors:
        print('ERROR', p, ':', e)
    raise SystemExit(1)
