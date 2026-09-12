"""Regenerate docs/readme.txt as a plain-text mirror of README.md.

docs/readme.txt carries the same content as README.md so there is one source of
truth: edit README.md, then run this script.

    python scripts/generate_readme_txt.py
"""
import io, re, sys, textwrap
from pathlib import Path

# Keep hyphenated terms and identifiers intact when wrapping.
WRAP = dict(break_on_hyphens=False, break_long_words=False)

_ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "README.md"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else _ROOT / "docs" / "readme.txt"
W = 78

# Plain-text mirror stays ASCII so it renders in any editor/console.
# Applied before wrapping so multi-char expansions are measured correctly.
ASCII_MAP = {
    '—': '-', '–': '-', '−': '-', '·': '|',
    '≥': '>=', '≤': '<=', '’': "'", '‘': "'",
    '“': '"', '”': '"', '…': '...', '×': 'x',
    'σ': 'sigma', '√': 'sqrt', '₂': '2', '→': '->',
    '±': '+/-', '°': ' deg', ' ': ' ',
}


def deascii(t):
    for a, b in ASCII_MAP.items():
        t = t.replace(a, b)
    return t


def inline(t):
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: m.group(1)
               if m.group(1).rstrip('.').replace(':','') in m.group(2) else f"{m.group(1)} ({m.group(2)})", t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)
    t = re.sub(r'`([^`]+)`', r'\1', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', t)
    return deascii(t)

lines = io.open(SRC, encoding='utf-8').read().split('\n')
out, buf = [], []

def flush(indent=' '):
    global buf
    if buf:
        if out and out[-1] != '':
            out.append('')
        txt = inline(' '.join(buf).strip())
        out.extend(textwrap.wrap(txt, W, initial_indent=indent, subsequent_indent=indent, **WRAP))
        out.append('')
        buf = []

i = 0
first_h1 = True
in_numbered = False
while i < len(lines):
    l = lines[i]; s = l.strip()

    if s.startswith('```'):
        flush(); i += 1
        code = []
        while i < len(lines) and not lines[i].strip().startswith('```'):
            code.append(lines[i]); i += 1
        i += 1
        out.extend(('     ' + c).rstrip() for c in code)
        out.append('')
        continue

    if not s:
        flush(); i += 1; continue

    if re.fullmatch(r'-{3,}', s):
        flush(); i += 1; continue

    h = re.match(r'^(#{1,6})\s+(.*)$', s)
    if h:
        flush()
        in_numbered = False
        lvl, title = len(h.group(1)), inline(h.group(2))
        if lvl == 1 and first_h1:
            first_h1 = False
            out += ['=' * W, f" ABEL - Active-learning Behavior Estimation and Labeling", '=' * W, '']
        elif lvl == 2:
            if out and out[-1] != '': out.append('')
            out += ['', ' ' + deascii(title).upper(), '-' * W, '']
        else:
            out += [' ' + deascii(title), '']
        i += 1
        continue

    li = re.match(r'^(\s*)(?:([-*])|(\d+)\.)\s+(.*)$', l)
    if li:
        flush()
        ind, bullet, num, body = li.groups()
        n_ind = len(ind)
        depth = 0 if n_ind < 5 else (1 if n_ind < 7 else 2)
        i += 1
        while i < len(lines):
            n = lines[i]
            if not n.strip() or re.match(r'^\s*(?:[-*]|\d+\.)\s+', n) or n.strip().startswith(('#', '```')):
                break
            body += ' ' + n.strip(); i += 1
        marker = f"{num}. " if num else "- "
        if num and depth == 0:
            in_numbered = True
        # Bullets nested under a numbered step line up with that step's text.
        base = 3 if (in_numbered and not num) else 0
        pad = ' ' + ' ' * base + '  ' * depth
        out.extend(textwrap.wrap(inline(body), W,
                                 initial_indent=pad + marker,
                                 subsequent_indent=pad + ' ' * len(marker), **WRAP))
        continue

    if s == '**ABEL — Active-learning Behavior Estimation and Labeling**':
        i += 1; continue
    buf.append(s); i += 1

flush()
txt = re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip() + '\n'

txt = deascii(txt)
_left = sorted({c for c in txt if ord(c) > 126})
if _left:
    print('WARNING: non-ASCII remains ->', _left)

io.open(DST, 'w', encoding='ascii', errors='replace', newline='\r\n').write(txt)
print('wrote', DST, len(txt.split('\n')), 'lines')
