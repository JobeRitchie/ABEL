"""Fix mojibake (UTF-8 bytes decoded as Latin-1) in behavior_analytics_tab.py."""
import pathlib, re, sys

path = pathlib.Path(r"c:\Users\jober\Desktop\ABEL realism\abel\ui\tabs\behavior_analytics_tab.py")
text = path.read_text(encoding="utf-8")

# Common UTF-8-as-Windows-1252 mojibake mappings
# When UTF-8 bytes are misread as Windows-1252, 0x80 becomes € (U+20AC)
FIXES = {
    "\u00e2\u20ac\u00a6": "\u2026",  # … (ellipsis)  E2 80 A6 -> â€¦
    "\u00e2\u20ac\u201d": "\u2014",  # — (em dash)   E2 80 94 -> â€"
    "\u00e2\u20ac\u201c": "\u2013",  # – (en dash)   E2 80 93 -> â€"
    "\u00e2\u20ac\u2122": "\u2019",  # ' (right single quote) E2 80 99
    "\u00e2\u20ac\u0153": "\u201c",  # " (left double quote)  E2 80 9C
    "\u00e2\u20ac\u009d": "\u201d",  # " (right double quote) E2 80 9D
    "\u00c2\u00b2": "\u00b2",        # ² (superscript 2)
    "\u00c2\u00b1": "\u00b1",        # ± (plus-minus)
    "\u00c2\u00a0": " ",             # non-breaking space
    "\u00c3\u00a9": "\u00e9",        # é
    "\u00e2\u2020\u2019": "\u2192",  # → (right arrow)   E2 86 92
    "\u00e2\u2030\u00a5": "\u2265",  # ≥ (greater-equal) E2 89 A5
    "\u00e2\u201d\u20ac": "\u2500",  # ─ (box drawing)   E2 94 80
    "\u00c3\u2014": "\u00d7",        # × (multiplication) C3 97
}

count = 0
for bad, good in FIXES.items():
    n = text.count(bad)
    if n:
        print(f"  Replacing {repr(bad)} -> {repr(good)}  ({n} occurrences)")
        text = text.replace(bad, good)
        count += n

if count == 0:
    print("No mojibake patterns found.")
    sys.exit(0)

path.write_text(text, encoding="utf-8")
print(f"\nFixed {count} corrupted sequences. File saved as UTF-8.")
