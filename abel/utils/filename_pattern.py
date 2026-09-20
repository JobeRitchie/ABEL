"""Build subject/session regexes from a highlighted part of an example filename.

The user selects the characters of one filename that name the subject (or the
session); :func:`generate_capture_regex` turns that selection into a pattern
that extracts the equivalent part from every similarly-structured filename.

A filename stem is read as separator-delimited *fields* (``_ - . space``), and
each field as *runs*: letters, digits, and camelCase words.  The generated
pattern skips the same number of leading fields, generalizes the runs before
the selection inside its field by shape (``[A-Z][a-z]+``, ``\\d+`` …), and
captures either whole fields (when the selection spans them) or the selection's
run shapes.  Tracker suffixes appended after the example's last field
(``.tracked.sleap``, ``DLC_resnet50…``) therefore don't disturb extraction.
"""

from __future__ import annotations

import re

SEPARATOR_CHARS = "_-. "
SEP = r"[_\-. ]"
NOT_SEP = r"[^_\-. ]"


def _fields(stem: str) -> list[tuple[int, int]]:
    """Spans of the separator-delimited fields of *stem*."""
    return [m.span() for m in re.finditer(rf"{NOT_SEP}+", stem)]


def _runs(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Spans of letter/digit/other runs, split at camelCase boundaries."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|[^A-Za-z\d]+", text):
        spans.append((m.start() + offset, m.end() + offset))
    return spans


def _shape(run: str) -> str:
    if run.isdigit():
        return r"\d+"
    if re.fullmatch(r"[A-Z][a-z]+", run):
        return r"[A-Z][a-z]+"
    if re.fullmatch(r"[a-z]+", run):
        return r"[a-z]+"
    if re.fullmatch(r"[A-Z]+", run):
        # Stop before the capital that starts a following camelCase word.
        return r"[A-Z]+(?![a-z])"
    return re.escape(run)


def snap_selection(stem: str, start: int, end: int) -> tuple[int, int]:
    """Trim separators off a selection and widen it to whole runs."""
    start, end = max(0, min(start, end)), min(len(stem), max(start, end))
    while start < end and stem[start] in SEPARATOR_CHARS:
        start += 1
    while end > start and stem[end - 1] in SEPARATOR_CHARS:
        end -= 1
    if start >= end:
        raise ValueError("Select at least one letter or digit of the filename.")
    for fs, fe in _fields(stem):
        for rs, re_ in _runs(stem[fs:fe], fs):
            if rs < start < re_:
                start = rs
            if rs < end < re_:
                end = re_
    return start, end


def generate_capture_regex(stem: str, start: int, end: int) -> str:
    """Regex whose group 1 extracts ``stem[start:end]`` (snapped to whole runs).

    Raises ``ValueError`` for an empty selection.
    """
    start, end = snap_selection(stem, start, end)
    fields = _fields(stem)
    fs = next(i for i, (a, b) in enumerate(fields) if a <= start < b)
    fe = next(i for i, (a, b) in enumerate(fields) if a < end <= b)
    f_start, _ = fields[fs]
    _, f_end = fields[fe]

    pattern = rf"^{SEP}*"
    if fs:
        pattern += rf"(?:{NOT_SEP}+{SEP}+){{{fs}}}"
    pattern += "".join(_shape(stem[a:b]) for a, b in _runs(stem[f_start:start], f_start))

    whole_fields = start == f_start and end == f_end
    if whole_fields:
        body = f"{NOT_SEP}+"
        if fe > fs:
            body += rf"(?:{SEP}+{NOT_SEP}+){{{fe - fs}}}"
    else:
        parts: list[str] = []
        for i in range(fs, fe + 1):
            a = max(start, fields[i][0])
            b = min(end, fields[i][1])
            if i > fs:
                parts.append(f"{SEP}+")
            parts.extend(_shape(stem[x:y]) for x, y in _runs(stem[a:b], a))
        body = "".join(parts)
    pattern += f"({body})"

    match = re.search(pattern, stem)
    if not match or match.group(1) != stem[start:end]:
        # Shape generalization could not reproduce the selection; fall back to
        # the literal text around it so the example itself still extracts.
        pattern = "^" + re.escape(stem[:start]) + "(" + re.escape(stem[start:end]) + ")"
    return pattern
