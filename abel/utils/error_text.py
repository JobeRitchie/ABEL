"""Turn a worker traceback into something a user can act on.

Worker threads hand the UI a full traceback string.  Truncating it from the
front (``tb[:600]``) keeps the call frames and cuts off the exception message —
the one line that says what actually went wrong.  These helpers surface the
message first and keep the frames as supporting detail.
"""

from __future__ import annotations

__all__ = ["exception_message", "format_task_error"]


def exception_message(traceback_text: str) -> str:
    """Extract the trailing ``ExcType: message`` from a formatted traceback.

    The message may span several lines (an exception whose text is a list, say),
    so everything from the exception line to the end is returned.  Falls back to
    the whole input when it does not look like a traceback.
    """
    lines = (traceback_text or "").rstrip().splitlines()
    last_frame = -1
    for i, line in enumerate(lines):
        if line.lstrip().startswith('File "'):
            last_frame = i
    for j in range(last_frame + 1, len(lines)):
        line = lines[j]
        # The exception starts at the first column-0 line after the last frame;
        # source lines under a frame are always indented.
        if line and not line[0].isspace():
            return "\n".join(lines[j:]).strip()
    return (traceback_text or "").strip()


def format_task_error(traceback_text: str, *, frames_chars: int = 600) -> str:
    """Message-first rendering of a traceback for a log pane or dialog.

    The exception message is never truncated; the call frames below it are
    capped at ``frames_chars`` since they exist only to locate the failure.
    """
    msg = exception_message(traceback_text)
    frames = (traceback_text or "").strip()
    if not frames or frames == msg:
        return msg
    if len(frames) > frames_chars:
        frames = frames[:frames_chars].rstrip() + "\n  … (full traceback in the log file)"
    return f"{msg}\n\n{frames}"
