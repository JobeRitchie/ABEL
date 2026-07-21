"""CSV + self-contained HTML roll-up for a validation run."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd


def _img_tag(path: Path, width: int = 760) -> str:
    """Embed a PNG as a base64 <img> so the report is self-contained."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:{width}px;width:100%;margin:8px 0;border:1px solid #ddd;border-radius:6px;"/>'


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><em>(no data)</em></p>"
    return df.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0,
                      classes="vtab", justify="left")


def build_html(
    run_id: str,
    overview: dict,
    sections: list[tuple[str, str]],
    save_path: Path,
) -> Path:
    """Assemble the report. ``sections`` = list of (heading, inner_html)."""
    ov = " &nbsp;|&nbsp; ".join(f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
                                for k, v in overview.items())
    body = "\n".join(
        f'<section><h2>{html.escape(h)}</h2>{inner}</section>' for h, inner in sections
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ABEL Validation — {html.escape(run_id)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a2e;background:#fafafc;}}
 h1{{font-size:22px;}} h2{{font-size:17px;margin-top:28px;border-bottom:2px solid #2196F3;padding-bottom:4px;}}
 .overview{{background:#fff;padding:12px 14px;border-radius:8px;border:1px solid #e0e0e8;}}
 table.vtab{{border-collapse:collapse;font-size:13px;margin:8px 0;}}
 table.vtab td,table.vtab th{{border:1px solid #e0e0e8;padding:4px 8px;text-align:left;}}
 table.vtab th{{background:#eef3fb;}}
 section{{background:#fff;padding:14px 18px;border-radius:8px;border:1px solid #e8e8ee;margin:14px 0;}}
</style></head><body>
<h1>ABEL Validation &amp; Meta-Analysis — {html.escape(run_id)}</h1>
<div class="overview">{ov}</div>
{body}
</body></html>"""
    save_path = Path(save_path)
    save_path.write_text(doc, encoding="utf-8")
    return save_path


def img_section(paths: list[Path]) -> str:
    return "\n".join(_img_tag(p) for p in paths if p and Path(p).exists())


def table_section(df: pd.DataFrame) -> str:
    return _table(df)
