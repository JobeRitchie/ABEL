"""The consolidated summary report: a paginated, print-ready HTML → PDF.

This is deliberately NOT ``report.html``.  That one is the exhaustive dump —
every table, every figure — and it is what you open when you want to dig.  This
one is the thing you hand to a co-author or drop into a supplement: the findings
in words, the key table per analysis, and only the *headline* figure(s).  Every
other figure and the full data tables go to the export bundle instead.

Rendering is done by QtWebEngine's ``printToPdf``, so there is no new dependency
and the PDF gets real typography, tables and page breaks.  The HTML is written
first and always survives, so a machine with no working WebEngine can still
open it and print to PDF from a browser.
"""

from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from abel.validation.findings import KIND_CAVEAT, KIND_WARNING, Finding

# A4 at 96dpi minus 12mm margins ≈ 700px of usable width.
_FIG_W = 700
# Tables longer than this are truncated in the PDF (the full table is in the CSV).
_MAX_TABLE_ROWS = 14


@dataclass(frozen=True)
class FigureSpec:
    """Which figures are the *headline* for a section, and how many to show."""

    subdir: str
    patterns: tuple[str, ...]
    max_n: int = 2


@dataclass(frozen=True)
class TableSpec:
    """The one small table worth printing for a section."""

    subdir: str
    filename: str
    columns: tuple[str, ...] = ()   # empty = all columns
    max_rows: int = _MAX_TABLE_ROWS


@dataclass(frozen=True)
class SectionSpec:
    key: str                        # matches Finding.analysis
    title: str
    figures: FigureSpec | None = None
    table: TableSpec | None = None
    blurb: str = ""


# The report's spine.  Order is the order a reader should meet the evidence:
# what the models score, how much labeling that took, what the features buy,
# what the model still cannot separate, and whether it agrees with a human.
SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "Overview", "Overview",
        figures=FigureSpec("cross_project",
                           ("0_forest_by_behavior.png", "accuracy_bars.png"), 2),
        table=TableSpec("cross_project", "publication_metrics.csv"),
        blurb="Held-out accuracy for every (project × behavior), with the "
              "imbalance-robust metrics (MCC, balanced accuracy, ROC-AUC, κ) that "
              "reviewers of automated-behavior work expect alongside F1.",
    ),
    SectionSpec(
        "Learning curves", "Data efficiency — how many clips do you need?",
        figures=FigureSpec("learning_curves", ("0_AVERAGE__f1_prauc.png",), 1),
        table=TableSpec("learning_curves", "optimal_clips_summary.csv"),
        blurb="Held-out F1 as a function of labeled clips. The knee — 95% of peak "
              "F1 — is the recommended labeling budget.",
    ),
    SectionSpec(
        "Ablation (detection)", "Feature ablation — what does each family add?",
        figures=FigureSpec("ablation", ("feature_impact__*.png",), 3),
        blurb="Baseline is pose-only with every enhancement off. Each bar adds ONE "
              "enhancement on its own; gains are paired per seed. Faded bars have a "
              "95% CI that overlaps zero — a small ± there is noise, not harm.",
    ),
    SectionSpec(
        "Discrimination (pairwise)",
        "Discrimination — can the features tell similar behaviors apart?",
        figures=FigureSpec("discrimination",
                           ("*__separability_matrix__*.png",
                            "*__feature_gain_by_pair.png"), 4),
        table=TableSpec("discrimination", "confusable_pairs.csv", max_rows=10),
        blurb="Ablation asks a DETECTION question (behavior vs. everything else), "
              "where 'everything else' is mostly easy negatives. This asks the "
              "DISCRIMINATION question — for every behavior pair, a binary A-vs-B "
              "model per feature family on the same clips. Pairs the pose baseline "
              "already solves are greyed out.",
    ),
    SectionSpec(
        "Generalization", "Generalization — does it agree with the human scorer?",
        figures=FigureSpec("generalization", ("model_vs_human_kappa.png",), 1),
        table=TableSpec("generalization", "agreement.csv"),
        blurb="Trained on training-pool subjects, scored on held-out subjects the "
              "model never saw, against the reviewer's labels.",
    ),
    SectionSpec(
        "Biological readout", "Biological readout — prevalence & bout agreement",
        figures=FigureSpec("time_budget", ("0_AGREEMENT_FOREST__*.png",), 2),
        table=TableSpec("time_budget", "time_budget_agreement.csv"),
        blurb="Does the model recover the measure a scorer would report? Per-session "
              "prevalence, model vs. reviewed, with Lin's CCC and Bland-Altman limits "
              "of agreement.",
    ),
    SectionSpec(
        "Calibration", "Calibration — do the probabilities mean anything?",
        figures=FigureSpec("calibration", ("*.png",), 2),
        table=TableSpec("calibration", "calibration.csv"),
        blurb="Whether a predicted probability of 0.8 really means right 80% of the "
              "time. Bins holding fewer than 10 held-out segments are drawn hollow.",
    ),
    SectionSpec(
        "Active learning", "Active learning vs. random clip selection",
        figures=FigureSpec("active_learning", ("*.png",), 2),
        table=TableSpec("active_learning", "al_vs_random_summary.csv"),
        blurb="Both arms warm-start from the same seed set and are scored on the same "
              "held-out data; they differ only in which clips get reviewed next.",
    ),
    SectionSpec(
        "Behaviorscape", "Behaviorscape — which feature types drive which behaviors?",
        figures=FigureSpec("behaviorscape",
                           ("behaviorscape_modality_bars.png",
                            "behaviorscape_distinctiveness.png"), 2),
        table=TableSpec("behaviorscape", "behaviorscape_distinctiveness.csv"),
        blurb="Per-behavior feature importance, classified into data modalities, "
              "pooled across projects, with a PERMANOVA test of whether behaviors "
              "genuinely rely on different feature types.",
    ),
    SectionSpec(
        "Video features", "Video-feature value (paired with vs. without)",
        figures=FigureSpec("video_value", ("video_value.png",), 1),
        table=TableSpec("video_value", "video_value.csv"),
        blurb="The same held-out split and the same training subsample, differing ONLY "
              "in the video-feature columns — a clean paired estimate of what the video "
              "motion features add.",
    ),
    SectionSpec(
        "Throughput", "Pipeline throughput",
        figures=FigureSpec("throughput", ("benchmark.png",), 1),
        table=TableSpec("throughput", "benchmark.csv"),
        blurb="Wall-clock time per pipeline stage on one representative session per "
              "project, normalized by the video's true duration.",
    ),
)

_CSS = """
@page { size: A4 portrait; margin: 14mm 12mm 16mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", -apple-system, Roboto, sans-serif; color: #16161d;
       font-size: 10.5pt; line-height: 1.45; margin: 0; }
h1 { font-size: 21pt; margin: 0 0 2mm; letter-spacing: -0.2pt; }
h2 { font-size: 13pt; margin: 0 0 3mm; padding-bottom: 1.5mm;
     border-bottom: 2px solid #2563eb; color: #16161d; }
h3 { font-size: 10.5pt; margin: 4mm 0 1.5mm; color: #475569; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.5pt; }
p  { margin: 0 0 2mm; }
.sub { color: #64748b; font-size: 9pt; margin-bottom: 6mm; }
.meta { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;
        padding: 3mm 4mm; font-size: 9pt; margin-bottom: 6mm; }
.meta b { color: #334155; }
.blurb { color: #64748b; font-size: 9pt; font-style: italic; margin-bottom: 3mm; }

/* A section should not be split across a page break if it can be helped, and a
   figure must never be. */
section { page-break-inside: avoid; margin-bottom: 7mm; }
section.pagebreak { page-break-before: always; }
figure { page-break-inside: avoid; margin: 3mm 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 3px; }

ul.findings { list-style: none; padding: 0; margin: 0 0 3mm; }
ul.findings li { page-break-inside: avoid; margin-bottom: 2.5mm; padding: 2mm 3mm;
                 border-left: 3px solid #2563eb; background: #f8fafc; border-radius: 0 3px 3px 0; }
ul.findings li.caveat  { border-left-color: #d97706; background: #fffbeb; }
ul.findings li.warning { border-left-color: #dc2626; background: #fef2f2; }
.headline { font-weight: 600; display: block; }
.detail { color: #475569; font-size: 8.5pt; display: block; margin-top: 1mm; }
.tag { font-size: 7.5pt; font-weight: 700; text-transform: uppercase;
       letter-spacing: 0.5pt; margin-right: 1.5mm; }
.tag.caveat  { color: #d97706; }
.tag.warning { color: #dc2626; }

table { border-collapse: collapse; font-size: 8pt; width: 100%; margin: 2mm 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #e2e8f0; padding: 1.2mm 2mm; text-align: left; }
th { background: #f1f5f9; font-weight: 600; color: #334155; }
tr:nth-child(even) td { background: #fafafa; }
.trunc { color: #94a3b8; font-size: 8pt; font-style: italic; }
.empty { color: #94a3b8; font-style: italic; font-size: 9pt; }
"""


def _img(path: Path, width: int = _FIG_W) -> str:
    """Embed a PNG as base64 so the HTML is self-contained (and prints offline)."""
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return ""
    return (f'<figure><img src="data:image/png;base64,{b64}" '
            f'style="width:{width}px;max-width:100%;"/></figure>')


def _pick_figures(run_dir: Path, spec: FigureSpec) -> list[Path]:
    """Resolve a section's headline figures, in pattern order, deduped and capped."""
    d = run_dir / spec.subdir
    if not d.is_dir():
        return []
    picked: list[Path] = []
    seen: set[Path] = set()
    for pattern in spec.patterns:
        for p in sorted(d.glob(pattern)):
            if p not in seen and p.is_file():
                picked.append(p)
                seen.add(p)
            if len(picked) >= spec.max_n:
                return picked
    return picked


def headline_figures(run_dir: str | Path) -> list[Path]:
    """Every figure the summary report leads with, in report order.

    The GUI's Run-All tab shows exactly this list, so the on-screen "headline
    figures" and the PDF's cannot drift apart.
    """
    run_dir = Path(run_dir)
    out: list[Path] = []
    for spec in SECTIONS:
        if spec.figures:
            out.extend(_pick_figures(run_dir, spec.figures))
    return out


def _table_html(run_dir: Path, spec: TableSpec) -> str:
    path = run_dir / spec.subdir / spec.filename
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001 — a malformed CSV must not sink the report
        return ""
    if df.empty:
        return ""
    if spec.columns:
        keep = [c for c in spec.columns if c in df.columns]
        if keep:
            df = df[keep]
    total = len(df)
    if total > spec.max_rows:
        df = df.head(spec.max_rows)
    out = df.to_html(index=False, border=0, justify="left",
                     float_format=lambda v: f"{v:.3f}", na_rep="—")
    if total > spec.max_rows:
        out += (f'<p class="trunc">Showing {spec.max_rows} of {total} rows — the full '
                f'table is in <code>{spec.subdir}/{spec.filename}</code>.</p>')
    return out


def _findings_html(items: list[Finding]) -> str:
    if not items:
        return '<p class="empty">No findings for this section.</p>'
    li = []
    for f in items:
        cls = {KIND_CAVEAT: "caveat", KIND_WARNING: "warning"}.get(f.kind, "")
        tag = ""
        if f.kind == KIND_CAVEAT:
            tag = '<span class="tag caveat">Caveat</span>'
        elif f.kind == KIND_WARNING:
            tag = '<span class="tag warning">Warning</span>'
        detail = (f'<span class="detail">{html.escape(f.detail)}</span>'
                  if f.detail else "")
        li.append(f'<li class="{cls}"><span class="headline">{tag}'
                  f'{html.escape(f.headline)}</span>{detail}</li>')
    return '<ul class="findings">' + "".join(li) + "</ul>"


def build_summary_html(
    run_id: str,
    run_dir: Path,
    findings: list[Finding],
    overview: dict,
    project_meta: list[dict],
    save_path: Path | None = None,
) -> Path:
    """Write the print-ready summary HTML. Returns its path."""
    run_dir = Path(run_dir)
    save_path = Path(save_path) if save_path else run_dir / "summary_report.html"

    by_analysis: dict[str, list[Finding]] = {}
    for f in findings:
        by_analysis.setdefault(f.analysis, []).append(f)

    projects_txt = ", ".join(
        f"{p.get('name', '?')} ({len(p.get('behaviors', []))} behaviors)"
        for p in project_meta) or "—"
    meta = (
        f'<div class="meta">'
        f'<b>Run</b>: {html.escape(run_id)}<br/>'
        f'<b>Projects</b>: {html.escape(projects_txt)}<br/>'
        f'<b>Held-out design</b>: whole subjects/sessions reserved before any training; '
        f'reviewer-corrected and imported rows are excluded from the held-out partition '
        f'so a held-out session contributes nothing to training.'
        f'</div>'
    )

    body: list[str] = []
    first = True
    for spec in SECTIONS:
        items = by_analysis.pop(spec.key, [])
        figs = _pick_figures(run_dir, spec.figures) if spec.figures else []
        table = _table_html(run_dir, spec.table) if spec.table else ""
        if not items and not figs and not table:
            continue  # analysis wasn't run
        cls = "" if first else " pagebreak"
        first = False
        parts = [f'<section class="{cls.strip()}"><h2>{html.escape(spec.title)}</h2>']
        if spec.blurb:
            parts.append(f'<p class="blurb">{html.escape(spec.blurb)}</p>')
        parts.append("<h3>Findings</h3>")
        parts.append(_findings_html(items))
        if table:
            parts.append("<h3>Key results</h3>")
            parts.append(table)
        if figs:
            parts.append("<h3>Headline figure"
                         + ("s" if len(figs) > 1 else "") + "</h3>")
            parts.extend(_img(p) for p in figs)
        parts.append("</section>")
        body.append("".join(parts))

    # Anything the spec didn't anticipate still gets its findings printed.
    for analysis, items in by_analysis.items():
        body.append(f'<section class="pagebreak"><h2>{html.escape(analysis)}</h2>'
                    f'{_findings_html(items)}</section>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ABEL Validation Report — {html.escape(run_id)}</title>
<style>{_CSS}</style></head><body>
<h1>ABEL Validation &amp; Meta-Analysis</h1>
<p class="sub">Consolidated summary report · {html.escape(run_id)}</p>
{meta}
{"".join(body)}
</body></html>"""
    save_path.write_text(doc, encoding="utf-8")
    return save_path


# ── HTML → PDF ──────────────────────────────────────────────────────────────

def render_pdf(html_path: Path, pdf_path: Path, timeout_ms: int = 60_000) -> Path:
    """Render ``html_path`` to ``pdf_path`` with QtWebEngine.

    MUST be called from the GUI thread: QtWebEngine needs the Qt event loop.  A
    nested :class:`QEventLoop` is used so this can be called synchronously from a
    button handler while the app's own loop is running.

    Raises ``RuntimeError`` if WebEngine is unavailable or the render fails — the
    caller should fall back to pointing the user at the HTML.
    """
    html_path, pdf_path = Path(html_path), Path(pdf_path)
    try:
        from PySide6.QtCore import QEvent, QEventLoop, QMarginsF, QTimer, QUrl
        from PySide6.QtGui import QPageLayout, QPageSize
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "QtWebEngine is not available, so the PDF could not be rendered. "
            f"The summary HTML is at {html_path} — open it and print to PDF."
        ) from exc

    if QApplication.instance() is None:
        raise RuntimeError("render_pdf needs a running QApplication (call it from the GUI).")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    view = QWebEngineView()
    loop = QEventLoop()
    state: dict = {"error": None}

    def _on_pdf(data: bytes) -> None:
        # printToPdf hands back empty bytes on failure rather than raising.
        if not data:
            state["error"] = "QtWebEngine produced an empty PDF."
        else:
            pdf_path.write_bytes(data)
        loop.quit()

    def _on_load(ok: bool) -> None:
        if not ok:
            state["error"] = f"Could not load {html_path}."
            loop.quit()
            return
        layout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(12, 14, 12, 16),
            QPageLayout.Unit.Millimeter,
        )
        view.page().printToPdf(_on_pdf, layout)

    def _on_timeout() -> None:
        state["error"] = "Timed out rendering the PDF."
        loop.quit()

    view.loadFinished.connect(_on_load)
    QTimer.singleShot(timeout_ms, _on_timeout)
    # Load from disk rather than setHtml(): setHtml caps content at ~2MB, and a
    # report with base64-embedded figures blows straight past that.
    view.load(QUrl.fromLocalFile(str(html_path.resolve())))
    loop.exec()

    # Tear the WebEngine view down NOW, not whenever the deferred-delete queue
    # happens to be drained.  deleteLater() alone leaves the view (and its render
    # process) alive until control returns to an event loop; a headless caller that
    # never runs one exits with QApplication being destroyed while WebEngine is
    # still up, and the process segfaults on the way out.
    #
    # Flush ONLY the deferred deletes — not processEvents(), which would also
    # deliver queued input events and could re-enter a run from a click that landed
    # while this was rendering.
    view.loadFinished.disconnect()
    view.page().deleteLater()
    view.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    if state["error"]:
        raise RuntimeError(f"{state['error']} The summary HTML is at {html_path} — "
                           f"open it and print to PDF.")
    return pdf_path
