"""Report generation — HTML and CSV export for ablation benchmark results."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from abel.benchmark.metrics import (
    apply_behavior_names,
    compute_deltas,
    load_behavior_names,
    rank_features_by_impact,
    results_to_dataframe,
)
from abel.benchmark.runner import RunResult


def export_csv(results: list[RunResult], output_path: Path) -> Path:
    """Write raw metric summary as CSV."""
    df = results_to_dataframe(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _img_to_base64(path: Path) -> str:
    """Encode an image file as a base64 data URI."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def export_html(
    results: list[RunResult],
    output_path: Path,
    plot_dir: Path | None = None,
    project_name: str = "",
    behavior_names: dict[str, str] | None = None,
) -> Path:
    """Generate a self-contained HTML report with embedded plots.

    Results may span multiple behaviors; the report shows per-behavior tables.
    """
    from abel.benchmark.metrics import format_mean_sem

    # Resolve behaviour names from project if not provided
    if not behavior_names and results:
        try:
            project_root = Path(results[0].overrides.get("_project_root", ""))
            if not project_root.exists():
                # Attempt to infer from output_path
                project_root = output_path.parent.parent.parent
            behavior_names = load_behavior_names(project_root)
        except Exception:
            behavior_names = {}

    df = results_to_dataframe(results)
    if behavior_names:
        df = apply_behavior_names(df, behavior_names)
    behaviors = sorted(df["Behavior"].unique().tolist())

    # ── Embed plot images ─────────────────────────────────────────
    plot_images: list[tuple[str, str]] = []
    if plot_dir and plot_dir.exists():
        for png in sorted(plot_dir.glob("ablation_*.png")):
            b64 = _img_to_base64(png)
            if b64:
                title = png.stem.replace("ablation_", "").replace("_", " ").title()
                plot_images.append((title, b64))

    # ── Build HTML ────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"ABEL Ablation Report — {project_name or 'Project'}"

    def _df_to_html(frame: pd.DataFrame, table_id: str = "") -> str:
        if frame.empty:
            return "<p><em>No data.</em></p>"
        html = frame.to_html(
            index=False, border=0, classes="data-table", table_id=table_id,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
        return html

    css = """
    <style>
        :root { --bg: #1e1e2e; --fg: #cdd6f4; --accent: #89b4fa; --surface: #313244;
                --red: #f38ba8; --green: #a6e3a1; --border: #45475a; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
               color: var(--fg); margin: 0; padding: 20px 40px; }
        h1 { color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 8px; }
        h2 { color: var(--accent); margin-top: 36px; }
        h3 { color: #b4befe; }
        .meta { color: #a6adc8; font-size: 0.9em; margin-bottom: 24px; }
        .data-table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0;
                      font-size: 0.88em; }
        .data-table th { background: var(--surface); color: var(--accent); padding: 8px 12px;
                         text-align: left; border-bottom: 2px solid var(--border); }
        .data-table td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
        .data-table tr:hover td { background: var(--surface); }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container img { max-width: 100%; border-radius: 8px;
                              box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
        .summary-card { background: var(--surface); border-radius: 10px; padding: 16px 24px;
                        margin: 16px 0; border-left: 4px solid var(--accent); }
        .positive { color: var(--green); } .negative { color: var(--red); }
    </style>
    """

    # ── Per-behavior sections ─────────────────────────────────────
    behavior_sections = ""
    for behavior in behaviors:
        bdf = df[df["Behavior"] == behavior].copy()
        beh_label = behavior[:24] if behavior else "(all)"

        # Formatted table with mean ± SEM
        display_rows = []
        for _, row in bdf.iterrows():
            display_rows.append({
                "Run": row["Run"],
                "Precision": format_mean_sem(row["Precision"], row["Precision SEM"]),
                "Recall": format_mean_sem(row["Recall"], row["Recall SEM"]),
                "F1": format_mean_sem(row["F1"], row["F1 SEM"]),
                "PR-AUC": format_mean_sem(row["PR-AUC"], row["PR-AUC SEM"]),
                "Folds": int(row["Folds"]),
                "Features": int(row["Features"]) if pd.notna(row["Features"]) else "",
                "Time (s)": row["Time (s)"],
            })
        display_df = pd.DataFrame(display_rows)

        deltas = compute_deltas(df, behavior=behavior)
        ranked = rank_features_by_impact(bdf)

        # Summary card
        baseline = bdf[bdf["Run"] == "baseline_all_on"]
        all_off = bdf[bdf["Run"] == "baseline_all_off"]
        summary_card = ""
        if not baseline.empty and not all_off.empty:
            b_f1 = float(baseline.iloc[0]["F1"])
            o_f1 = float(all_off.iloc[0]["F1"])
            diff = b_f1 - o_f1
            cls = "positive" if diff > 0 else "negative"
            summary_card = f"""
            <div class="summary-card">
                <strong>Pipeline Impact — {beh_label}</strong><br>
                All ON: F1 = <strong>{b_f1:.4f}</strong> &nbsp;|&nbsp;
                All OFF: F1 = <strong>{o_f1:.4f}</strong> &nbsp;|&nbsp;
                Net: <span class="{cls}"><strong>{diff:+.4f}</strong></span>
            </div>
            """

        behavior_sections += f"""
        <h2>Behavior: {beh_label}</h2>
        {summary_card}
        <h3>Metric Summary (mean ± SEM)</h3>
        {_df_to_html(display_df, f"metrics_{behavior[:8]}")}
        <h3>Feature Impact (Δ vs. Baseline)</h3>
        {_df_to_html(deltas, f"deltas_{behavior[:8]}")}
        <h3>Feature Importance Ranking</h3>
        {_df_to_html(ranked, f"ranking_{behavior[:8]}")}
        """

    plots_section = ""
    for plot_title, b64_data in plot_images:
        plots_section += f"""
        <div class="plot-container">
            <h3>{plot_title}</h3>
            <img src="{b64_data}" alt="{plot_title}">
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
    <h1>{title}</h1>
    <div class="meta">
        Generated: {now} &nbsp;|&nbsp;
        Behaviors: {len(behaviors)} &nbsp;|&nbsp;
        Runs per behavior: {len(df) // max(1, len(behaviors))}
    </div>

    {behavior_sections}

    <h2>Visualizations</h2>
    {plots_section if plots_section else '<p><em>No plots available. Install matplotlib to generate visualizations.</em></p>'}

    <hr style="border-color: var(--border); margin-top: 40px;">
    <p style="color: #585b70; font-size: 0.8em; text-align: center;">
        Generated by ABEL Ablation Benchmark Suite
    </p>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
