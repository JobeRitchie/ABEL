"""Publication-quality UMAP QC plot functions for Keypoint-MoSeq syllable analysis.

All functions return a ``matplotlib.figure.Figure`` ready for display or export.
They are pure-function with no GUI or file I/O dependencies so they can be
called from the background thread or unit tests without a running Qt event loop.

Plot types provided
-------------------
plot_standard_umap         — coloured scatter, centroid labels, optional contours
plot_density_umap          — hexbin density map with centroid overlay
plot_syllable_highlights   — per-syllable panel grid (one highlighted syllable per cell)
plot_compactness_chart     — ranked bar / dot chart of within-cluster spread
plot_transition_graph      — directed syllable transition graph
build_qc_dashboard         — composite multi-panel summary figure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("abel")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass
class ThemeSettings:
    """Colour palette for dark or light QC figures."""

    # Figure / axes backgrounds
    fig_bg: str = "#0d1117"
    ax_bg: str = "#0d1117"

    # Text colours
    title_color: str = "#ECEFF1"
    label_color: str = "#B0BEC5"
    tick_color: str = "#546E7A"
    point_label_color: str = "white"

    # Axis decorations
    spine_color: str = "#37474F"
    grid_color: str = "#263238"

    # Legend
    legend_bg: str = "#1a2333"
    legend_edge: str = "#37474F"
    legend_text: str = "#ECEFF1"

    # Metrics bar colours
    bar_ok: str = "#42A5F5"
    bar_warn: str = "#FFA726"
    bar_bad: str = "#EF5350"

    # Transition graph
    node_cmap: str = "turbo"
    edge_color: str = "#546E7A"


DARK_THEME = ThemeSettings()

LIGHT_THEME = ThemeSettings(
    fig_bg="white",
    ax_bg="#F8F9FA",
    title_color="#212121",
    label_color="#424242",
    tick_color="#757575",
    point_label_color="#212121",
    spine_color="#BDBDBD",
    grid_color="#E0E0E0",
    legend_bg="white",
    legend_edge="#BDBDBD",
    legend_text="#212121",
    bar_ok="#1565C0",
    bar_warn="#E65100",
    bar_bad="#B71C1C",
    edge_color="#9E9E9E",
)


def get_theme(dark: bool) -> ThemeSettings:
    return DARK_THEME if dark else LIGHT_THEME


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------


def get_syllable_palette(n: int, cmap_name: str = "turbo") -> list:
    """Return *n* RGBA colours spaced across a matplotlib colourmap."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    cmap = plt.colormaps.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _auto_alpha(n_points: int) -> float:
    """Choose marker alpha so dense clouds do not over-saturate."""
    if n_points < 5_000:
        return 0.55
    if n_points < 20_000:
        return 0.35
    if n_points < 100_000:
        return 0.18
    return 0.08


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


def compute_centroids(
    xy: np.ndarray, labels: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Return {syllable_id: (cx, cy)} mean-centroid for each syllable."""
    centroids: dict[int, tuple[float, float]] = {}
    unique = np.unique(labels)
    for sid in unique:
        if sid < 0:
            continue
        mask = labels == sid
        if not mask.any():
            continue
        centroids[int(sid)] = (float(xy[mask, 0].mean()), float(xy[mask, 1].mean()))
    return centroids


def compute_medoid_centroids(
    xy: np.ndarray, labels: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Return {syllable_id: (cx, cy)} medoid (most central actual point)."""
    centroids: dict[int, tuple[float, float]] = {}
    unique = np.unique(labels)
    for sid in unique:
        if sid < 0:
            continue
        mask = labels == sid
        pts = xy[mask]
        if len(pts) == 0:
            continue
        mean_pt = pts.mean(axis=0)
        dists = np.linalg.norm(pts - mean_pt, axis=1)
        medoid = pts[np.argmin(dists)]
        centroids[int(sid)] = (float(medoid[0]), float(medoid[1]))
    return centroids


# ---------------------------------------------------------------------------
# Label placement helpers
# ---------------------------------------------------------------------------


def _place_labels_simple(
    ax,
    centroids: dict[int, tuple[float, float]],
    theme: ThemeSettings,
    fontsize: float = 6.5,
) -> None:
    """Place syllable ID labels at centroid positions."""
    for sid, (cx, cy) in centroids.items():
        ax.text(
            cx, cy,
            str(sid),
            fontsize=fontsize,
            color=theme.point_label_color,
            ha="center",
            va="center",
            fontweight="bold",
            alpha=0.92,
            zorder=6,
        )


def _place_labels_with_repel(
    ax,
    centroids: dict[int, tuple[float, float]],
    theme: ThemeSettings,
    fontsize: float = 6.5,
) -> None:
    """Use adjustText if available for collision-aware label placement."""
    try:
        from adjustText import adjust_text  # type: ignore[import]  # noqa: PLC0415

        texts = []
        for sid, (cx, cy) in centroids.items():
            t = ax.text(
                cx, cy,
                str(sid),
                fontsize=fontsize,
                color=theme.point_label_color,
                ha="center",
                va="center",
                fontweight="bold",
                alpha=0.92,
                zorder=6,
            )
            texts.append(t)

        adjust_text(
            texts,
            ax=ax,
            force_text=(0.2, 0.5),
            arrowprops={"arrowstyle": "-", "color": "#546E7A", "lw": 0.5, "alpha": 0.5},
        )
    except ImportError:
        # Fallback to simple placement when adjustText is not installed.
        _place_labels_simple(ax, centroids, theme, fontsize)


def place_labels(
    ax,
    centroids: dict[int, tuple[float, float]],
    theme: ThemeSettings,
    *,
    repel: bool = True,
    fontsize: float | None = None,
) -> None:
    n = len(centroids)
    fs = fontsize if fontsize is not None else (5.0 if n > 40 else 6.5)
    if repel:
        _place_labels_with_repel(ax, centroids, theme, fontsize=fs)
    else:
        _place_labels_simple(ax, centroids, theme, fontsize=fs)


# ---------------------------------------------------------------------------
# Axes styling utility
# ---------------------------------------------------------------------------


def _style_ax(ax, theme: ThemeSettings, title: str = "", xlabel: str = "UMAP 1", ylabel: str = "UMAP 2") -> None:
    ax.set_facecolor(theme.ax_bg)
    ax.set_xlabel(xlabel, color=theme.label_color, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=theme.label_color, fontsize=10, labelpad=8)
    ax.set_title(title, color=theme.title_color, fontsize=11, pad=10)
    ax.tick_params(colors=theme.tick_color, labelsize=7, length=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# 1. Standard syllable UMAP
# ---------------------------------------------------------------------------


def plot_standard_umap(
    xy: np.ndarray,
    labels: np.ndarray,
    theme: ThemeSettings | None = None,
    *,
    include_labels: bool = True,
    repel_labels: bool = True,
    contours: bool = False,
    figsize: tuple[float, float] = (11, 9),
    dpi: int = 150,
    title_extra: str = "",
    cmap: str = "turbo",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Publication-quality syllable UMAP scatter plot.

    Parameters
    ----------
    xy:             (N, 2) UMAP embedding coordinates.
    labels:         (N,) integer syllable IDs (0-based).
    theme:          ThemeSettings; if None, dark theme is used.
    include_labels: Draw syllable numbers at cluster centroids.
    repel_labels:   Use adjustText (if installed) for non-overlapping labels.
    contours:       Draw KDE contours around dense regions.
    """
    import matplotlib.figure as mfig  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME

    n_syllables = int(labels.max()) + 1
    palette = get_syllable_palette(n_syllables, cmap)

    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)
    ax = fig.add_subplot(111)

    alpha = _auto_alpha(len(xy))
    label_arr = labels.astype(int)

    for syl_id in range(n_syllables):
        mask = label_arr == syl_id
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=[palette[syl_id]],
            s=1.5,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if contours:
        _add_contours(ax, xy, labels, n_syllables, palette, theme)

    centroids = compute_centroids(xy, labels)
    if include_labels and centroids:
        place_labels(ax, centroids, theme, repel=repel_labels)

    # Legend (only manageable sizes get inline legend; else skip)
    if n_syllables <= 30:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=palette[i],
                markersize=5,
                label=f"Syllable {i}",
            )
            for i in range(n_syllables)
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=5.5,
            framealpha=0.25,
            facecolor=theme.legend_bg,
            edgecolor=theme.legend_edge,
            labelcolor=theme.legend_text,
            ncol=max(1, n_syllables // 15),
            markerscale=1.4,
        )

    subtitle = f"{n_syllables} syllables · {len(xy):,} frames"
    if title_extra:
        subtitle = f"{subtitle} · {title_extra}"
    _style_ax(ax, theme, title=subtitle)
    fig.tight_layout(pad=1.4)
    return fig


def _add_contours(ax, xy, labels, n_syllables, palette, theme):
    """Overlay light KDE contours for each syllable."""
    try:
        from scipy.stats import gaussian_kde  # noqa: PLC0415
    except ImportError:
        return

    label_arr = labels.astype(int)
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    xg, yg = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]

    for syl_id in range(n_syllables):
        mask = label_arr == syl_id
        pts = xy[mask]
        if len(pts) < 10:
            continue
        try:
            kde = gaussian_kde(pts.T)
            z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
            ax.contour(xg, yg, z, levels=2, colors=[palette[syl_id]], alpha=0.35, linewidths=0.8)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Density UMAP
# ---------------------------------------------------------------------------


def plot_density_umap(
    xy: np.ndarray,
    labels: np.ndarray,
    theme: ThemeSettings | None = None,
    *,
    mode: str = "hexbin",
    overlay_centroids: bool = True,
    include_labels: bool = True,
    figsize: tuple[float, float] = (11, 9),
    dpi: int = 150,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Density-based UMAP visualisation (hexbin, 2D histogram, or KDE).

    Parameters
    ----------
    mode:  'hexbin' | 'hist2d' | 'kde'
    """
    import matplotlib.figure as mfig  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME

    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)
    ax = fig.add_subplot(111)

    if mode == "hexbin":
        hb = ax.hexbin(
            xy[:, 0],
            xy[:, 1],
            gridsize=60,
            cmap="inferno" if theme.fig_bg.startswith("#0") else "hot",
            mincnt=1,
            linewidths=0.1,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Frame count", color=theme.label_color, fontsize=8)
        cbar.ax.tick_params(colors=theme.tick_color, labelsize=7)

    elif mode == "hist2d":
        bins = 80
        h, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
        ax.imshow(
            h.T,
            origin="lower",
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            cmap="inferno",
            aspect="auto",
            interpolation="bilinear",
        )

    elif mode == "kde":
        try:
            from scipy.stats import gaussian_kde  # noqa: PLC0415
            k = 150
            xs = np.linspace(xy[:, 0].min(), xy[:, 0].max(), k)
            ys = np.linspace(xy[:, 1].min(), xy[:, 1].max(), k)
            xg, yg = np.meshgrid(xs, ys)
            # Subsample for KDE tractability
            n_kde = min(len(xy), 15_000)
            rng = np.random.default_rng(42)
            idx = rng.choice(len(xy), n_kde, replace=False)
            kde = gaussian_kde(xy[idx].T, bw_method=0.15)
            z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
            ax.contourf(xg, yg, z, levels=18, cmap="inferno")
        except ImportError:
            # Fallback to hexbin when scipy is absent
            ax.hexbin(xy[:, 0], xy[:, 1], gridsize=60, cmap="inferno", mincnt=1)
    else:
        ax.hexbin(xy[:, 0], xy[:, 1], gridsize=60, cmap="inferno", mincnt=1)

    if overlay_centroids or include_labels:
        n_syllables = int(labels.max()) + 1
        palette = get_syllable_palette(n_syllables)
        centroids = compute_centroids(xy, labels)

        if overlay_centroids:
            for syl_id, (cx, cy) in centroids.items():
                ax.scatter(
                    cx, cy,
                    c=[palette[syl_id % len(palette)]],
                    s=30,
                    marker="*",
                    linewidths=0.5,
                    edgecolors="white",
                    zorder=5,
                )

        if include_labels:
            place_labels(ax, centroids, theme, repel=False)

    _style_ax(ax, theme, title=f"Density UMAP · {len(xy):,} frames ({mode})")
    fig.tight_layout(pad=1.4)
    return fig


# ---------------------------------------------------------------------------
# 3. Per-syllable highlight panels
# ---------------------------------------------------------------------------


def plot_syllable_highlights(
    xy: np.ndarray,
    labels: np.ndarray,
    metrics: list[dict[str, Any]] | None = None,
    theme: ThemeSettings | None = None,
    *,
    n_cols: int = 6,
    panel_size: float = 2.8,
    cmap: str = "turbo",
    dpi: int = 150,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """One panel per syllable: grey background + highlight + compact stats.

    Parameters
    ----------
    metrics:   List of per-syllable metric dicts (optional); used for captions.
    """
    import matplotlib.figure as mfig  # noqa: PLC0415
    import matplotlib.gridspec as gridspec  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME

    n_syllables = int(labels.max()) + 1
    palette = get_syllable_palette(n_syllables, cmap)
    metrics_by_id: dict[int, dict[str, Any]] = {}
    if metrics:
        for m in metrics:
            sid = int(m.get("syllable_id", -1))
            if sid >= 0:
                metrics_by_id[sid] = m

    # Order syllables by n_bouts descending; fall back to syllable ID when no metrics
    ordered_syl_ids = sorted(
        range(n_syllables),
        key=lambda sid: -metrics_by_id.get(sid, {}).get("n_bouts", 0),
    )

    n_rows = int(np.ceil(n_syllables / n_cols))
    fig_w = panel_size * n_cols
    fig_h = panel_size * n_rows + 0.4

    fig = mfig.Figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.15)

    # Sub-sample background points for speed (max 20 k)
    bg_cap = 20_000
    if len(xy) > bg_cap:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(len(xy), bg_cap, replace=False)
        bg_xy = xy[bg_idx]
    else:
        bg_xy = xy

    label_arr = labels.astype(int)
    centroids = compute_centroids(xy, labels)

    for panel_idx, syl_id in enumerate(ordered_syl_ids):
        row, col = divmod(panel_idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(theme.ax_bg)

        # Background cloud (grey)
        ax.scatter(
            bg_xy[:, 0],
            bg_xy[:, 1],
            c=theme.tick_color,
            s=0.4,
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )

        # Highlighted syllable
        mask = label_arr == syl_id
        if mask.any():
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=[palette[syl_id]],
                s=1.5,
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
            cx, cy = centroids.get(syl_id, (float(xy[mask, 0].mean()), float(xy[mask, 1].mean())))
            ax.scatter(cx, cy, c="white", s=18, marker="*", zorder=5, linewidths=0)

        # Compact caption
        m = metrics_by_id.get(syl_id, {})
        occ = m.get("occupancy_fraction", 0)
        n_bouts = m.get("n_bouts", "?")
        mbl = m.get("mean_bout_length", 0)
        caption = f"S{syl_id}  occ={occ:.1%}  bouts={n_bouts}"
        if mbl:
            caption += f"  mbl={mbl:.1f}"
        ax.set_title(caption, fontsize=4.5, color=theme.title_color, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for extra in range(n_syllables, n_rows * n_cols):
        row, col = divmod(extra, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    fig.suptitle(
        f"Per-Syllable Highlights  ·  {n_syllables} syllables",
        color=theme.title_color,
        fontsize=10,
        y=1.0,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Compactness / QC metrics chart
# ---------------------------------------------------------------------------


def plot_compactness_chart(
    metrics: list[dict[str, Any]],
    theme: ThemeSettings | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Ranked horizontal bar chart of within-cluster spread + occupancy."""
    import matplotlib.figure as mfig  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME
    if not metrics:
        fig = mfig.Figure(figsize=(6, 2), dpi=dpi)
        fig.patch.set_facecolor(theme.fig_bg)
        return fig

    # Sort by compactness (lower spread = more compact = better)
    sorted_m = sorted(metrics, key=lambda m: m.get("within_cluster_spread", 1e9))
    sids = [f"S{m['syllable_id']}" for m in sorted_m]
    spread = [float(m.get("within_cluster_spread", 0)) for m in sorted_m]
    occupancy = [float(m.get("occupancy_fraction", 0)) for m in sorted_m]

    n = len(sids)
    fs_y = max(3.0, min(8.0, 200 / max(n, 1)))
    if figsize is None:
        figsize = (10, max(3.5, n * 0.22 + 1.2))

    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Spread bars
    max_s = max(spread) if spread else 1
    spread_colors = [
        theme.bar_bad if s > 0.7 * max_s
        else (theme.bar_warn if s > 0.4 * max_s else theme.bar_ok)
        for s in spread
    ]
    y = list(range(n))
    ax1.barh(y, spread, color=spread_colors, edgecolor=theme.ax_bg, linewidth=0.3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sids, fontsize=fs_y, color=theme.label_color)
    ax1.set_xlabel("Within-cluster spread (lower = more compact)", color=theme.label_color, fontsize=8)
    ax1.set_title("Compactness", color=theme.title_color, fontsize=9)
    ax1.set_facecolor(theme.ax_bg)
    ax1.tick_params(colors=theme.tick_color, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(theme.spine_color)

    # Occupancy bars
    ax2.barh(y, occupancy, color=theme.bar_ok, edgecolor=theme.ax_bg, linewidth=0.3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sids, fontsize=fs_y, color=theme.label_color)
    ax2.set_xlabel("Occupancy fraction", color=theme.label_color, fontsize=8)
    ax2.set_title("Occupancy", color=theme.title_color, fontsize=9)
    ax2.set_facecolor(theme.ax_bg)
    ax2.tick_params(colors=theme.tick_color, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor(theme.spine_color)

    fig.tight_layout(pad=1.4)
    return fig


# ---------------------------------------------------------------------------
# 5. Transition graph
# ---------------------------------------------------------------------------


def plot_transition_graph(
    trans_matrix: np.ndarray,
    occupancy: np.ndarray | None = None,
    theme: ThemeSettings | None = None,
    *,
    threshold: float = 0.02,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 150,
    cmap: str = "turbo",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Directed syllable transition graph using networkx or matplotlib arrows.

    Node size  = occupancy.
    Edge width = transition probability.
    Edges below *threshold* are hidden.
    """
    import matplotlib.figure as mfig  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME

    n = trans_matrix.shape[0]
    if occupancy is None:
        occupancy = np.ones(n) / n

    palette = get_syllable_palette(n, cmap)

    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.ax_bg)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    try:
        import networkx as nx  # type: ignore[import]  # noqa: PLC0415
        _draw_nx_graph(ax, trans_matrix, occupancy, palette, theme, threshold, n)
    except ImportError:
        logger.info("networkx not installed — using circular layout fallback for transition graph.")
        _draw_circle_graph(ax, trans_matrix, occupancy, palette, theme, threshold, n)

    ax.set_title(
        f"Syllable Transitions  ·  {n} nodes  ·  threshold={threshold:.2f}",
        color=theme.title_color, fontsize=11, pad=10,
    )
    fig.tight_layout(pad=1.4)
    return fig


def _draw_nx_graph(ax, trans_matrix, occupancy, palette, theme, threshold, n):
    import networkx as nx  # noqa: PLC0415

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    edges = []
    widths = []
    for i in range(n):
        row_sum = trans_matrix[i].sum()
        for j in range(n):
            if i == j:
                continue
            prob = trans_matrix[i, j] / max(row_sum, 1e-12)
            if prob >= threshold:
                G.add_edge(i, j, weight=float(prob))
                edges.append((i, j))
                widths.append(float(prob))

    pos = nx.spring_layout(G, seed=42, k=2.5 / max(np.sqrt(n), 1))

    # Draw edges
    if edges and widths:
        max_w = max(widths)
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            width=[w / max_w * 3.5 for w in widths],
            edge_color=theme.edge_color,
            alpha=0.65,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
        )

    # Draw nodes
    node_sizes = [max(150, float(occupancy[i]) * 4000) for i in range(n)]
    node_colors = [palette[i] for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5, font_color=theme.point_label_color)


def _draw_circle_graph(ax, trans_matrix, occupancy, palette, theme, threshold, n):
    """Fallback: nodes on a circle, matplotlib arrows for edges."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    # Edges first (underneath)
    for i in range(n):
        row_sum = trans_matrix[i].sum()
        for j in range(n):
            if i == j:
                continue
            prob = trans_matrix[i, j] / max(row_sum, 1e-12)
            if prob < threshold:
                continue
            lw = max(0.3, prob * 6.0)
            ax.annotate(
                "",
                xy=(xs[j], ys[j]),
                xytext=(xs[i], ys[i]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": theme.edge_color,
                    "lw": lw,
                    "alpha": 0.6,
                    "connectionstyle": "arc3,rad=0.12",
                },
                zorder=1,
            )

    # Nodes
    for i in range(n):
        size = 80 + float(occupancy[i]) * 3000
        ax.scatter(xs[i], ys[i], s=size, c=[palette[i]], zorder=3, linewidths=0.5, edgecolors="white")
        ax.text(
            xs[i] * 1.14, ys[i] * 1.14,
            str(i),
            fontsize=5.5, color=theme.point_label_color,
            ha="center", va="center", fontweight="bold",
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)


# ---------------------------------------------------------------------------
# 6. Composite QC dashboard
# ---------------------------------------------------------------------------


def build_qc_dashboard(
    xy: np.ndarray,
    labels: np.ndarray,
    metrics: list[dict[str, Any]] | None = None,
    trans_matrix: np.ndarray | None = None,
    theme: ThemeSettings | None = None,
    *,
    figsize: tuple[float, float] = (22, 14),
    dpi: int = 150,
    cmap: str = "turbo",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Single composite dashboard figure containing:

    Row 1:  Standard UMAP  |  Density UMAP
    Row 2:  Compactness ranking  |  Occupancy (embedded in compactness fig)  |  Transition graph
    """
    import matplotlib.figure as mfig  # noqa: PLC0415
    import matplotlib.gridspec as gridspec  # noqa: PLC0415

    if theme is None:
        theme = DARK_THEME

    n_syllables = int(labels.max()) + 1
    palette = get_syllable_palette(n_syllables, cmap)
    centroids = compute_centroids(xy, labels)

    fig = mfig.Figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.fig_bg)

    # Grid: 2 rows × 3 cols; row 1 spans cols 0-1 and 1-2, row 2 splits 3 ways
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 0.85],
        hspace=0.38, wspace=0.28,
    )

    # ── Cell (0,0): Standard UMAP ─────────────────────────────────────
    ax_umap = fig.add_subplot(gs[0, :2])
    ax_umap.set_facecolor(theme.ax_bg)
    alpha = _auto_alpha(len(xy))
    label_arr = labels.astype(int)
    for syl_id in range(n_syllables):
        mask = label_arr == syl_id
        if not mask.any():
            continue
        ax_umap.scatter(
            xy[mask, 0], xy[mask, 1],
            c=[palette[syl_id]], s=1.2, alpha=alpha, linewidths=0, rasterized=True,
        )
    place_labels(ax_umap, centroids, theme, repel=False, fontsize=5.5)
    _style_ax(ax_umap, theme, title=f"Syllable UMAP · {n_syllables} syllables · {len(xy):,} frames")

    # ── Cell (0,2): Density UMAP (hexbin) ─────────────────────────────
    ax_dens = fig.add_subplot(gs[0, 2])
    ax_dens.set_facecolor(theme.ax_bg)
    hb = ax_dens.hexbin(xy[:, 0], xy[:, 1], gridsize=50, cmap="inferno", mincnt=1, linewidths=0.1)
    for syl_id, (cx, cy) in centroids.items():
        ax_dens.scatter(cx, cy, c="white", s=15, marker="*", zorder=5, linewidths=0)
    _style_ax(ax_dens, theme, title="Density (hexbin)")

    # ── Cell (1,0): Compactness bars ──────────────────────────────────
    ax_compact = fig.add_subplot(gs[1, 0])
    ax_compact.set_facecolor(theme.ax_bg)
    if metrics:
        sorted_m = sorted(metrics, key=lambda m: m.get("within_cluster_spread", 1e9))
        sids = [f"S{m['syllable_id']}" for m in sorted_m]
        spread = [float(m.get("within_cluster_spread", 0)) for m in sorted_m]
        max_s = max(spread) if spread else 1.0
        colors = [
            theme.bar_bad if s > 0.7 * max_s
            else (theme.bar_warn if s > 0.4 * max_s else theme.bar_ok)
            for s in spread
        ]
        n = len(sids)
        fs_y = max(3.0, min(7.0, 180 / max(n, 1)))
        ax_compact.barh(list(range(n)), spread, color=colors, edgecolor=theme.ax_bg, linewidth=0.2)
        ax_compact.set_yticks(list(range(n)))
        ax_compact.set_yticklabels(sids, fontsize=fs_y, color=theme.label_color)
        ax_compact.set_xlabel("Within-cluster spread", color=theme.label_color, fontsize=7)
    ax_compact.set_title("Compactness", color=theme.title_color, fontsize=9)
    ax_compact.tick_params(colors=theme.tick_color, labelsize=6)
    for spine in ax_compact.spines.values():
        spine.set_edgecolor(theme.spine_color)

    # ── Cell (1,1): Occupancy bars ────────────────────────────────────
    ax_occ = fig.add_subplot(gs[1, 1])
    ax_occ.set_facecolor(theme.ax_bg)
    if metrics:
        sorted_occ = sorted(metrics, key=lambda m: m.get("occupancy_fraction", 0), reverse=True)
        occ_sids = [f"S{m['syllable_id']}" for m in sorted_occ]
        occs = [float(m.get("occupancy_fraction", 0)) for m in sorted_occ]
        n2 = len(occ_sids)
        fs_y2 = max(3.0, min(7.0, 180 / max(n2, 1)))
        ax_occ.barh(list(range(n2)), occs, color=theme.bar_ok, edgecolor=theme.ax_bg, linewidth=0.2)
        ax_occ.set_yticks(list(range(n2)))
        ax_occ.set_yticklabels(occ_sids, fontsize=fs_y2, color=theme.label_color)
        ax_occ.set_xlabel("Occupancy fraction", color=theme.label_color, fontsize=7)
    ax_occ.set_title("Occupancy", color=theme.title_color, fontsize=9)
    ax_occ.tick_params(colors=theme.tick_color, labelsize=6)
    for spine in ax_occ.spines.values():
        spine.set_edgecolor(theme.spine_color)

    # ── Cell (1,2): Transition graph ──────────────────────────────────
    ax_trans = fig.add_subplot(gs[1, 2])
    ax_trans.set_facecolor(theme.ax_bg)
    ax_trans.set_aspect("equal")
    ax_trans.set_xticks([])
    ax_trans.set_yticks([])
    for spine in ax_trans.spines.values():
        spine.set_visible(False)

    if trans_matrix is not None and trans_matrix.ndim == 2:
        # Compute occupancy from data
        counts = np.bincount(label_arr, minlength=n_syllables).astype(float)
        occ_arr = counts / max(counts.sum(), 1)
        try:
            import networkx as nx  # type: ignore  # noqa: PLC0415
            _draw_nx_graph(ax_trans, trans_matrix, occ_arr, palette, theme, threshold=0.02, n=n_syllables)
        except ImportError:
            _draw_circle_graph(ax_trans, trans_matrix, occ_arr, palette, theme, threshold=0.02, n=n_syllables)
    else:
        ax_trans.text(
            0.5, 0.5, "Transition data\nnot available",
            ha="center", va="center", color=theme.tick_color, fontsize=9,
            transform=ax_trans.transAxes,
        )
    ax_trans.set_title("Transitions", color=theme.title_color, fontsize=9)

    fig.suptitle(
        "Keypoint-MoSeq  ·  Model QC Dashboard",
        color=theme.title_color, fontsize=13, y=1.0,
    )
    return fig
