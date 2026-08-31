"""Rendering for the all-project embedding — every knob that changes the picture.

Deliberately split from :mod:`abel.validation.analyses.all_project_umap`: that module
computes coordinates (minutes), this one draws them (a second).  Everything here
reads ``embedding.parquet``, so the figure can be restyled — label distance,
palette, hulls, faceting — over and over without re-running the reducer.  That is
the whole workflow this tab is built around: embed once, then tune the plot.

The label placement is the part with real machinery behind it.  A cluster label
dropped on its own centroid is unreadable on a dense map — it sits inside the points
it names, and neighbouring labels overlap each other.  So labels are placed in three
steps, each exposed as a setting:

1. **Anchor** — the point in the cluster the label refers to (``label_anchor``):
   centroid (can land in a hole for a crescent-shaped cluster), medoid (a real
   point, always inside), or density peak (the visual centre of mass).
2. **Push** — move the label *off* the cluster by ``label_offset``, measured as a
   fraction of the axis span so it means the same thing at any zoom or figure
   size.  ``label_push`` chooses the direction: radially outward from the map's
   centre, toward whichever nearby direction has the fewest points, straight up,
   or ``perimeter`` — every label out to the figure's rim, packed in bearing order
   so leader lines never cross, which is the only thing that stays readable once
   thirty-odd clusters pile into one dense region.
3. **Repel** — labels then push each other apart until none overlap
   (``label_repel_iters``, ``label_min_gap``, ``label_spring``), tethered to their
   pushed position by ``label_leash`` so a label can never wander to a different
   cluster.  (The perimeter push packs exactly and skips this step.)

A leader line is drawn from the anchor to the final label position, which is what
makes a large offset legible rather than ambiguous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon
    from matplotlib import patheffects
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# ── Palettes ────────────────────────────────────────────────────────────────

#: The suite's own palette (matches abel.validation.plots), extended so a map with
#: ~30 assay-scoped groups does not recycle a colour every ten entries.
_ABEL = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39",
    "#3F51B5", "#009688", "#FF5722", "#8BC34A", "#673AB7",
    "#FFC107", "#03A9F4", "#B71C1C", "#1B5E20", "#4A148C",
]

#: Maximally-distinct categorical colours for the 30+ group case, where a
#: perceptual gradient palette (viridis, turbo) makes adjacent groups
#: indistinguishable.
_DISTINCT = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffe119", "#00c2a0", "#ff5005", "#5a0007", "#809693", "#1ce6ff",
    "#ff34ff", "#006fa6", "#a30059", "#7a4900", "#0000a6", "#63ffac",
]

PALETTES = {
    "abel": "ABEL suite palette (matches the other figures)",
    "distinct": "Maximally distinct — best above ~15 groups",
    "tab20": "Matplotlib tab20",
    "assay_shades": "One hue per project, one shade per behavior within it",
    "turbo": "Turbo gradient (ordered — implies a sequence that isn't there)",
}


def _palette_colors(groups: list[str], projects_of: dict[str, str],
                    name: str) -> dict[str, str]:
    """Group → hex colour, under the chosen palette."""
    if name == "assay_shades":
        # A hue per project, lightness varying within it. Reads as "these five
        # clusters are the same assay" at a glance, which is exactly the question
        # an all-project map raises.
        import colorsys

        proj_order = sorted({projects_of.get(g, "") for g in groups})
        out: dict[str, str] = {}
        for pi, proj in enumerate(proj_order):
            members = [g for g in groups if projects_of.get(g, "") == proj]
            hue = (pi / max(1, len(proj_order))) % 1.0
            for mi, g in enumerate(sorted(members)):
                light = 0.35 + 0.42 * (mi / max(1, len(members) - 1 or 1))
                r, gg, b = colorsys.hls_to_rgb(hue, min(0.80, light), 0.72)
                out[g] = "#%02x%02x%02x" % (int(r * 255), int(gg * 255), int(b * 255))
        return out
    if name == "tab20":
        cmap = plt.get_cmap("tab20")
        return {g: matplotlib.colors.to_hex(cmap(i % 20)) for i, g in enumerate(groups)}
    if name == "turbo":
        cmap = plt.get_cmap("turbo")
        n = max(1, len(groups) - 1)
        return {g: matplotlib.colors.to_hex(cmap(i / n)) for i, g in enumerate(groups)}
    base = _DISTINCT if name == "distinct" else _ABEL
    return {g: base[i % len(base)] for i, g in enumerate(groups)}


# ── Settings ────────────────────────────────────────────────────────────────


@dataclass
class PlotSettings:
    """Everything that changes the *picture* and nothing that changes the layout.

    Every one of these is safe to tweak and re-render — the coordinates are already
    on disk.
    """

    # ── Colour ──
    #: What the colours mean. ``group`` = one colour per (project · behavior),
    #: which is the assay-scoped default. ``project`` = one colour per project,
    #: the fastest way to see whether the map is really a project map.
    #: ``behavior`` = one colour per behavior *name*, pooled across projects —
    #: use it to ask whether two assays' Rear land in the same place.
    color_by: str = "group"
    #: See PALETTES.
    palette: str = "distinct"
    #: Marker area in points². 3-8 for a dense map, 15+ for a sparse one.
    point_size: float = 6.0
    #: Draw a second, larger, fainter copy of every marker underneath the first,
    #: at this multiple of the point size (0 = off). Overlapping halos sum where
    #: points are dense, so a cluster's core lights up while stragglers stay
    #: individually visible — the density is read as brightness rather than as
    #: an undifferentiated blob. Costs one extra scatter pass.
    point_glow: float = 0.0
    #: Opacity of that halo pass. Keep it low (0.03-0.10); the glow works by
    #: accumulating, so a visible single halo is already too strong.  Dark theme
    #: only in practice: on white the halos accumulate *toward* the background and
    #: wash the map out instead of lighting it up.
    point_glow_alpha: float = 0.05
    #: Marker opacity. Below ~0.4 overlapping clusters blend into their true
    #: density instead of whichever was drawn last.
    point_alpha: float = 0.55
    #: Outline width per marker (0 = none). Non-zero costs a lot at 50k points.
    edge_width: float = 0.0
    #: Draw points in random order rather than group by group, so no cluster is
    #: buried under whichever group happened to be plotted last.
    shuffle_draw: bool = True
    #: Cap on points actually drawn (0 = all). Purely cosmetic/performance —
    #: it does not change the embedding, only what is rendered.
    plot_max_points: int = 40000
    #: Render markers as a single raster layer. Keeps a 50k-point PDF openable
    #: while leaving text and lines as vectors.
    rasterize_points: bool = True

    # ── Labels: what they say ──
    #: ``none`` · ``anchor`` (text sits on the cluster) · ``offset`` (pushed away,
    #: with a leader line — the readable choice on a dense map).
    label_mode: str = "offset"
    #: Point of the cluster the label refers to and the leader line starts from.
    #: ``medoid`` is always a real point inside the cluster; ``centroid`` can fall
    #: in a hole; ``density`` is the visual centre of mass.
    label_anchor: str = "medoid"
    #: **How far the label sits from its cluster**, as a fraction of the axis span
    #: (0.10 ≈ a tenth of the plot width). 0 puts it on the anchor. Under
    #: ``label_push="perimeter"`` this is instead the inset from the figure edge.
    label_offset: float = 0.10
    #: Direction of that push. ``radial`` = straight away from the map's centre.
    #: ``sparse`` = toward the emptiest nearby direction (best when clusters are
    #: interleaved). ``perimeter`` = fan every label out to the figure's rim at its
    #: own bearing, leader lines pointing back in — the readable choice when many
    #: clusters pile into one dense region. ``up`` = straight up. ``none`` = no push.
    label_push: str = "radial"
    #: Overlap-resolution passes. 0 disables repulsion entirely (labels may then
    #: sit on top of each other); 300-800 for a crowded map.
    label_repel_iters: int = 400
    #: Minimum clear space between two label boxes, in axis fractions.
    label_min_gap: float = 0.012
    #: How hard each label is pulled back to its ideal pushed position each pass.
    #: Low (0.01-0.03) lets crowded labels spread out properly; high (0.1+) keeps
    #: them near their clusters and accepts some overlap. This is the knob to turn
    #: when labels still collide after raising the iteration count.
    label_spring: float = 0.02
    #: How far repulsion may drag a label from its pushed position, in axis
    #: fractions. The tether is what stops a crowded label from drifting across
    #: the map and appearing to name a different cluster.
    label_leash: float = 0.22
    #: Also repel labels away from dense point regions, so text does not land on
    #: top of a cluster it does not name. Costs one KD-tree query per label.
    label_avoid_points: bool = True

    # ── Labels: how they look ──
    label_font_size: float = 9.0
    #: ``normal`` · ``bold``.
    label_font_weight: str = "bold"
    #: White/black outline stroke width behind the glyphs (0 = off). The cheapest
    #: way to keep text legible over points without an opaque box.
    label_halo: float = 2.2
    #: ``cluster`` (the group's own colour) · ``foreground`` (theme text colour).
    label_color: str = "cluster"
    #: Draw a filled box behind each label. Very legible, and it takes up more
    #: room, which makes the repulsion push labels further apart.
    label_box: bool = False
    label_box_alpha: float = 0.80
    #: Truncate long "Project · Behavior" labels to this many characters
    #: (0 = never truncate).
    max_label_chars: int = 30
    #: Append the point count, e.g. "EPM · Head Dip (n=412)".
    label_show_counts: bool = False
    #: Drop labels for groups with fewer than this many points — small groups
    #: produce the most crowding for the least information (0 = label everything).
    label_min_points: int = 0

    # ── Leader lines ──
    leader_lines: bool = True
    leader_width: float = 0.7
    leader_alpha: float = 0.65
    #: Any matplotlib linestyle: ``-`` ``--`` ``:`` ``-.``
    leader_style: str = "-"
    #: Colour the leader line like the cluster instead of the theme's muted grey.
    leader_color_by_cluster: bool = True

    # ── Cluster overlays ──
    #: ``none`` · ``hull`` (convex hull) · ``ellipse`` (2-SD covariance ellipse,
    #: robust to stragglers) · ``density`` (KDE contour at the given quantile).
    overlay: str = "none"
    overlay_alpha: float = 0.12
    #: Line width of the overlay outline (0 = fill only).
    overlay_edge_width: float = 0.8
    #: ``hull``: fraction of the most central points enclosed, which trims the
    #: outliers that otherwise stretch a hull across the whole figure.
    #: ``density``: the contour level, as a fraction of mass enclosed.
    overlay_quantile: float = 0.90
    #: Mark each cluster's anchor with a ringed dot.
    centroid_marker: bool = False

    # ── Frame ──
    #: ``none`` · ``project`` · ``behavior`` — one small panel per project or per
    #: behavior name, all sharing the single embedding's axes, so panels are
    #: directly comparable. Grey context points show the rest of the map.
    facet_by: str = "none"
    facet_cols: int = 3
    #: Draw the other panels' points in grey behind each facet.
    facet_context: bool = True
    #: ``none`` · ``right`` · ``below``.
    legend: str = "right"
    legend_cols: int = 1
    legend_font_size: float = 8.0
    #: ``light`` · ``dark``. Dark reads well on screen; light is what a journal wants.
    theme: str = "light"
    #: UMAP axes carry no units and no meaning — off by default for that reason.
    show_axes: bool = False
    #: Lock the aspect ratio so distances mean the same thing in x and y. Turning
    #: it off lets the map fill the page and silently distorts every cluster shape.
    equal_aspect: bool = True
    fig_width: float = 12.0
    fig_height: float = 9.0
    dpi: int = 200
    title: str = ""
    #: Small caption under the map holding the QC line (behavior vs project
    #: structure) so a figure lifted out of the folder still carries its caveat.
    subtitle: str = ""
    #: Also write a PDF next to the PNG.
    save_pdf: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PlotSettings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


_THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a1a", "muted": "#8a8a8a",
              "context": "#d8d8d8", "halo": "#ffffff"},
    "dark": {"bg": "#11121a", "fg": "#e8e8ef", "muted": "#8b90a8",
             "context": "#2c2e3d", "halo": "#11121a"},
}


# ── Label geometry ──────────────────────────────────────────────────────────


def _anchor_xy(pts: np.ndarray, mode: str) -> np.ndarray:
    """The point in a cluster its label refers to."""
    if len(pts) == 1:
        return pts[0]
    if mode == "centroid":
        return pts.mean(axis=0)
    if mode == "density":
        # Coarse 2-D histogram peak: the visual centre of mass, unlike the
        # centroid, which a crescent-shaped cluster puts in its own empty middle.
        bins = max(6, int(np.sqrt(len(pts)) / 2))
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        return np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    # medoid — the real point closest to the centroid, so it is always inside.
    c = pts.mean(axis=0)
    sub = pts if len(pts) <= 4000 else pts[
        np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    return sub[int(np.argmin(((sub - c) ** 2).sum(axis=1)))]


def _push_direction(anchor: np.ndarray, all_pts: np.ndarray, centre: np.ndarray,
                    mode: str) -> np.ndarray:
    """Unit vector along which to push a label off its cluster."""
    if mode == "up":
        return np.array([0.0, 1.0])
    if mode == "sparse":
        # Sixteen candidate bearings; pick the one with the fewest points inside a
        # wedge of the local radius. On a crowded map this is what keeps a label
        # from being pushed straight into the neighbouring cluster.
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        rel = all_pts - anchor
        d = np.hypot(rel[:, 0], rel[:, 1])
        near = rel[(d > 0) & (d < np.percentile(d, 25))]
        if len(near) < 5:
            mode = "radial"
        else:
            bearing = np.arctan2(near[:, 1], near[:, 0])
            counts = [int(np.sum(np.abs(np.angle(np.exp(1j * (bearing - a)))) < np.pi / 8))
                      for a in angles]
            a = angles[int(np.argmin(counts))]
            return np.array([np.cos(a), np.sin(a)])
    if mode == "none":
        return np.array([0.0, 0.0])
    v = anchor - centre
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([0.0, 1.0])


def _label_extent(text: str, font_size: float, ax_w_in: float, ax_h_in: float,
                  boxed: bool) -> tuple[float, float]:
    """Half-width and half-height of a label, in axis fractions.

    Estimated from the character count rather than measured with a renderer: the
    repulsion runs before the text exists, and an estimate good to ~15% is enough
    to separate labels. Deliberately generous so boxes end up with visible air
    between them instead of just touching.
    """
    pad = 1.35 if boxed else 1.12
    char_w_in = font_size * 0.55 / 72.0
    w = len(text) * char_w_in * pad
    h = font_size * 1.45 / 72.0 * pad
    return (w / 2.0) / max(ax_w_in, 1e-6), (h / 2.0) / max(ax_h_in, 1e-6)


def _rim_point(t: float, half: float, L: float) -> tuple[np.ndarray, bool]:
    """Point at arc-length ``t`` around the inset box, and whether that edge is
    horizontal.  Walk order: bottom → right → top → left, starting bottom-left."""
    side = 2.0 * half
    t = float(t) % L
    lo, hi = 0.5 - half, 0.5 + half
    if t < side:
        return np.array([lo + t, lo]), True
    t -= side
    if t < side:
        return np.array([hi, lo + t]), False
    t -= side
    if t < side:
        return np.array([hi - t, hi]), True
    t -= side
    return np.array([lo, hi - t]), False


def _rim_t(u: np.ndarray, half: float, L: float) -> float:
    """Arc-length at which the ray from the box centre along ``u`` leaves the box."""
    ux, uy = float(u[0]), float(u[1])
    sx = half / abs(ux) if abs(ux) > 1e-9 else np.inf
    sy = half / abs(uy) if abs(uy) > 1e-9 else np.inf
    k = min(sx, sy)
    p = np.array([0.5, 0.5]) + np.array([ux, uy]) * k
    side = 2.0 * half
    lo, hi = 0.5 - half, 0.5 + half
    if abs(p[1] - lo) < 1e-6:
        return float(p[0] - lo)
    if abs(p[0] - hi) < 1e-6:
        return float(side + (p[1] - lo))
    if abs(p[1] - hi) < 1e-6:
        return float(2 * side + (hi - p[0]))
    return float(3 * side + (hi - p[1]))


def _pack_perimeter(anc_f: np.ndarray, half_sizes: np.ndarray, inset: float,
                    gap: float) -> np.ndarray:
    """Fan labels around the figure's rim, ordered by bearing and packed to fit.

    A pure "push each label to the rim along its own bearing" fans them out but
    still stacks several on the same stretch of edge.  Packing them by arc length —
    each label reserving its own width along the rim — is what actually resolves
    that, and it keeps the *order* around the map, so a leader line never crosses
    its neighbour's.
    """
    n = len(anc_f)
    half = max(0.02, 0.5 - float(inset))
    L = 8.0 * half
    centre = np.array([0.5, 0.5])

    t = np.empty(n)
    for i in range(n):
        v = anc_f[i] - centre
        nv = float(np.hypot(*v))
        t[i] = _rim_t(v / nv if nv > 1e-9 else np.array([0.0, 1.0]), half, L)

    def footprints() -> np.ndarray:
        # A label's footprint along the rim is its *width* on a horizontal edge but
        # only its *height* on a vertical one — roughly five times cheaper. It must
        # therefore be recomputed as labels slide between edges, or the top edge
        # stays oversubscribed while the sides reserve room they do not need.
        f = np.empty(n)
        for i in range(n):
            _, horiz = _rim_point(t[i], half, L)
            f[i] = (2 * half_sizes[i][0] if horiz else 2 * half_sizes[i][1]) + gap
        return f

    order = np.argsort(t)
    for _ in range(60):
        foot = footprints()
        if float(foot.sum()) >= L:
            # More label than rim will hold. Spread *proportionally* to each
            # label's footprint rather than evenly: even spacing gives a 3-character
            # label on a side edge the same rim as a 22-character one on the top,
            # so the sides end up empty while the top stays illegible. Everything
            # is still too tight — raise the figure size, shrink the font, cap
            # max_label_chars or set label_min_points — but the crowding is at
            # least shared out.
            scale = L / float(foot.sum())
            base = run = float(t[order[0]])
            for i in order:
                t[i] = run + foot[i] * scale / 2.0
                run += foot[i] * scale
            t[order[0]] = base + foot[order[0]] * scale / 2.0
            continue  # footprints change as labels migrate between edges
        moved = False
        for k in range(n):
            a, b = order[k], order[(k + 1) % n]
            need = (foot[a] + foot[b]) / 2.0
            d = (t[b] - t[a]) % L
            if d < need - 1e-9:
                push = (need - d) / 2.0
                t[a] -= push
                t[b] += push
                moved = True
        if not moved:
            break

    pos = np.array([_rim_point(t[i], half, L)[0] for i in range(n)])
    # Keep the text on the canvas: a long label centred on the left rim would
    # otherwise run off the page.
    for i in range(n):
        pos[i] = np.clip(pos[i], half_sizes[i] + 0.004, 1.0 - half_sizes[i] - 0.004)
    return pos


def place_labels(
    anchors: dict[str, np.ndarray],
    texts: dict[str, str],
    all_pts: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    s: PlotSettings,
    ax_size_in: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Anchors → final label positions, in data coordinates.

    Works internally in axis fractions so ``label_offset``, ``label_min_gap`` and
    ``label_leash`` all mean the same thing regardless of the embedding's arbitrary
    coordinate scale — a UMAP run can land on a span of 8 units or 80.
    """
    if not anchors:
        return {}
    xspan = max(float(xlim[1] - xlim[0]), 1e-9)
    yspan = max(float(ylim[1] - ylim[0]), 1e-9)

    def to_frac(p: np.ndarray) -> np.ndarray:
        return np.array([(p[0] - xlim[0]) / xspan, (p[1] - ylim[0]) / yspan])

    def to_data(p: np.ndarray) -> np.ndarray:
        return np.array([xlim[0] + p[0] * xspan, ylim[0] + p[1] * yspan])

    keys = list(anchors)
    centre = all_pts.mean(axis=0)
    anc_f = np.array([to_frac(anchors[k]) for k in keys])

    ax_w, ax_h = ax_size_in or (float(s.fig_width), float(s.fig_height))
    half = np.array([
        _label_extent(texts[k], s.label_font_size, ax_w, ax_h, s.label_box)
        for k in keys
    ])
    gap = float(s.label_min_gap)
    leash = float(s.label_leash)

    # Step 2 — push each label off its cluster along the chosen direction. The
    # direction is computed in data space (that is where the points are) then
    # normalized in fraction space, so an anisotropic axis range does not turn a
    # "radial" push into a mostly-vertical one.
    targets = anc_f.copy()
    if s.label_mode == "offset" and s.label_push == "perimeter":
        # Packing around the rim is a complete 1-D solution — the 2-D repulsion
        # below would only drag labels back off the edge, so it is skipped.
        return {
            k: to_data(p) for k, p in zip(
                keys, _pack_perimeter(anc_f, half,
                                      float(np.clip(s.label_offset, 0.0, 0.45)), gap))
        }
    if s.label_mode == "offset" and s.label_offset > 0:
        for i, k in enumerate(keys):
            d = _push_direction(anchors[k], all_pts, centre, s.label_push)
            df = np.array([d[0] / xspan, d[1] / yspan])
            n = float(np.hypot(*df))
            if n > 1e-12:
                targets[i] = anc_f[i] + (df / n) * float(s.label_offset)

    pos = targets.copy()

    # Optional: sample the point cloud so labels can be pushed off dense regions.
    cloud = None
    if s.label_avoid_points and len(all_pts) > 0:
        rng = np.random.default_rng(0)
        sub = (all_pts if len(all_pts) <= 4000
               else all_pts[rng.choice(len(all_pts), 4000, replace=False)])
        cloud = np.column_stack([(sub[:, 0] - xlim[0]) / xspan,
                                 (sub[:, 1] - ylim[0]) / yspan])

    # Step 3 — repel. Overlapping boxes push apart along the axis of least
    # overlap; a spring pulls every label back toward its target and the leash
    # hard-clamps the total drift.
    for _ in range(max(0, int(s.label_repel_iters))):
        moved = False
        for i in range(len(keys)):
            force = np.zeros(2)
            for j in range(len(keys)):
                if i == j:
                    continue
                d = pos[i] - pos[j]
                need = half[i] + half[j] + gap
                over = need - np.abs(d)
                if over[0] > 0 and over[1] > 0:
                    # Separate along whichever axis needs the smaller move.
                    ax = 0 if over[0] / max(need[0], 1e-9) < over[1] / max(need[1], 1e-9) else 1
                    sign = 1.0 if d[ax] >= 0 else -1.0
                    force[ax] += sign * over[ax] * 0.6
                    moved = True
            if cloud is not None:
                d = cloud - pos[i]
                inside = (np.abs(d[:, 0]) < half[i][0]) & (np.abs(d[:, 1]) < half[i][1])
                n_in = int(inside.sum())
                if n_in > 3:
                    push = -d[inside].mean(axis=0)
                    n = float(np.hypot(*push))
                    if n > 1e-9:
                        force += (push / n) * min(0.01, 0.0004 * n_in)
                        moved = True
            # Spring back toward the pushed target. Weak by default: a strong
            # spring wins every argument with the separation force and the labels
            # simply stay stacked on top of each other.
            force += (targets[i] - pos[i]) * float(s.label_spring)
            pos[i] = pos[i] + force
            # Leash from the *target*, not the anchor — under the perimeter push
            # the target is deliberately far from the cluster.
            drift = pos[i] - targets[i]
            dn = float(np.hypot(*drift))
            if dn > leash:
                pos[i] = targets[i] + drift / dn * leash
            pos[i] = np.clip(pos[i], half[i] + 0.004, 1.0 - half[i] - 0.004)
        if not moved:
            break

    return {k: to_data(pos[i]) for i, k in enumerate(keys)}


# ── Cluster overlays ────────────────────────────────────────────────────────


def _central_points(pts: np.ndarray, q: float) -> np.ndarray:
    """The most central ``q`` fraction of a cluster, by distance to its centroid."""
    if q >= 1.0 or len(pts) < 4:
        return pts
    c = pts.mean(axis=0)
    d = np.hypot(*(pts - c).T)
    return pts[d <= np.quantile(d, float(q))]


def _draw_overlay(ax, pts: np.ndarray, colour: str, s: PlotSettings) -> None:
    if s.overlay == "none" or len(pts) < 4:
        return
    edge = dict(edgecolor=colour, linewidth=float(s.overlay_edge_width)) \
        if s.overlay_edge_width > 0 else dict(edgecolor="none", linewidth=0)
    if s.overlay == "ellipse":
        c = pts.mean(axis=0)
        cov = np.cov(pts.T)
        if not np.all(np.isfinite(cov)):
            return
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        w, h = 2 * 2 * np.sqrt(vals)        # 2 SD on each semi-axis
        ax.add_patch(Ellipse(c, w, h, angle=angle, facecolor=colour,
                             alpha=float(s.overlay_alpha), zorder=1, **edge))
        return
    if s.overlay == "density":
        try:
            from scipy.stats import gaussian_kde  # noqa: PLC0415
        except ImportError:
            return
        sub = pts if len(pts) <= 3000 else pts[
            np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
        try:
            kde = gaussian_kde(sub.T)
        except Exception:  # noqa: BLE001 — singular cluster, nothing to contour
            return
        pad = 0.15
        xs = np.linspace(sub[:, 0].min(), sub[:, 0].max(), 60)
        ys = np.linspace(sub[:, 1].min(), sub[:, 1].max(), 60)
        xr, yr = float(np.ptp(xs)) * pad, float(np.ptp(ys)) * pad
        xs = np.linspace(xs[0] - xr, xs[-1] + xr, 60)
        ys = np.linspace(ys[0] - yr, ys[-1] + yr, 60)
        XX, YY = np.meshgrid(xs, ys)
        Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        # Contour at the level enclosing `overlay_quantile` of the probability mass.
        flat = np.sort(Z.ravel())[::-1]
        csum = np.cumsum(flat) / flat.sum()
        level = float(flat[np.searchsorted(csum, float(s.overlay_quantile))])
        ax.contourf(XX, YY, Z, levels=[level, Z.max()], colors=[colour],
                    alpha=float(s.overlay_alpha), zorder=1)
        if s.overlay_edge_width > 0:
            ax.contour(XX, YY, Z, levels=[level], colors=[colour],
                       linewidths=float(s.overlay_edge_width), zorder=1)
        return
    # hull
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415
    except ImportError:
        return
    sub = _central_points(pts, s.overlay_quantile)
    if len(sub) < 4:
        return
    try:
        hull = ConvexHull(sub)
    except Exception:  # noqa: BLE001 — collinear cluster has no hull
        return
    ax.add_patch(Polygon(sub[hull.vertices], closed=True, facecolor=colour,
                         alpha=float(s.overlay_alpha), zorder=1, **edge))


# ── The figure ──────────────────────────────────────────────────────────────


def _colour_key(frame: pd.DataFrame, color_by: str) -> pd.Series:
    if color_by == "project":
        return frame["project_id"].astype(str)
    if color_by == "behavior":
        return frame["behavior_name"].astype(str)
    return frame["group"].astype(str)


def _shorten(text: str, n: int) -> str:
    if not n or len(text) <= n:
        return text
    return text[: max(1, n - 1)] + "…"


def _scatter(ax, frame: pd.DataFrame, key: pd.Series, colours: dict[str, str],
             s: PlotSettings, rng: np.random.Generator) -> None:
    x = frame["x"].to_numpy()
    y = frame["y"].to_numpy()
    c = np.array([colours.get(k, "#888888") for k in key])
    if s.shuffle_draw:
        order = rng.permutation(len(x))
        x, y, c = x[order], y[order], c[order]
    if s.point_glow > 0:
        ax.scatter(x, y, s=float(s.point_size) * float(s.point_glow), c=c,
                   alpha=float(s.point_glow_alpha), linewidths=0, edgecolors="none",
                   rasterized=bool(s.rasterize_points), zorder=1.5)
    ax.scatter(x, y, s=float(s.point_size), c=c, alpha=float(s.point_alpha),
               linewidths=float(s.edge_width),
               edgecolors="none" if s.edge_width <= 0 else "#00000055",
               rasterized=bool(s.rasterize_points), zorder=2)


def _annotate(ax, frame: pd.DataFrame, colours: dict[str, str], theme: dict,
              s: PlotSettings, label_key: str) -> None:
    """Anchor, push, repel and draw every cluster label on one axes."""
    if s.label_mode == "none":
        return
    pts_all = frame[["x", "y"]].to_numpy()
    anchors: dict[str, np.ndarray] = {}
    texts: dict[str, str] = {}
    for name, sub in frame.groupby(label_key, sort=False):
        if s.label_min_points and len(sub) < int(s.label_min_points):
            continue
        pts = sub[["x", "y"]].to_numpy()
        anchors[str(name)] = _anchor_xy(pts, s.label_anchor)
        label = _shorten(str(name), int(s.max_label_chars))
        if s.label_show_counts:
            label = f"{label} (n={len(sub)})"
        texts[str(name)] = label
    if not anchors:
        return

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    # Size the label boxes against the *axes*, not the figure: with a legend and
    # several facets an axes can be a third of the page, and estimating text width
    # against the figure would under-separate the labels by that same factor.
    fig = ax.figure
    box = ax.get_position()
    ax_w = box.width * fig.get_figwidth()
    ax_h = box.height * fig.get_figheight()
    if s.equal_aspect:
        # ``set_aspect("equal", adjustable="box")`` shrinks the drawn axes to the
        # data's own aspect at draw time, which is after this runs. Predict it,
        # or every label box is measured against a box wider than the real one.
        want = abs(ylim[1] - ylim[0]) / max(abs(xlim[1] - xlim[0]), 1e-12)
        if ax_h / max(ax_w, 1e-12) > want:
            ax_h = ax_w * want
        else:
            ax_w = ax_h / max(want, 1e-12)
    ax_size = (ax_w, ax_h)
    placed = place_labels(anchors, texts, pts_all, xlim, ylim, s, ax_size_in=ax_size)

    for name, xy in placed.items():
        col = colours.get(name, theme["fg"])
        anc = anchors[name]
        if s.leader_lines and float(np.hypot(*(xy - anc))) > 1e-9:
            ax.plot([anc[0], xy[0]], [anc[1], xy[1]],
                    color=col if s.leader_color_by_cluster else theme["muted"],
                    linewidth=float(s.leader_width), alpha=float(s.leader_alpha),
                    linestyle=s.leader_style, zorder=3,
                    solid_capstyle="round")
        if s.centroid_marker:
            ax.plot(anc[0], anc[1], marker="o", markersize=5, color=col,
                    markeredgecolor=theme["halo"], markeredgewidth=1.0, zorder=4)
        txt_col = col if s.label_color == "cluster" else theme["fg"]
        bbox = (dict(boxstyle="round,pad=0.28", facecolor=theme["bg"],
                     edgecolor=col, alpha=float(s.label_box_alpha), linewidth=0.6)
                if s.label_box else None)
        t = ax.text(xy[0], xy[1], texts[name], color=txt_col,
                    fontsize=float(s.label_font_size), fontweight=s.label_font_weight,
                    ha="center", va="center", zorder=5, bbox=bbox)
        if s.label_halo > 0 and not s.label_box:
            t.set_path_effects([
                patheffects.withStroke(linewidth=float(s.label_halo),
                                       foreground=theme["halo"])])


def _style_axes(ax, theme: dict, s: PlotSettings, reducer: str) -> None:
    ax.set_facecolor(theme["bg"])
    if s.equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if s.show_axes:
        label = {"umap": "UMAP", "tsne": "t-SNE", "pca": "PC"}.get(reducer, "Dim")
        ax.set_xlabel(f"{label} 1", color=theme["muted"], fontsize=9)
        ax.set_ylabel(f"{label} 2", color=theme["muted"], fontsize=9)
        ax.tick_params(colors=theme["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(theme["muted"])
            sp.set_linewidth(0.6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


def render(
    frame: pd.DataFrame,
    s: PlotSettings,
    out_dir: Path,
    *,
    reducer: str = "umap",
    stem: str = "all_project_umap",
) -> list[Path]:
    """Draw the map.  Returns every image written (PNG first, then any PDF)."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is not installed — cannot render the map.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theme = _THEMES.get(s.theme, _THEMES["light"])
    rng = np.random.default_rng(0)

    plot_frame = frame
    if s.plot_max_points and len(frame) > int(s.plot_max_points):
        # Cosmetic only — labels and overlays are still computed from every point,
        # so thinning the render never moves a label.
        plot_frame = frame.sample(int(s.plot_max_points), random_state=0)

    key_all = _colour_key(frame, s.color_by)
    groups = sorted(key_all.unique())
    # group → project, for the assay_shades palette.
    proj_of = {str(g): str(sub["project_id"].iloc[0])
               for g, sub in frame.groupby(key_all, sort=False)}
    colours = _palette_colors(groups, proj_of, s.palette)

    label_key = "group" if s.color_by == "group" else (
        "project_id" if s.color_by == "project" else "behavior_name")

    facets: list[tuple[str, pd.DataFrame]]
    if s.facet_by == "project":
        facets = [(str(k), v) for k, v in frame.groupby("project_id", sort=True)]
    elif s.facet_by == "behavior":
        facets = [(str(k), v) for k, v in frame.groupby("behavior_name", sort=True)]
    else:
        facets = [("", frame)]

    ncols = max(1, int(s.facet_cols)) if len(facets) > 1 else 1
    nrows = int(np.ceil(len(facets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(float(s.fig_width), float(s.fig_height)),
        squeeze=False, facecolor=theme["bg"])
    flat = [ax for row in axes for ax in row]

    xlim = (float(frame["x"].min()), float(frame["x"].max()))
    ylim = (float(frame["y"].min()), float(frame["y"].max()))
    padx, pady = 0.06 * (xlim[1] - xlim[0]), 0.06 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - padx, xlim[1] + padx)
    ylim = (ylim[0] - pady, ylim[1] + pady)

    for ax, (fname, sub) in zip(flat, facets):
        # Shared limits across facets, so panels are directly comparable.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if len(facets) > 1 and s.facet_context:
            ctx = plot_frame if len(plot_frame) <= 15000 else plot_frame.sample(
                15000, random_state=1)
            ax.scatter(ctx["x"], ctx["y"], s=max(1.0, float(s.point_size) * 0.5),
                       c=theme["context"], alpha=0.5, linewidths=0,
                       rasterized=bool(s.rasterize_points), zorder=1)
        draw = sub if len(facets) > 1 else plot_frame
        if len(facets) > 1 and s.plot_max_points and len(draw) > int(s.plot_max_points):
            draw = draw.sample(int(s.plot_max_points), random_state=0)
        if s.overlay != "none":
            for g, gs in sub.groupby(_colour_key(sub, s.color_by), sort=False):
                _draw_overlay(ax, gs[["x", "y"]].to_numpy(),
                              colours.get(str(g), "#888888"), s)
        _scatter(ax, draw, _colour_key(draw, s.color_by), colours, s, rng)
        _annotate(ax, sub, colours, theme, s, label_key)
        _style_axes(ax, theme, s, reducer)
        if fname:
            ax.set_title(fname, color=theme["fg"], fontsize=10, fontweight="bold")

    for ax in flat[len(facets):]:
        ax.set_visible(False)

    if s.legend != "none" and groups:
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                              color=colours.get(g, "#888"), label=_shorten(g, 34))
                   for g in groups]
        kwargs = dict(frameon=False, fontsize=float(s.legend_font_size),
                      labelcolor=theme["fg"], ncol=max(1, int(s.legend_cols)))
        if s.legend == "below":
            fig.legend(handles=handles, loc="lower center",
                       bbox_to_anchor=(0.5, -0.02), **kwargs)
        else:
            fig.legend(handles=handles, loc="center left",
                       bbox_to_anchor=(1.005, 0.5), **kwargs)

    if s.title:
        fig.suptitle(s.title, color=theme["fg"], fontsize=13, fontweight="bold")
    if s.subtitle:
        fig.text(0.5, 0.005, s.subtitle, ha="center", va="bottom",
                 color=theme["muted"], fontsize=8, wrap=True)

    fig.tight_layout(rect=(0, 0.02 if s.subtitle else 0, 1, 0.97 if s.title else 1))

    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=int(s.dpi), bbox_inches="tight", facecolor=theme["bg"])
    written = [png]
    if s.save_pdf:
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor=theme["bg"])
        written.append(pdf)
    plt.close(fig)
    return written


def render_qc(frame: pd.DataFrame, qc: dict, out_dir: Path,
              s: PlotSettings) -> Path | None:
    """The companion panel: is this a behavior map or a project map?

    Two small views of the same coordinates — coloured by behavior, then by
    project — plus the four structure numbers.  Published next to the main figure
    so nobody has to take the caption's word for it.
    """
    if not _HAS_MPL:
        return None
    theme = _THEMES.get(s.theme, _THEMES["light"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(float(s.fig_width), float(s.fig_height) * 0.55),
                             facecolor=theme["bg"])
    for ax, mode, title in ((axes[0], "behavior", "Coloured by behavior"),
                            (axes[1], "project", "Coloured by project")):
        key = _colour_key(frame, mode)
        groups = sorted(key.unique())
        proj_of = {str(g): str(sub["project_id"].iloc[0])
                   for g, sub in frame.groupby(key, sort=False)}
        colours = _palette_colors(groups, proj_of, s.palette)
        sub = frame if len(frame) <= 25000 else frame.sample(25000, random_state=0)
        _scatter(ax, sub, _colour_key(sub, mode), colours, s, rng)
        ax.set_title(title, color=theme["fg"], fontsize=10, fontweight="bold")
        _style_axes(ax, theme, s, "umap")
        if s.equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    sb, sp = qc.get("silhouette_behavior_feat"), qc.get("silhouette_project_feat")
    kb, kp = qc.get("knn_purity_behavior_feat"), qc.get("knn_purity_project_feat")
    kc = qc.get("knn_purity_project_chance")
    bits = []
    if sb is not None and sp is not None:
        bits.append(f"feature-space silhouette — behavior {sb:+.3f} · project {sp:+.3f}")
    if kb is not None and kp is not None:
        bits.append(f"kNN purity (k=15) — behavior {kb:.2f} · project {kp:.2f}"
                    + (f" (chance {kc:.2f})" if kc is not None else ""))
    if bits:
        fig.text(0.5, 0.005, "   |   ".join(bits), ha="center", va="bottom",
                 color=theme["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out_dir / "all_project_umap_qc.png"
    fig.savefig(path, dpi=int(s.dpi), bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)
    return path
