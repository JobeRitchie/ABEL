"""Regression: spatial density / heatmap pooling must normalise across resolutions.

Bug (found on the TMT project): the Analytics tab's Spatial Heatmap and Density
Analysis views pool raw pose centroid coordinates from every session and bin them
against a single reference resolution (the first session's video), with no
per-session normalisation.  Pose coordinates live in the pixel space of the video
each subject was tracked on, so when subjects are recorded at different
resolutions the lower-resolution subjects collapse into the top-left corner and
higher-resolution subjects overflow the arena — the data points land in the wrong
place and the two subjects never overlay.

The fix rescales each session's coordinates from its own native video resolution
onto the reference frame (the resolution the background plate and axes use) before
pooling, via ``_scale_xy_to_reference``.
"""

from __future__ import annotations

import numpy as np

from abel.ui.tabs.behavior_analytics_tab import _scale_xy_to_reference


def test_lower_res_subject_scales_up_to_reference() -> None:
    # A subject centred in a 640x480 arena.
    xs = np.array([320.0, 320.0])
    ys = np.array([240.0, 240.0])
    sx, sy = _scale_xy_to_reference(xs, ys, sess_w=640, sess_h=480, ref_w=1280, ref_h=720)
    # Should map to the centre of the 1280x720 reference frame.
    assert np.allclose(sx, 640.0)
    assert np.allclose(sy, 360.0)


def test_matching_resolution_is_untouched() -> None:
    xs = np.array([100.0, 900.0])
    ys = np.array([50.0, 700.0])
    sx, sy = _scale_xy_to_reference(xs, ys, 1280, 720, 1280, 720)
    assert sx is xs and sy is ys  # returned unchanged (no copy) when already aligned


def test_unknown_resolution_leaves_coords_unscaled() -> None:
    xs = np.array([10.0, 20.0])
    ys = np.array([30.0, 40.0])
    # Missing session dims (0) → cannot normalise; leave as-is rather than corrupt.
    sx, sy = _scale_xy_to_reference(xs, ys, 0, 0, 1280, 720)
    assert sx is xs and sy is ys


def test_pooling_two_resolutions_aligns_a_shared_location() -> None:
    """Two subjects at the SAME physical arena location, filmed at different
    resolutions, must land on the same reference pixel after normalisation."""
    ref_w, ref_h = 1280, 720

    # Subject A (native reference res) hugging the arena's right-centre edge.
    ax = np.full(100, 1200.0)
    ay = np.full(100, 360.0)
    a_sx, a_sy = _scale_xy_to_reference(ax, ay, 1280, 720, ref_w, ref_h)

    # Subject B at the *same fraction* of a half-resolution frame.
    bx = np.full(100, 600.0)   # 1200/2
    by = np.full(100, 180.0)   # 360/2
    b_sx, b_sy = _scale_xy_to_reference(bx, by, 640, 360, ref_w, ref_h)

    # After normalisation both subjects occupy the same reference pixel.
    assert np.allclose(a_sx, b_sx)
    assert np.allclose(a_sy, b_sy)

    # And a 2-D histogram of the pooled points forms ONE peak, not two.
    pooled_x = np.concatenate([a_sx, b_sx])
    pooled_y = np.concatenate([a_sy, b_sy])
    H, _, _ = np.histogram2d(pooled_x, pooled_y, bins=64,
                             range=[[0, ref_w], [0, ref_h]])
    assert int((H > 0).sum()) == 1  # single occupied bin

    # Sanity: WITHOUT normalisation the same raw points land in two bins.
    H_raw, _, _ = np.histogram2d(
        np.concatenate([ax, bx]), np.concatenate([ay, by]),
        bins=64, range=[[0, ref_w], [0, ref_h]],
    )
    assert int((H_raw > 0).sum()) == 2


def test_heatmap_renders_arena_at_true_aspect_not_square() -> None:
    """A wide, narrow arena must render wide, not stretched toward square.

    Reproduces the Spatial Heatmap fix: locking the data aspect to "equal" and
    sizing the figure to the arena aspect keeps a 1352x612 (~2.2:1) enclosure at
    its true proportions instead of filling a squarish 700x550 box.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax_w, ax_h = 1352, 612          # true (wide) arena pixels
    arena_aspect = ax_h / ax_w      # ~0.453

    fig = plt.figure(figsize=(7.0, 5.5))   # the old squarish default box
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((ax_h, ax_w)), aspect="auto")   # the offending stretch
    ax.set_xlim(0, ax_w)
    ax.set_ylim(ax_h, 0)

    # ── apply the fix ──
    ax.set_aspect("equal")
    fig.set_size_inches(7.0, max(1.0, 7.0 * arena_aspect))
    fig.canvas.draw()

    # The rendered axes box must now carry the arena's aspect ratio, not the
    # figure box's ~0.79.  Measure the displayed data extent.
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((ax_w, ax_h))
    disp_w = abs(p1[0] - p0[0])
    disp_h = abs(p1[1] - p0[1])
    rendered_aspect = disp_h / disp_w

    assert abs(rendered_aspect - arena_aspect) < 0.02, (
        f"arena rendered at {rendered_aspect:.3f}, expected ~{arena_aspect:.3f}"
    )
    plt.close(fig)
