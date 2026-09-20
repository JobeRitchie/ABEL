"""Full-frame clip extraction: the whole arena instead of a crop."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from abel.models.schemas import PreprocessingPreset  # noqa: E402
from abel.services.preprocessing_service import ClipExtractionService  # noqa: E402


def _video(path, w=160, h=80, n=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    assert writer.isOpened()
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = i * 10  # something that varies so frames aren't identical
        writer.write(frame)
    writer.release()
    return path


def _preset(**kw):
    base = dict(
        preset_id="t", name="t", resize_width=64, resize_height=64,
        crop_margin_px=16, output_fps=15.0,
    )
    return PreprocessingPreset(**(base | kw))


def _extract(tmp_path, preset):
    src = _video(tmp_path / "src.mp4")
    cap = cv2.VideoCapture(str(src))
    out = tmp_path / f"clip_{preset.full_frame}.mp4"
    try:
        err = ClipExtractionService()._write_clip(
            cap=cap, out_path=out, start_frame=2, end_frame=8,
            cx=40.0, cy=20.0, preset=preset,
        )
    finally:
        cap.release()
    assert err is None, err
    probe = cv2.VideoCapture(str(out))
    ok, frame = probe.read()
    probe.release()
    assert ok
    return frame.shape[:2]


def test_cropped_clips_keep_the_preset_output_size(tmp_path):
    assert _extract(tmp_path, _preset()) == (64, 64)


def test_full_frame_clips_keep_the_video_aspect_ratio(tmp_path):
    # 160x80 source at width 64 -> 64x32, not the preset's square.
    assert _extract(tmp_path, _preset(full_frame=True)) == (32, 64)


def test_full_frame_ignores_dynamic_centering(tmp_path):
    src = _video(tmp_path / "src.mp4")
    cap = cv2.VideoCapture(str(src))
    out = tmp_path / "clip.mp4"
    try:
        err = ClipExtractionService()._write_clip(
            cap=cap, out_path=out, start_frame=0, end_frame=6,
            cx=40.0, cy=20.0, preset=_preset(full_frame=True),
            # A centroid track that would otherwise pan the crop around.
            pose_centroid_x=np.linspace(0, 150, 12),
            pose_centroid_y=np.linspace(0, 70, 12),
            static_center=False,
        )
    finally:
        cap.release()
    assert err is None, err
    probe = cv2.VideoCapture(str(out))
    ok, frame = probe.read()
    probe.release()
    assert ok and frame.shape[:2] == (32, 64)


# ── Keeping every animal in view ──────────────────────────────────────
def _overlays(a_xy, b_xy, n=12):
    return [
        {"name": "a", "color": (0, 0, 255),
         "cx": np.full(n, a_xy[0], dtype=float), "cy": np.full(n, a_xy[1], dtype=float)},
        {"name": "b", "color": (255, 0, 0),
         "cx": np.full(n, b_xy[0], dtype=float), "cy": np.full(n, b_xy[1], dtype=float)},
    ]


def test_box_covers_both_animals():
    box = ClipExtractionService._all_animals_box(
        _overlays((100.0, 100.0), (300.0, 120.0)), 0, 10, margin=20, vid_w=640, vid_h=480,
    )
    x1, y1, x2, y2 = box
    assert x1 <= 100 and x2 >= 300
    assert y1 <= 100 and y2 >= 120
    assert (x2 - x1) == (y2 - y1)  # square, so a square output is not stretched


def test_box_is_never_tighter_than_the_preset_crop():
    box = ClipExtractionService._all_animals_box(
        _overlays((200.0, 200.0), (202.0, 201.0)), 0, 10, margin=50, vid_w=640, vid_h=480,
    )
    x1, _y1, x2, _y2 = box
    assert (x2 - x1) >= 100


def test_box_stays_inside_the_frame():
    box = ClipExtractionService._all_animals_box(
        _overlays((5.0, 5.0), (630.0, 470.0)), 0, 10, margin=40, vid_w=640, vid_h=480,
    )
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480


def test_single_animal_sessions_keep_the_normal_crop():
    one = _overlays((100.0, 100.0), (300.0, 120.0))[:1]
    assert ClipExtractionService._all_animals_box(one, 0, 10, 20, 640, 480) is None
    assert ClipExtractionService._all_animals_box(None, 0, 10, 20, 640, 480) is None


def test_animals_without_usable_positions_fall_back_to_the_crop():
    blind = _overlays((np.nan, np.nan), (np.nan, np.nan))
    assert ClipExtractionService._all_animals_box(blind, 0, 10, 20, 640, 480) is None


def _bright_video(path, w=640, h=360, n=12):
    """A uniformly light source, so the legend's black backing is measurable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.full((h, w, 3), 220, dtype=np.uint8))
    writer.release()
    return path


def _legend_frame(tmp_path, preset, names=("track_0", "track_1")):
    """Render one clip frame carrying the per-animal legend."""
    src = _bright_video(tmp_path / f"legend_{preset.resize_width}.mp4")
    overlays = [
        {"name": n, "color": (0, 0, 255), "cx": np.full(12, 100.0), "cy": np.full(12, 100.0)}
        for n in names
    ]
    cap = cv2.VideoCapture(str(src))
    out = tmp_path / f"legend_{preset.resize_width}_{preset.full_frame}.mp4"
    try:
        err = ClipExtractionService()._write_clip(
            cap=cap, out_path=out, start_frame=2, end_frame=6,
            cx=320.0, cy=180.0, preset=preset, individual_overlays=overlays,
        )
    finally:
        cap.release()
    assert err is None, err
    probe = cv2.VideoCapture(str(out))
    ok, frame = probe.read()
    probe.release()
    assert ok
    return frame


def _legend_extent(frame):
    """Fraction of the frame width/height taken by the black legend backing.

    Measured in the top-right quadrant, where the legend is drawn, the animal
    dots elsewhere are black-outlined too.
    """
    h, w = frame.shape[:2]
    quadrant = frame[: h // 2, w // 2:]
    ys, xs = np.nonzero(quadrant.max(axis=2) < 60)
    assert ys.size, "legend backing not found"
    return (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h


def test_legend_stays_a_margin_note_on_small_full_frame_clips(tmp_path):
    # The whole arena at 224 px wide: the legend must not eat the picture.
    fw, fh = _legend_extent(_legend_frame(tmp_path, _preset(resize_width=224, full_frame=True)))
    assert fw < 0.30 and fh < 0.25


def test_legend_grows_with_the_clip_but_stays_proportional(tmp_path):
    small = _legend_extent(_legend_frame(tmp_path, _preset(resize_width=224, full_frame=True)))
    large = _legend_extent(_legend_frame(tmp_path, _preset(resize_width=640, full_frame=True)))
    assert large[0] < 0.30 and large[1] < 0.25
    assert abs(large[0] - small[0]) < 0.15  # same relative footprint, not a fixed pixel size


def test_long_track_names_do_not_widen_the_legend_past_the_budget(tmp_path):
    frame = _legend_frame(
        tmp_path, _preset(resize_width=224, full_frame=True),
        names=("black_mouse_cage_left", "green_mouse_cage_right"),
    )
    fw, _fh = _legend_extent(frame)
    assert fw < 0.35
