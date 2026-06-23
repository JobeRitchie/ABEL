from abel.services.preprocessing_service import ClipExtractionService


def test_scaled_crop_margin_grows_for_higher_resolution() -> None:
    base = 80
    low_res_margin = ClipExtractionService._scaled_crop_margin(base, frame_width=640, frame_height=480)
    high_res_margin = ClipExtractionService._scaled_crop_margin(base, frame_width=1920, frame_height=1080)

    # Area target is +25%, so linear margin grows by sqrt(1.25) at baseline resolution.
    assert low_res_margin == 89
    assert high_res_margin > low_res_margin


def test_scaled_crop_margin_never_exceeds_half_short_edge() -> None:
    margin = ClipExtractionService._scaled_crop_margin(800, frame_width=640, frame_height=480)
    assert margin <= (480 // 2 - 1)
    assert margin >= 8
