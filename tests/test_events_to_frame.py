import numpy as np
import pytest

from eventify import events_to_frame

# DVS-style palette (BGR): near-black background, deep royal blue for ON,
# warm amber/gold for OFF. Matches the reference visualization the palette
# is drawn from.
BG_BGR = (0, 0, 0)
ON_BGR = (180, 70, 0)      # deep blue
OFF_BGR = (0, 170, 220)    # amber/gold


def _all_equal(img, bgr):
    return (
        np.all(img[..., 0] == bgr[0])
        and np.all(img[..., 1] == bgr[1])
        and np.all(img[..., 2] == bgr[2])
    )


def test_zero_delta_produces_black_background():
    delta = np.zeros((6, 6), dtype=np.float32)
    img = events_to_frame(delta)
    assert img.shape == (6, 6, 3)
    assert img.dtype == np.uint8
    assert _all_equal(img, BG_BGR)


def test_positive_delta_renders_deep_blue_at_saturation():
    delta = np.zeros((4, 4), dtype=np.float32)
    delta[1, 1] = 1.0
    img = events_to_frame(delta)

    assert tuple(img[1, 1]) == ON_BGR
    # Every other pixel stays background.
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False
    assert _all_equal(img[mask], BG_BGR)


def test_negative_delta_renders_amber_at_saturation():
    delta = np.zeros((4, 4), dtype=np.float32)
    delta[2, 3] = -1.0
    img = events_to_frame(delta)

    assert tuple(img[2, 3]) == OFF_BGR
    mask = np.ones((4, 4), dtype=bool)
    mask[2, 3] = False
    assert _all_equal(img[mask], BG_BGR)


def test_magnitude_scales_color_intensity_when_normalized():
    delta = np.zeros((1, 3), dtype=np.float32)
    delta[0, 0] = 0.25
    delta[0, 1] = 0.5
    delta[0, 2] = 1.0  # saturates under per-frame normalization

    img = events_to_frame(delta)  # normalize by default

    # Blue channel should grow monotonically with delta magnitude.
    b0, b1, b2 = img[0, 0, 0], img[0, 1, 0], img[0, 2, 0]
    assert b0 < b1 < b2
    # Saturated pixel matches full ON color.
    assert tuple(img[0, 2]) == ON_BGR


def test_max_delta_override_uses_fixed_scale():
    delta = np.zeros((1, 2), dtype=np.float32)
    delta[0, 0] = 0.5
    delta[0, 1] = 1.0

    img = events_to_frame(delta, max_delta=1.0)

    # Fully saturated pixel matches full ON color.
    assert tuple(img[0, 1]) == ON_BGR
    # Half-magnitude pixel: blue channel is ~half of the ON blue value.
    assert 60 < img[0, 0, 0] < 130


def test_max_delta_clips_beyond_ceiling():
    delta = np.array([[2.0, -2.0]], dtype=np.float32)
    img = events_to_frame(delta, max_delta=1.0)

    assert tuple(img[0, 0]) == ON_BGR
    assert tuple(img[0, 1]) == OFF_BGR


def test_output_shape_matches_input():
    delta = np.zeros((13, 27), dtype=np.float32)
    img = events_to_frame(delta)
    assert img.shape == (13, 27, 3)


def test_normalize_all_zero_delta_stays_background():
    # Edge case: no events at all — must not divide by zero.
    delta = np.zeros((5, 5), dtype=np.float32)
    img = events_to_frame(delta)
    assert _all_equal(img, BG_BGR)
