import numpy as np
import pytest

from eventify import events_to_frame

GRAY = 128
# OpenCV convention: BGR channel order.
BLUE_BGR = (255, 0, 0)     # positive delta (brightening)
YELLOW_BGR = (0, 255, 255)  # negative delta (darkening)


def test_zero_delta_produces_gray_background():
    delta = np.zeros((6, 6), dtype=np.float32)
    img = events_to_frame(delta)
    assert img.shape == (6, 6, 3)
    assert img.dtype == np.uint8
    assert np.all(img == GRAY)


def test_positive_delta_renders_blue():
    delta = np.zeros((4, 4), dtype=np.float32)
    delta[1, 1] = 1.0  # single saturated positive event
    img = events_to_frame(delta)

    # The saturated pixel should be pure blue.
    assert tuple(img[1, 1]) == BLUE_BGR
    # Every other pixel stays gray.
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False
    assert np.all(img[mask] == GRAY)


def test_negative_delta_renders_yellow():
    delta = np.zeros((4, 4), dtype=np.float32)
    delta[2, 3] = -1.0
    img = events_to_frame(delta)

    assert tuple(img[2, 3]) == YELLOW_BGR
    mask = np.ones((4, 4), dtype=bool)
    mask[2, 3] = False
    assert np.all(img[mask] == GRAY)


def test_magnitude_scales_color_intensity_when_normalized():
    delta = np.zeros((1, 3), dtype=np.float32)
    delta[0, 0] = 0.25
    delta[0, 1] = 0.5
    delta[0, 2] = 1.0  # this one saturates under per-frame normalization

    img = events_to_frame(delta)  # normalize by default

    # Blue channel should grow monotonically with delta magnitude.
    b0, b1, b2 = img[0, 0, 0], img[0, 1, 0], img[0, 2, 0]
    assert b0 < b1 < b2
    assert b2 == 255
    # Red and green channels stay at gray baseline for pure blue.
    assert img[0, 2, 1] == 0 and img[0, 2, 2] == 0


def test_max_delta_override_uses_fixed_scale():
    delta = np.zeros((1, 2), dtype=np.float32)
    delta[0, 0] = 0.5
    delta[0, 1] = 1.0

    img = events_to_frame(delta, max_delta=1.0)

    # With max_delta=1.0, 0.5 -> half saturation, 1.0 -> full.
    assert img[0, 1, 0] == 255       # fully saturated blue
    assert 100 < img[0, 0, 0] < 200  # roughly half; interpolated from gray toward blue


def test_max_delta_clips_beyond_ceiling():
    delta = np.array([[2.0, -2.0]], dtype=np.float32)  # both exceed max_delta
    img = events_to_frame(delta, max_delta=1.0)

    assert tuple(img[0, 0]) == BLUE_BGR
    assert tuple(img[0, 1]) == YELLOW_BGR


def test_output_shape_matches_input():
    delta = np.zeros((13, 27), dtype=np.float32)
    img = events_to_frame(delta)
    assert img.shape == (13, 27, 3)


def test_normalize_all_zero_delta_stays_gray():
    # Edge case: no events at all — must not divide by zero.
    delta = np.zeros((5, 5), dtype=np.float32)
    img = events_to_frame(delta)
    assert np.all(img == GRAY)
