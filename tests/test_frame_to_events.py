import numpy as np
import pytest

from eventify import frame_to_events


def _uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


def test_identical_frames_produce_no_events():
    frame = _uniform((8, 8), 100.0)
    delta = frame_to_events(frame, frame.copy(), c_thresh=0.15)
    assert delta.shape == frame.shape
    assert np.all(delta == 0.0)


def test_brightening_region_produces_positive_delta_only_there():
    prev = _uniform((8, 8), 50.0)
    curr = prev.copy()
    curr[2:5, 3:6] = 200.0  # brighten a rectangle

    delta = frame_to_events(prev, curr, c_thresh=0.15)

    # Positive deltas only inside the brightened region.
    assert np.all(delta[2:5, 3:6] > 0)
    # Everywhere else: no event (untouched pixels).
    mask = np.ones_like(delta, dtype=bool)
    mask[2:5, 3:6] = False
    assert np.all(delta[mask] == 0.0)


def test_darkening_region_produces_negative_delta_only_there():
    prev = _uniform((8, 8), 200.0)
    curr = prev.copy()
    curr[1:4, 2:5] = 30.0  # darken a rectangle

    delta = frame_to_events(prev, curr, c_thresh=0.15)

    assert np.all(delta[1:4, 2:5] < 0)
    mask = np.ones_like(delta, dtype=bool)
    mask[1:4, 2:5] = False
    assert np.all(delta[mask] == 0.0)


def test_higher_threshold_produces_fewer_or_equal_events():
    rng = np.random.default_rng(0)
    prev = rng.uniform(20, 220, size=(32, 32)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 30, size=prev.shape), 1, 255).astype(np.float32)

    low = frame_to_events(prev, curr, c_thresh=0.05)
    high = frame_to_events(prev, curr, c_thresh=0.5)

    low_events = np.count_nonzero(low)
    high_events = np.count_nonzero(high)
    assert high_events <= low_events
    # And a threshold that exceeds any possible log-delta should kill all events.
    huge = frame_to_events(prev, curr, c_thresh=1000.0)
    assert np.count_nonzero(huge) == 0


def test_delta_magnitude_reflects_log_intensity_change():
    prev = _uniform((4, 4), 50.0)
    curr = _uniform((4, 4), 200.0)
    delta = frame_to_events(prev, curr, c_thresh=0.15, eps=1.0)

    expected = np.log(200.0 + 1.0) - np.log(50.0 + 1.0)
    assert np.allclose(delta, expected, atol=1e-5)


def test_sub_threshold_change_is_zeroed():
    prev = _uniform((4, 4), 100.0)
    # A tiny bump: log(101/101)~0.01, well below threshold.
    curr = _uniform((4, 4), 101.0)
    delta = frame_to_events(prev, curr, c_thresh=0.15)
    assert np.all(delta == 0.0)


def test_accepts_uint8_input():
    prev = np.full((4, 4), 50, dtype=np.uint8)
    curr = np.full((4, 4), 200, dtype=np.uint8)
    delta = frame_to_events(prev, curr, c_thresh=0.15)
    assert delta.dtype.kind == "f"
    assert np.all(delta > 0)


def test_shape_mismatch_raises():
    prev = _uniform((4, 4), 100.0)
    curr = _uniform((5, 5), 100.0)
    with pytest.raises(ValueError):
        frame_to_events(prev, curr)
