import numpy as np
import pytest

from eventify import EVENT_DTYPE, frame_to_event_tuples


def _uniform(shape, value):
    return np.full(shape, value, dtype=np.float32)


def test_returns_structured_event_array():
    prev = _uniform((8, 8), 100.0)
    curr = prev.copy()
    curr[2:4, 3:5] = 250.0
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert isinstance(events, np.ndarray)
    assert events.dtype == EVENT_DTYPE
    # Fields present and named correctly.
    assert set(events.dtype.names) == {"x", "y", "t", "p"}


def test_identical_frames_produce_no_events():
    prev = _uniform((8, 8), 100.0)
    events = frame_to_event_tuples(prev, prev.copy(), prev_t_us=0, curr_t_us=1000)
    assert len(events) == 0


def test_brightening_pixel_yields_polarity_1():
    prev = _uniform((4, 4), 50.0)
    curr = prev.copy()
    curr[1, 2] = 250.0
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert len(events) == 1
    e = events[0]
    assert (e["x"], e["y"]) == (2, 1)  # (x=col, y=row)
    assert e["p"] == 1


def test_darkening_pixel_yields_polarity_0():
    prev = _uniform((4, 4), 250.0)
    curr = prev.copy()
    curr[3, 0] = 20.0
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert len(events) == 1
    e = events[0]
    assert (e["x"], e["y"]) == (0, 3)
    assert e["p"] == 0


def test_polarity_is_binary_only():
    rng = np.random.default_rng(0)
    prev = rng.uniform(20, 220, size=(32, 32)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 50, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert set(np.unique(events["p"]).tolist()).issubset({0, 1})


def test_timestamps_lie_in_interval():
    rng = np.random.default_rng(0)
    prev = rng.uniform(20, 220, size=(16, 16)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 60, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=1000, curr_t_us=2000)

    assert len(events) > 0
    assert np.all(events["t"] >= 1000)
    assert np.all(events["t"] <= 2000)


def test_timestamps_uniformly_distributed():
    # A large event count should span most of the interval, not cluster at one end.
    rng = np.random.default_rng(1)
    prev = rng.uniform(20, 220, size=(64, 64)).astype(np.float32)
    curr = np.clip(prev + rng.normal(0, 80, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=10000)

    ts = events["t"]
    assert len(ts) >= 100
    # Spread should cover a large fraction of the interval.
    assert ts.min() < 2000
    assert ts.max() > 8000


def test_coordinates_within_frame_bounds():
    rng = np.random.default_rng(2)
    prev = rng.uniform(20, 220, size=(16, 24)).astype(np.float32)  # h=16, w=24
    curr = np.clip(prev + rng.normal(0, 60, size=prev.shape), 1, 255).astype(np.float32)
    events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)

    assert np.all(events["x"] >= 0) and np.all(events["x"] < 24)
    assert np.all(events["y"] >= 0) and np.all(events["y"] < 16)


def test_sensor_size_resizes_before_event_gen():
    prev = _uniform((100, 200), 100.0)
    curr = prev.copy()
    curr[:] = 250.0  # global brightening -> every pixel fires

    events = frame_to_event_tuples(
        prev, curr, prev_t_us=0, curr_t_us=1000, sensor_size=(32, 32)  # (w, h)
    )
    # After resize to 32x32, at most 32*32 events.
    assert np.all(events["x"] < 32)
    assert np.all(events["y"] < 32)
    assert len(events) <= 32 * 32


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        frame_to_event_tuples(
            _uniform((4, 4), 100.0),
            _uniform((5, 5), 100.0),
            prev_t_us=0,
            curr_t_us=1000,
        )


def test_negative_interval_raises():
    prev = _uniform((4, 4), 100.0)
    curr = _uniform((4, 4), 200.0)
    with pytest.raises(ValueError):
        frame_to_event_tuples(prev, curr, prev_t_us=1000, curr_t_us=500)
