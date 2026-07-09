import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from eventify import video_to_events


@pytest.fixture
def synthetic_video(tmp_path):
    """Write a 5-frame 32x32 video where each frame gets progressively brighter."""
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32), isColor=True)
    assert writer.isOpened(), "Could not open VideoWriter — codec unavailable?"

    frames = []
    for i in range(5):
        val = 40 + i * 40  # 40, 80, 120, 160, 200
        frame = np.full((32, 32, 3), val, dtype=np.uint8)
        writer.write(frame)
        frames.append(val)
    writer.release()
    return path, frames


def test_yields_one_fewer_event_frame_than_source(synthetic_video):
    path, frames = synthetic_video
    events = list(video_to_events(str(path)))
    # N source frames -> N-1 deltas.
    assert len(events) == len(frames) - 1


def test_yields_timestamp_and_delta_pairs(synthetic_video):
    path, _ = synthetic_video
    events = list(video_to_events(str(path)))
    for item in events:
        assert len(item) == 2
        ts, delta = item
        assert isinstance(ts, float)
        assert isinstance(delta, np.ndarray)
        assert delta.ndim == 2  # grayscale delta map


def test_timestamps_are_monotonically_increasing(synthetic_video):
    path, _ = synthetic_video
    events = list(video_to_events(str(path)))
    timestamps = [ts for ts, _ in events]
    assert timestamps == sorted(timestamps)
    assert all(t2 > t1 for t1, t2 in zip(timestamps, timestamps[1:]))


def test_brightening_video_produces_positive_deltas(synthetic_video):
    path, _ = synthetic_video
    events = list(video_to_events(str(path), c_thresh=0.05))
    # Every frame in the fixture brightens uniformly -> every non-zero delta is positive.
    for _, delta in events:
        assert np.all(delta >= 0)
        assert np.count_nonzero(delta) > 0


def test_grayscale_delta_shape_matches_frame_dims(synthetic_video):
    path, _ = synthetic_video
    events = list(video_to_events(str(path), grayscale=True))
    for _, delta in events:
        assert delta.shape == (32, 32)


def test_end_of_stream_is_handled_gracefully(synthetic_video):
    path, _ = synthetic_video
    # Fully draining the generator must not raise.
    gen = video_to_events(str(path))
    count = 0
    for _ in gen:
        count += 1
    assert count > 0
    # Re-draining a spent generator yields nothing.
    assert list(gen) == []


def test_nonexistent_source_raises(tmp_path):
    missing = tmp_path / "does_not_exist.avi"
    with pytest.raises((IOError, RuntimeError, ValueError)):
        list(video_to_events(str(missing)))


def test_accepts_integer_device_index_type(monkeypatch):
    """We don't open a real webcam in tests, but the API must accept an int."""
    opened = {}

    class FakeCap:
        def __init__(self, src):
            opened["src"] = src

        def isOpened(self):
            return False  # short-circuit so no frames read

        def read(self):
            return False, None

        def get(self, prop):
            return 0.0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    with pytest.raises((IOError, RuntimeError, ValueError)):
        list(video_to_events(0))
    assert opened["src"] == 0
