"""DVS-Gesture-compatible event export path.

Emits individual ``(x, y, t_us, polarity)`` tuples with binary polarity,
matching the record format used by the DVS128 Gesture dataset. Kept
strictly separate from the intensity-modulated video render path in
``eventify.core``: rendering keeps magnitude for visualization, this
module discards it because real DVS sensors only report threshold
crossings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import h5py
import numpy as np

# NumPy structured dtype for a single DVS event.
# int16 coords cover any realistic sensor; int64 for µs timestamps to
# hold long captures without overflow. Polarity ∈ {0, 1}.
EVENT_DTYPE = np.dtype(
    [("x", "<i2"), ("y", "<i2"), ("t", "<i8"), ("p", "<i1")]
)


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32, copy=False)


def _maybe_resize(frame: np.ndarray, sensor_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if sensor_size is None:
        return frame
    w, h = sensor_size
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def frame_to_event_tuples(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_t_us: int,
    curr_t_us: int,
    c_thresh: float = 0.15,
    eps: float = 1.0,
    sensor_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Emit binary-polarity DVS event tuples for a single frame pair.

    Timestamps for the emitted events are uniformly distributed across
    ``[prev_t_us, curr_t_us]`` in raster order. The dataset ships as
    binary polarity, so magnitude is intentionally discarded here.
    """
    if prev_frame.shape != curr_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )
    if curr_t_us < prev_t_us:
        raise ValueError(
            f"curr_t_us ({curr_t_us}) must be >= prev_t_us ({prev_t_us})"
        )

    prev = _maybe_resize(_to_gray_float(prev_frame), sensor_size)
    curr = _maybe_resize(_to_gray_float(curr_frame), sensor_size)

    delta = np.log(curr + eps) - np.log(prev + eps)

    on_mask = delta > c_thresh
    off_mask = delta < -c_thresh
    fired = on_mask | off_mask

    n = int(fired.sum())
    events = np.zeros(n, dtype=EVENT_DTYPE)
    if n == 0:
        return events

    ys, xs = np.nonzero(fired)  # numpy is row-major: (row=y, col=x)
    events["x"] = xs.astype(np.int16)
    events["y"] = ys.astype(np.int16)
    events["p"] = on_mask[ys, xs].astype(np.int8)  # 1 for ON, 0 for OFF

    # Uniform spread across the inter-frame interval, in raster order.
    interval = curr_t_us - prev_t_us
    if n == 1:
        events["t"] = prev_t_us + interval // 2
    else:
        events["t"] = (prev_t_us + np.linspace(0, interval, n, dtype=np.float64)).astype(np.int64)

    return events


def video_to_event_stream(
    source: Union[str, int],
    c_thresh: float = 0.15,
    sensor_size: Optional[Tuple[int, int]] = None,
) -> Generator[np.ndarray, None, None]:
    """Yield per-frame-pair structured event arrays from a video or webcam.

    ``source`` is a file path or an integer webcam device index. Events
    across all chunks form a monotonic microsecond stream. Frames are
    kept at native resolution unless ``sensor_size=(w, h)`` overrides it.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_period_us = int(1_000_000 / fps) if fps > 0 else 33_333  # ~30 FPS webcam fallback

    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            return
        prev_t_us = 0

        # Cache grayscale conversion so we don't recompute for the next iteration.
        prev_gray_full = _to_gray_float(prev)
        prev_processed = _maybe_resize(prev_gray_full, sensor_size)

        frame_idx = 1
        while True:
            ok, curr = cap.read()
            if not ok or curr is None:
                break

            curr_gray_full = _to_gray_float(curr)
            curr_processed = _maybe_resize(curr_gray_full, sensor_size)

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                curr_t_us = int(pos_ms * 1000)
            else:
                curr_t_us = frame_idx * frame_period_us

            # Pass already-resized frames to avoid a second resize inside the helper.
            events = frame_to_event_tuples(
                prev_processed,
                curr_processed,
                prev_t_us=prev_t_us,
                curr_t_us=curr_t_us,
                c_thresh=c_thresh,
                sensor_size=None,
            )
            yield events

            prev_processed = curr_processed
            prev_t_us = curr_t_us
            frame_idx += 1
    finally:
        cap.release()


def write_hdf5(
    path: Union[str, os.PathLike],
    events: np.ndarray,
    sensor_shape: Tuple[int, int],
) -> None:
    """Write events to an HDF5 file using the DVS-Gesture reprocessed layout.

    Layout::

        /events                     (group)
            .attrs["sensor_shape"]  (2,) int  – (height, width)
            /xs   int16   x coords
            /ys   int16   y coords
            /ts   int64   timestamps (µs)
            /ps   int8    polarities ∈ {0, 1}
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        grp = f.create_group("events")
        grp.attrs["sensor_shape"] = np.array(sensor_shape, dtype=np.int64)
        grp.create_dataset("xs", data=events["x"].astype(np.int16), dtype="<i2")
        grp.create_dataset("ys", data=events["y"].astype(np.int16), dtype="<i2")
        grp.create_dataset("ts", data=events["t"].astype(np.int64), dtype="<i8")
        grp.create_dataset("ps", data=events["p"].astype(np.int8), dtype="<i1")
