"""Core event-camera simulation primitives.

Simulates DVS-style output by thresholding log-intensity change between
consecutive frames. Unlike a binary polarity map, events here carry the
signed log-delta magnitude so downstream renderers can express how strong
a change was, not just its sign.
"""

from __future__ import annotations

from typing import Generator, Union

import cv2
import numpy as np

# BGR triplets — OpenCV's native channel order.
_GRAY = np.array([128, 128, 128], dtype=np.float32)
_BLUE = np.array([255, 0, 0], dtype=np.float32)     # positive delta
_YELLOW = np.array([0, 255, 255], dtype=np.float32)  # negative delta


def frame_to_events(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    c_thresh: float = 0.15,
    eps: float = 1.0,
) -> np.ndarray:
    """Return a signed log-intensity delta map, thresholded at ``c_thresh``.

    Values with ``|delta| < c_thresh`` are zeroed out. Positive values
    correspond to brightening pixels (ON events), negative values to
    darkening (OFF events).
    """
    if prev_frame.shape != curr_frame.shape:
        raise ValueError(
            f"Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}"
        )

    prev = prev_frame.astype(np.float32, copy=False)
    curr = curr_frame.astype(np.float32, copy=False)

    delta = np.log(curr + eps) - np.log(prev + eps)
    delta[np.abs(delta) < c_thresh] = 0.0
    return delta


def events_to_frame(
    delta: np.ndarray,
    max_delta: Union[float, None] = None,
) -> np.ndarray:
    """Render a delta map into a BGR uint8 image.

    Positive deltas fade from gray to blue, negative deltas fade to yellow.
    If ``max_delta`` is None, saturation is scaled per-frame to the largest
    absolute delta present. Otherwise magnitudes are clipped to ``max_delta``.
    """
    h, w = delta.shape
    abs_delta = np.abs(delta)

    if max_delta is None:
        peak = float(abs_delta.max()) if abs_delta.size else 0.0
    else:
        peak = float(max_delta)

    if peak <= 0.0:
        img = np.broadcast_to(_GRAY, (h, w, 3)).astype(np.uint8).copy()
        return img

    intensity = np.clip(abs_delta / peak, 0.0, 1.0)[..., None]  # (h, w, 1)

    # Choose the target color per-pixel based on delta sign; sign==0 -> gray,
    # which produces intensity==0 anyway, so the choice is irrelevant there.
    target = np.where(delta[..., None] >= 0, _BLUE, _YELLOW)  # (h, w, 3)

    img = _GRAY + intensity * (target - _GRAY)
    return np.clip(img, 0, 255).astype(np.uint8)


def video_to_events(
    source: Union[str, int],
    c_thresh: float = 0.15,
    grayscale: bool = True,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield ``(timestamp_seconds, delta_map)`` pairs from a video or webcam.

    ``source`` may be a filesystem path or an integer webcam device index.
    Timestamps come from the capture's ``CAP_PROP_POS_MSEC`` when available;
    for webcams that don't report it, a monotonic frame-counter fallback
    scaled by the reported FPS is used.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_period = 1.0 / fps if fps > 0 else 1.0 / 30.0  # webcam fallback

    try:
        ok, prev = cap.read()
        if not ok or prev is None:
            return

        if grayscale:
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        frame_idx = 1
        while True:
            ok, curr = cap.read()
            if not ok or curr is None:
                break

            if grayscale:
                curr_proc = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            else:
                curr_proc = curr

            delta = frame_to_events(prev, curr_proc, c_thresh=c_thresh)

            # Prefer the container's own timestamp; fall back for live sources.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                ts = pos_ms / 1000.0
            else:
                ts = frame_idx * frame_period

            yield float(ts), delta

            prev = curr_proc
            frame_idx += 1
    finally:
        cap.release()
