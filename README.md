# eventify

Convert video files or webcam feeds into simulated event-camera (DVS)
data using log-intensity differencing. A clean-room, dependency-light
reimplementation of the core idea behind v2e/ESIM — no CUDA, no PyTorch,
no pretrained models. Just NumPy, OpenCV, and h5py.

Two independent output paths:

1. **Visualization** — signed per-pixel log-delta rendered to color video.
   - **Blue** — brightening pixels (positive delta)
   - **Yellow** — darkening pixels (negative delta)
   - **Gray** — no event
2. **DVS export** — binary-polarity `(x, y, t_µs, p)` event tuples in an
   HDF5 layout compatible with the **DVS128 Gesture** dataset (as
   redistributed by Tonic/SpikingJelly).

## Install

Uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
uv sync
```

## CLI

Three subcommands, wired up as the `eventify` entry point:

```bash
# 1. Render a video file to an event-visualized video (magnitude preserved)
uv run eventify convert input.mp4 output.mp4
uv run eventify convert input.mp4 output.mp4 --threshold 0.2 --max-delta 1.0

# 2. Live webcam stream (press q to quit)
uv run eventify webcam
uv run eventify webcam --device 1 --threshold 0.1

# 3. Export DVS-Gesture-compatible events to HDF5
uv run eventify export input.mp4 events.h5
uv run eventify export input.mp4 events.h5 --sensor-size 128,128 --threshold 0.15
```

After `pip install .` the same commands work as `eventify convert ...`.

## Library

### Visualization path (magnitude preserved)

```python
import cv2
from eventify import frame_to_events, events_to_frame, video_to_events

prev = cv2.imread("a.png", cv2.IMREAD_GRAYSCALE)
curr = cv2.imread("b.png", cv2.IMREAD_GRAYSCALE)
delta = frame_to_events(prev, curr, c_thresh=0.15)  # signed float32 map
img = events_to_frame(delta)                        # BGR uint8 preview

for timestamp, delta in video_to_events("video.mp4"):
    ...
```

### DVS export path (binary polarity)

```python
import numpy as np
from eventify import (
    frame_to_event_tuples,
    video_to_event_stream,
    write_hdf5,
    EVENT_DTYPE,
)

# Per-pair event tuples
events = frame_to_event_tuples(prev, curr, prev_t_us=0, curr_t_us=1000)
# events["x"], events["y"], events["t"], events["p"]  — p ∈ {0, 1}

# Full stream
chunks = list(video_to_event_stream("video.mp4", sensor_size=(128, 128)))
all_events = np.concatenate(chunks)
write_hdf5("out.h5", all_events, sensor_shape=(128, 128))
```

## API reference

### Visualization

- **`frame_to_events(prev, curr, c_thresh=0.15, eps=1.0)`** — returns a
  2D `float32` array of `log(curr + eps) − log(prev + eps)` with values
  below `c_thresh` in magnitude zeroed. Positive = brightening,
  negative = darkening.
- **`events_to_frame(delta, max_delta=None)`** — returns an `H×W×3` BGR
  `uint8` image. Per-frame normalized by default; pass `max_delta` for a
  fixed saturation ceiling (magnitudes are clipped).
- **`video_to_events(source, c_thresh=0.15, grayscale=True)`** —
  generator yielding `(timestamp_seconds, delta)` tuples.

### DVS export

- **`frame_to_event_tuples(prev, curr, prev_t_us, curr_t_us, c_thresh=0.15, eps=1.0, sensor_size=None)`** —
  returns a NumPy structured array of dtype `EVENT_DTYPE` with fields
  `(x: i2, y: i2, t: i8, p: i1)`. Polarity is binary (0 = OFF,
  1 = ON). Timestamps are uniformly distributed across
  `[prev_t_us, curr_t_us]`.
- **`video_to_event_stream(source, c_thresh=0.15, sensor_size=None)`** —
  generator yielding one structured event array per frame-pair.
  Timestamps across chunks are monotonic microseconds. Pass
  `sensor_size=(w, h)` to resize frames before event generation.
- **`write_hdf5(path, events, sensor_shape)`** — writes events in the
  DVS-Gesture reprocessed layout:

  ```
  /events                          (group)
      .attrs["sensor_shape"]  (2,) i8   — (height, width)
      /xs   i2   x coords
      /ys   i2   y coords
      /ts   i8   timestamps (µs)
      /ps   i1   polarities ∈ {0, 1}
  ```

## Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
