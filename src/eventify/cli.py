"""Command-line entry point for eventify."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import cv2
import numpy as np

from eventify.core import events_to_frame, video_to_events
from eventify.dvs import video_to_event_stream, write_hdf5


def _parse_sensor_size(spec: str) -> tuple[int, int]:
    """Parse a "W,H" string into an (int, int) tuple."""
    try:
        w_str, h_str = spec.split(",")
        return int(w_str), int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--sensor-size must be 'W,H' (got {spec!r})"
        ) from exc


def _convert(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"error: could not open input video: {args.input}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        print(f"error: could not open output for writing: {args.output}", file=sys.stderr)
        return 1

    count = 0
    try:
        for _, delta in video_to_events(args.input, c_thresh=args.threshold):
            frame = events_to_frame(delta, max_delta=args.max_delta)
            writer.write(frame)
            count += 1
    finally:
        writer.release()

    print(f"wrote {count} event frames to {args.output}")
    return 0


def _webcam(args: argparse.Namespace) -> int:
    window = "eventify — press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for _, delta in video_to_events(args.device, c_thresh=args.threshold):
            frame = events_to_frame(delta, max_delta=args.max_delta)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def _export(args: argparse.Namespace) -> int:
    # Collect chunks; DVS files are typically small enough to hold in memory.
    # If this grows, switch to incremental h5py resizing.
    chunks = []
    total = 0
    resolved_shape = None
    for chunk in video_to_event_stream(
        args.input, c_thresh=args.threshold, sensor_size=args.sensor_size
    ):
        chunks.append(chunk)
        total += len(chunk)

    if args.sensor_size is not None:
        w, h = args.sensor_size
        resolved_shape = (h, w)
    else:
        cap = cv2.VideoCapture(args.input)
        resolved_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        cap.release()

    from eventify.dvs import EVENT_DTYPE
    events = np.concatenate(chunks) if chunks else np.zeros(0, dtype=EVENT_DTYPE)
    write_hdf5(args.output, events, sensor_shape=resolved_shape)
    print(f"wrote {total} events to {args.output} (sensor_shape={resolved_shape})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eventify",
        description="Convert video or webcam feeds into simulated event-camera data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert a video file to an event-rendered video.")
    convert.add_argument("input", help="Path to the input video file.")
    convert.add_argument("output", help="Path to write the event-rendered video (e.g. out.mp4).")
    convert.add_argument("--threshold", type=float, default=0.15, help="Log-intensity event threshold (default: 0.15).")
    convert.add_argument("--max-delta", type=float, default=None, help="Fixed saturation ceiling; omit for per-frame normalization.")
    convert.set_defaults(func=_convert)

    webcam = sub.add_parser("webcam", help="Show live event stream from the webcam.")
    webcam.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0).")
    webcam.add_argument("--threshold", type=float, default=0.15, help="Log-intensity event threshold (default: 0.15).")
    webcam.add_argument("--max-delta", type=float, default=None, help="Fixed saturation ceiling; omit for per-frame normalization.")
    webcam.set_defaults(func=_webcam)

    export = sub.add_parser(
        "export",
        help="Export a video's events to a DVS-Gesture-compatible HDF5 file.",
    )
    export.add_argument("input", help="Path to the input video file.")
    export.add_argument("output", help="Path to write the events HDF5 file (e.g. out.h5).")
    export.add_argument("--threshold", type=float, default=0.15, help="Log-intensity event threshold (default: 0.15).")
    export.add_argument(
        "--sensor-size",
        type=_parse_sensor_size,
        default=None,
        metavar="W,H",
        help="Override sensor resolution as 'W,H' (default: source video's native resolution).",
    )
    export.set_defaults(func=_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
