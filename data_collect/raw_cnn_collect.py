"""Command-line entry point for raw CNN slip data collection.

Subcommands:
    collect  - fixed-light consecutive raw frame capture, default cmd=13
    windows  - build multi-frame sliding-window CSV files for sequence models

Typical usage:
    python data_collect/raw_cnn_collect.py collect --port COM7 --preview
    python data_collect/raw_cnn_collect.py windows --data-root data/raw_cnn --window 8
"""

import argparse

from raw_sequence_recorder import collect_sequence
from raw_windows import make_windows


def add_collect_args(sub):
    sub.add_argument("--out-root", default="slip_detection/data/raw_cnn", help="Root folder for sequence sessions")
    sub.add_argument("--session-dir", default=None, help="Use an explicit session directory")
    sub.add_argument("--session-prefix", default="session", help="Auto session folder prefix")
    sub.add_argument("--cam-index", type=int, default=0)
    sub.add_argument("--backend", choices=["default", "dshow", "msmf"], default="dshow")
    sub.add_argument("--strict-backend", action="store_true", help="Fail instead of falling back to default backend if requested backend cannot open")
    sub.add_argument("--camera-fps", type=float, default=0.0, help="Requested hardware camera FPS; 0 leaves driver default")
    sub.add_argument("--fourcc", default="", help="Requested camera FOURCC, for example MJPG; empty leaves driver default")
    sub.add_argument("--auto-exposure", type=float, default=None, help="Optional CAP_PROP_AUTO_EXPOSURE override; omit for camera default brightness")
    sub.add_argument("--exposure", type=float, default=None, help="Optional CAP_PROP_EXPOSURE override; omit for camera default brightness")
    sub.add_argument("--width", type=int, default=1280)
    sub.add_argument("--height", type=int, default=720)
    sub.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    sub.add_argument("--jpg-quality", type=int, default=95)
    sub.add_argument("--png-compression", type=int, default=1)
    sub.add_argument("--warmup-frames", type=int, default=5)
    sub.add_argument("--min-frames", type=int, default=6)
    sub.add_argument("--max-frames", type=int, default=0, help="0 means no frame limit")
    sub.add_argument("--duration-sec", type=float, default=0.0, help="0 means no duration limit")
    sub.add_argument("--target-fps", type=float, default=30.0, help="Expected camera FPS for metadata/logging; frames are not software-throttled")
    sub.add_argument("--port", default="COM7")
    sub.add_argument("--baud", type=int, default=115200)
    sub.add_argument("--serial-timeout", type=float, default=0.2)
    sub.add_argument("--light-cmd", type=int, default=13)
    sub.add_argument("--light-settle", type=float, default=0.2)
    sub.add_argument("--no-serial", action="store_true")
    sub.add_argument("--material", default="plastic", help="Object material for this session")
    sub.add_argument("--label-0-desc", default="variable-force press without slip", help="Human-readable meaning of label 0")
    sub.add_argument("--label-1-desc", default="translation slip", help="Human-readable meaning of label 1")
    sub.add_argument("--slip-motion", default="translation", help="Slip motion type, for example translation or torsion")
    sub.add_argument("--session-note", default="", help="Free-form note stored in metadata.csv")
    sub.add_argument("--default-label", type=int, choices=[-1, 0, 1], default=0)
    sub.add_argument("--label-schedule", default="", help="Timed labels as seconds:label pairs, e.g. 0:-1,3:0,6:1; -1 is ignored by windows")
    sub.add_argument("--preview", action="store_true")
    sub.add_argument("--no-wait-start", action="store_true", help="Start recording immediately when preview opens")
    sub.add_argument("--auto-reference", action="store_true", help="Allow using the warmup frame as reference without pressing b")
    sub.add_argument("--log-every", type=int, default=30)


def add_window_args(sub):
    sub.add_argument("--data-root", default="slip_detection/data/raw_cnn")
    sub.add_argument("--sessions", default="", help="Comma-separated session names; empty means all sessions")
    sub.add_argument("--window", type=int, default=8)
    sub.add_argument("--stride", type=int, default=8, help="Default sliding-window stride; larger values reduce near-duplicate windows")
    sub.add_argument("--train-stride", type=int, default=None, help="Stride for train windows; defaults to --stride")
    sub.add_argument("--val-stride", type=int, default=None, help="Stride for val windows; defaults to --stride")
    sub.add_argument("--label-policy", choices=["any", "last", "majority"], default="last")
    sub.add_argument("--train-csv", default="slip_detection/data/raw_cnn/train_windows.csv")
    sub.add_argument("--val-csv", default="slip_detection/data/raw_cnn/val_windows.csv")
    sub.add_argument("--val-ratio", type=float, default=0.2)
    sub.add_argument("--split-by-session", action="store_true", help="Split train/val by whole sessions instead of by overlapping window rows")
    sub.add_argument("--val-sessions", default="", help="Comma-separated validation sessions, e.g. session_017,session_018")
    sub.add_argument("--balance-labels", action="store_true", help="Downsample the larger label class so labels 0 and 1 are 1:1; with session split, applies to train rows")
    sub.add_argument("--balance-val-labels", action="store_true", help="Also balance validation labels; usually keep this off for honest metrics")
    sub.add_argument("--seed", type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Raw CNN collection utilities for NLiPsTac")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    collect_p = subparsers.add_parser("collect", help="Collect fixed-light consecutive raw frames")
    add_collect_args(collect_p)

    windows_p = subparsers.add_parser("windows", help="Build sliding-window CSVs for sequence models")
    add_window_args(windows_p)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "collect":
        collect_sequence(args)
    elif args.cmd == "windows":
        make_windows(args)


if __name__ == "__main__":
    main()







