"""Continuous raw-frame sequence recording for slip CNN data."""

import csv
import os
import time
from datetime import datetime

import cv2

from SerialCamController import SerialCamController


def normalize_rel(path: str) -> str:
    return path.replace("\\", "/")


def next_session_dir(root: str, prefix: str) -> str:
    os.makedirs(root, exist_ok=True)
    existing = []
    for name in os.listdir(root):
        if name.startswith(prefix + "_"):
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing.append(int(suffix))
    idx = max(existing, default=0) + 1
    return os.path.join(root, f"{prefix}_{idx:03d}")


def image_write_params(args):
    ext = args.image_ext.lower()
    if ext in ("jpg", "jpeg"):
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)]
    if ext == "png":
        return [int(cv2.IMWRITE_PNG_COMPRESSION), int(args.png_compression)]
    return []


def checked_imwrite(path: str, image, params) -> None:
    ok = cv2.imwrite(path, image, params)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def reference_filename(args) -> str:
    return f"reference.{args.image_ext}"


def parse_label_schedule(value: str):
    if not value:
        return []
    schedule = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label schedule item: {item!r}; expected seconds:label")
        seconds_text, label_text = item.split(":", 1)
        seconds = float(seconds_text.strip())
        label = int(label_text.strip())
        if seconds < 0:
            raise ValueError("label schedule seconds must be non-negative")
        if label not in (-1, 0, 1):
            raise ValueError("label schedule labels must be -1, 0, or 1")
        schedule.append((seconds, label))
    schedule.sort(key=lambda item: item[0])
    return schedule


def scheduled_label_at(elapsed_sec: float, schedule, current_label: int) -> int:
    label = current_label
    for seconds, scheduled_label in schedule:
        if elapsed_sec >= seconds:
            label = scheduled_label
        else:
            break
    return label


def frame_timing_stats(frame_rows):
    times = [float(row["timestamp_sec"]) for row in frame_rows]
    if len(times) < 2:
        return {
            "timestamp_fps": 0.0,
            "mean_interval_ms": 0.0,
            "min_interval_ms": 0.0,
            "max_interval_ms": 0.0,
        }
    intervals = [b - a for a, b in zip(times, times[1:])]
    span = max(1e-9, times[-1] - times[0])
    return {
        "timestamp_fps": (len(times) - 1) / span,
        "mean_interval_ms": 1000.0 * sum(intervals) / len(intervals),
        "min_interval_ms": 1000.0 * min(intervals),
        "max_interval_ms": 1000.0 * max(intervals),
    }


def write_metadata(path: str, args, actual_width: int, actual_height: int, frames: int, fps: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("raw_sequence_metadata\n")
        f.write(f"created_at,{datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"material,{args.material}\n")
        f.write(f"label_0_desc,{args.label_0_desc}\n")
        f.write(f"label_1_desc,{args.label_1_desc}\n")
        f.write(f"slip_motion,{args.slip_motion}\n")
        f.write(f"session_note,{args.session_note}\n")
        f.write(f"camera_index,{args.cam_index}\n")
        f.write(f"requested_width,{args.width}\n")
        f.write(f"requested_height,{args.height}\n")
        f.write(f"actual_width,{actual_width}\n")
        f.write(f"actual_height,{actual_height}\n")
        f.write(f"light_cmd,{args.light_cmd}\n")
        f.write(f"target_fps,{args.target_fps if args.target_fps else ''}\n")
        f.write(f"frames,{frames}\n")
        f.write(f"actual_fps,{fps:.3f}\n")
        f.write(f"reference,{reference_filename(args)}\n")


def read_required_frame(ctrl: SerialCamController, context: str):
    ok, frame = ctrl.cap_frame()
    if not ok:
        raise RuntimeError(f"Cannot read camera frame during {context}")
    return frame


def warmup_reference(ctrl: SerialCamController, warmup_frames: int):
    reference_frame = None
    for _ in range(max(0, warmup_frames)):
        ok, frame = ctrl.cap_frame()
        if ok:
            reference_frame = frame.copy()
    if reference_frame is None:
        reference_frame = read_required_frame(ctrl, "reference warmup").copy()
    return reference_frame


def wait_for_reference_and_start(ctrl: SerialCamController, reference_frame, args):
    reference_confirmed = bool(args.auto_reference)
    print("[INFO] preview ready. Put sensor in non-contact state and press b to capture this session's reference.")
    print("[INFO] after reference is captured, press r or SPACE to start recording; q/ESC to exit.")
    while True:
        ok, frame = ctrl.cap_frame()
        if not ok:
            print("[WARN] failed to read frame before start")
            time.sleep(0.02)
            continue
        if reference_confirmed:
            prompt = "reference OK: press r/SPACE to record"
            color = (0, 255, 0)
        else:
            prompt = "non-contact: press b to capture reference"
            color = (0, 255, 255)
        cv2.putText(frame, prompt, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.imshow("raw sequence capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            reference_frame = frame.copy()
            reference_confirmed = True
            print("[REF] reference captured for this session")
            continue
        if key in (ord("r"), ord(" ")):
            if not reference_confirmed:
                print("[WARN] capture reference first: put sensor in non-contact state and press b")
                continue
            print("[INFO] recording started")
            return reference_frame, True
        if key in (27, ord("q")):
            print("[INFO] recording cancelled before start")
            return reference_frame, False


def collect_sequence(args) -> None:
    session_dir = args.session_dir or next_session_dir(args.out_root, args.session_prefix)
    rgb_dir = os.path.join(session_dir, "rgb")
    frame_rows = []
    captured_frames = []
    frame_label = int(args.default_label)
    label_schedule = parse_label_schedule(args.label_schedule)

    with SerialCamController(
        port=args.port,
        baud=args.baud,
        cam_index=args.cam_index,
        serial_timeout=args.serial_timeout,
        width=args.width,
        height=args.height,
        backend=args.backend,
        no_serial=args.no_serial,
        camera_fps=args.camera_fps,
        fourcc=args.fourcc,
        auto_exposure=args.auto_exposure,
        exposure=args.exposure,
        strict_backend=args.strict_backend,
    ) as ctrl:
        if not args.no_serial:
            ctrl.light(args.light_cmd)
            time.sleep(args.light_settle)

        actual_width = int(ctrl.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(ctrl.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] opened {actual_width}x{actual_height}")

        reference_frame = warmup_reference(ctrl, args.warmup_frames)
        if label_schedule:
            print(f"[INFO] timed labels: {label_schedule} (-1 means ignored by windows)")
            print("[INFO] controls while recording: q/ESC stop; timed labels override manual 0/1 labels")
        else:
            print("[INFO] controls while recording: q/ESC stop, s/1 slip, n/0 no-slip, space toggle")
        print(f"[INFO] writing to {session_dir}")
        if args.target_fps:
            print(f"[INFO] expected camera interval: {1000.0 / args.target_fps:.1f} ms")

        if args.preview and not args.no_wait_start:
            reference_frame, should_record = wait_for_reference_and_start(ctrl, reference_frame, args)
            if not should_record:
                return
        elif args.preview and args.no_wait_start and not args.auto_reference:
            print("[WARN] --no-wait-start uses warmup frame as reference; use preview start and press b for a clean per-session reference")
        elif not args.preview and not args.auto_reference:
            print("[WARN] no preview: using warmup frame as this session's reference")

        start = time.perf_counter()
        idx = 0

        while True:
            now = time.perf_counter()
            if args.duration_sec and now - start >= args.duration_sec and idx >= args.min_frames:
                break
            if args.max_frames and idx >= args.max_frames and idx >= args.min_frames:
                break

            ok, frame = ctrl.cap_frame()
            if not ok:
                print("[WARN] failed to read frame")
                continue

            now = time.perf_counter()

            rel_time = now - start
            if label_schedule:
                frame_label = scheduled_label_at(rel_time, label_schedule, frame_label)
            filename = f"{idx:06d}.{args.image_ext}"
            rel_path = normalize_rel(os.path.join("rgb", filename))
            captured_frames.append((filename, frame.copy()))
            frame_rows.append({"frame": rel_path, "index": idx, "timestamp_sec": f"{rel_time:.6f}", "label": frame_label})
            idx += 1

            if args.preview:
                text = f"frames={idx} label={frame_label} t={rel_time:.2f}s"
                cv2.putText(frame, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("raw sequence capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) and idx >= args.min_frames:
                    break
                if not label_schedule:
                    if key in (ord("s"), ord("1")):
                        frame_label = 1
                        print("[LABEL] slip from next frames")
                    elif key in (ord("n"), ord("0")):
                        frame_label = 0
                        print("[LABEL] no-slip from next frames")
                    elif key == ord(" "):
                        frame_label = 1 - frame_label
                        print(f"[LABEL] {frame_label} from next frames")
            elif idx % max(1, args.log_every) == 0:
                print(f"[CAP] frames={idx}")

        capture_end = time.perf_counter()
        if idx < args.min_frames:
            raise RuntimeError(f"Only captured {idx} frames; need at least {args.min_frames}")

        print(f"[SAVE] writing {len(captured_frames)} images...", flush=True)
        os.makedirs(rgb_dir, exist_ok=True)
        imwrite_params = image_write_params(args)
        checked_imwrite(os.path.join(session_dir, reference_filename(args)), reference_frame, imwrite_params)
        for filename, saved_frame in captured_frames:
            checked_imwrite(os.path.join(rgb_dir, filename), saved_frame, imwrite_params)

        frames_csv = os.path.join(session_dir, "frames.csv")
        with open(frames_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "index", "timestamp_sec", "label"])
            writer.writeheader()
            writer.writerows(frame_rows)

        elapsed = max(1e-9, capture_end - start)
        actual_fps = idx / elapsed
        timing = frame_timing_stats(frame_rows)
        write_metadata(os.path.join(session_dir, "metadata.csv"), args, actual_width, actual_height, idx, actual_fps)
        print(f"[DONE] captured {idx} frames in {elapsed:.3f}s")
        print(f"[DONE] loop fps: {actual_fps:.2f}")
        print(f"[DONE] timestamp fps: {timing['timestamp_fps']:.2f}")
        print(
            "[DONE] frame interval ms: "
            f"mean={timing['mean_interval_ms']:.1f}, "
            f"min={timing['min_interval_ms']:.1f}, "
            f"max={timing['max_interval_ms']:.1f}"
        )
        print(f"[DONE] frames csv: {frames_csv}")







