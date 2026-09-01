"""Probe OpenCV camera modes and real read FPS.

This is a diagnostic tool for checking whether backend / FOURCC / FPS /
exposure settings actually take effect on a USB camera.
"""

import argparse
import time
from typing import Iterable, Optional

import cv2


def backend_flag(name: str) -> int:
    name = (name or "default").lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def fourcc_to_text(value: float) -> str:
    code = int(value)
    chars = []
    for i in range(4):
        ch = (code >> (8 * i)) & 0xFF
        chars.append(chr(ch) if 32 <= ch <= 126 else "?")
    return "".join(chars)


def set_prop(cap, prop, value, name: str) -> None:
    ok = cap.set(prop, value)
    got = cap.get(prop)
    if prop == cv2.CAP_PROP_FOURCC:
        print(f"    set {name}={value} ok={ok} reported={fourcc_to_text(got)!r}")
    else:
        print(f"    set {name}={value} ok={ok} reported={got}")


def apply_step(cap, step: str, args) -> None:
    if step == "fourcc" and args.fourcc:
        code = cv2.VideoWriter_fourcc(*args.fourcc[:4].upper())
        set_prop(cap, cv2.CAP_PROP_FOURCC, code, "fourcc")
    elif step == "size":
        set_prop(cap, cv2.CAP_PROP_FRAME_WIDTH, args.width, "width")
        set_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT, args.height, "height")
    elif step == "fps" and args.camera_fps > 0:
        set_prop(cap, cv2.CAP_PROP_FPS, args.camera_fps, "fps")
    elif step == "autoexp" and args.auto_exposure is not None:
        set_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, args.auto_exposure, "auto_exposure")
    elif step == "exposure" and args.exposure is not None:
        set_prop(cap, cv2.CAP_PROP_EXPOSURE, args.exposure, "exposure")
    elif step == "buffersize" and args.buffer_size >= 0:
        set_prop(cap, cv2.CAP_PROP_BUFFERSIZE, args.buffer_size, "buffersize")


def report_state(cap, prefix: str = "    ") -> None:
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
    auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"{prefix}reported state: {w:.0f}x{h:.0f}, fps={fps:.3f}, fourcc={fourcc!r}, auto_exposure={auto_exp}, exposure={exp}")


def measure_read_fps(cap, seconds: float, preview: bool, title: str) -> float:
    for _ in range(5):
        cap.read()
    start = time.perf_counter()
    count = 0
    while True:
        now = time.perf_counter()
        if now - start >= seconds:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        count += 1
        if preview:
            cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("camera fps probe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    elapsed = max(1e-9, time.perf_counter() - start)
    return count / elapsed


def parse_order(value: str) -> Iterable[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def run_trial(order_name: str, order: str, args) -> None:
    print(f"\n[TRIAL] {order_name}: backend={args.backend}, order={order}")
    cap = cv2.VideoCapture(args.cam_index, backend_flag(args.backend))
    if not cap.isOpened():
        print("    open failed")
        return
    try:
        for step in parse_order(order):
            apply_step(cap, step, args)
        report_state(cap)
        measured = measure_read_fps(cap, args.duration_sec, args.preview, order_name)
        report_state(cap, prefix="    after read: ")
        print(f"    measured read fps={measured:.2f}")
    finally:
        cap.release()
        if args.preview:
            cv2.destroyWindow("camera fps probe")


def parse_args():
    p = argparse.ArgumentParser(description="Probe camera FOURCC/FPS/exposure behavior")
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--backend", choices=["default", "dshow", "msmf"], default="dshow")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-fps", type=float, default=30.0)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--auto-exposure", type=float, default=None)
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--buffer-size", type=int, default=-1, help="-1 leaves unchanged")
    p.add_argument("--duration-sec", type=float, default=5.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument(
        "--orders",
        default="fourcc,size,fps,autoexp,exposure;size,fourcc,fps,autoexp,exposure;fourcc,fps,size,autoexp,exposure;size,fps,fourcc,autoexp,exposure",
        help="Semicolon-separated setup orders; each order is comma-separated steps",
    )
    return p.parse_args()


def main():
    args = parse_args()
    for i, order in enumerate([item.strip() for item in args.orders.split(";") if item.strip()], start=1):
        run_trial(f"order_{i}", order, args)


if __name__ == "__main__":
    main()