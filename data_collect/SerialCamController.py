import os
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import serial
import serial.tools.list_ports


def camera_backend_flag(name: str) -> int:
    name = (name or "default").lower()
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


class SerialCamController:
    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        cam_index: int = 0,
        save_dir: str = "captures",
        serial_timeout: float = 0.2,
        auto_select_if_single: bool = True,
        width: int = 640,
        height: int = 480,
        backend: str = "default",
        no_serial: bool = False,
        camera_fps: float = 0.0,
        fourcc: str = "",
        auto_exposure: Optional[float] = None,
        exposure: Optional[float] = None,
        strict_backend: bool = False,
    ):
        self.baud = baud
        self.cam_index = cam_index
        self.save_dir = save_dir
        self.serial_timeout = serial_timeout
        self.port = port
        self.no_serial = no_serial
        self.ser = None
        self.cap = None
        self._preview_thread = None
        self._preview_stop_evt = None

        if not self.no_serial:
            if self.port is None:
                ports = self.list_ports()
                if not ports:
                    raise RuntimeError("Cannot find serial port")
                if auto_select_if_single and len(ports) == 1:
                    self.port = ports[0]
                    print(f"[INFO] auto-selected serial port: {self.port}")
                else:
                    raise ValueError("Serial port is required when multiple/no ports are available")
            self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.serial_timeout)
            time.sleep(0.5)
            print(f"[OK] serial {self.port} @ {self.baud} bps")
        else:
            print("[LIGHT] serial disabled; assuming light is already fixed")

        print(f"[CAM] opening index={self.cam_index} backend={backend}...", flush=True)
        self.cap = cv2.VideoCapture(self.cam_index, camera_backend_flag(backend))
        if not self.cap.isOpened() and backend != "default":
            if strict_backend:
                self.close_serial()
                raise RuntimeError(f"Cannot open camera index={self.cam_index} with backend={backend}")
            print("[CAM] requested backend failed; retrying default backend...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.close_serial()
            raise RuntimeError(f"Cannot open camera index={self.cam_index}")

        self.configure_stream(width, height, camera_fps, fourcc, auto_exposure, exposure)
        for _ in range(2):
            self.cap.read()
        print("[OK] camera is ready")

    @staticmethod
    def list_ports():
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            print(f"  - {port.device} ({port.description})")
        return [port.device for port in ports]

    def _readline_nonblock(self) -> str:
        if self.ser is None:
            return ""
        try:
            return self.ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def configure_stream(
        self,
        width: int,
        height: int,
        camera_fps: float = 0.0,
        fourcc: str = "",
        auto_exposure: Optional[float] = None,
        exposure: Optional[float] = None,
    ) -> None:
        # Some DirectShow UVC drivers only keep MJPG if the stream is configured
        # as size -> FPS -> FOURCC. Other orders can report success and still
        # fall back to YUY2 after the stream starts.
        ok = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        print(f"[CAM] set width={width} ok={ok}")
        ok = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"[CAM] set height={height} ok={ok}")
        if camera_fps and camera_fps > 0:
            ok = self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
            print(f"[CAM] set fps={camera_fps} ok={ok}")
        if fourcc:
            code = cv2.VideoWriter_fourcc(*fourcc[:4].upper())
            ok = self.cap.set(cv2.CAP_PROP_FOURCC, code)
            print(f"[CAM] set fourcc={fourcc[:4].upper()} ok={ok}")
        if auto_exposure is not None:
            ok = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
            print(f"[CAM] set auto_exposure={auto_exposure} ok={ok}")
        if exposure is not None:
            ok = self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"[CAM] set exposure={exposure} ok={ok}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
        actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        actual_auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"[CAM] resolution {w}x{h}")
        print(f"[CAM] reported fps={actual_fps:.3f} fourcc={fourcc_text!r} auto_exposure={actual_auto_exposure} exposure={actual_exposure}")

    def set_resolution(self, width: int, height: int) -> None:
        self.configure_stream(width, height)

    def light(self, cmd: int) -> None:
        if self.ser is None:
            return
        if (not (0 <= cmd <= 14)) and (not (21 <= cmd <= 28)):
            raise ValueError("cmd must be between 0-14 or 21-28")
        self.ser.reset_input_buffer()
        self.ser.write(f"{cmd}\n".encode("utf-8"))
        self.ser.flush()
        time.sleep(0.03)
        resp = self._readline_nonblock()
        if resp:
            print(f"[Arduino] {resp}")
        else:
            print(f"[Sent] cmd={cmd}")

    def cap_frame(self) -> Tuple[bool, Optional[object]]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def takephoto(self, tag: Optional[object] = None, times=None) -> str:
        ok, frame = self.cap_frame()
        if not ok:
            raise RuntimeError("Camera is not ready")
        save_dir = self.save_dir if times is None else os.path.join(self.save_dir, str(times))
        os.makedirs(save_dir, exist_ok=True)
        tag_str = "" if tag is None else f"{tag}"
        if tag_str.isdigit():
            filename = os.path.join(save_dir, f"photometric_sample_raw_00{int(tag_str):02d}.png")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(save_dir, f"{tag_str or 'capture'}_{ts}.png")
        if not cv2.imwrite(filename, frame):
            raise RuntimeError(f"Failed to save image: {filename}")
        print(f"[OK] saved {filename}")
        return filename

    def start_preview_async(self, window_name: str = "Preview", mirror: bool = False, fps: int = 30):
        self.stop_preview()
        self._preview_stop_evt = threading.Event()
        self._preview_window_name = window_name
        self._preview_mirror = mirror
        self._preview_interval = 1.0 / max(1, fps)

        def _loop():
            cv2.namedWindow(self._preview_window_name, cv2.WINDOW_NORMAL)
            while not self._preview_stop_evt.is_set():
                ok, frame = self.cap_frame()
                if not ok:
                    time.sleep(0.02)
                    continue
                if self._preview_mirror:
                    frame = cv2.flip(frame, 1)
                cv2.imshow(self._preview_window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._preview_stop_evt.set()
                    break
                time.sleep(self._preview_interval)
            cv2.destroyWindow(self._preview_window_name)

        self._preview_thread = threading.Thread(target=_loop, daemon=True)
        self._preview_thread.start()

    def stop_preview(self):
        if self._preview_thread and self._preview_thread.is_alive():
            if self._preview_stop_evt:
                self._preview_stop_evt.set()
            self._preview_thread.join(timeout=1.0)
        try:
            if hasattr(self, "_preview_window_name"):
                cv2.destroyWindow(self._preview_window_name)
        except cv2.error:
            pass

    def close_serial(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def close(self) -> None:
        self.stop_preview()
        if self.cap is not None:
            self.cap.release()
        self.close_serial()
        cv2.destroyAllWindows()
        print("[OK] released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()




