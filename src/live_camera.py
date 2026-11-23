"""A single shared, stoppable camera pipeline with a bounded latest-frame buffer."""
from argparse import Namespace
from pathlib import Path
import threading
import time

import cv2


class LiveCamera:
    def __init__(self, root, runner=None):
        self.root = Path(root).resolve()
        self.project = Path(__file__).resolve().parent.parent
        self.runner = runner
        self.condition = threading.Condition()
        self.thread = None
        self.stop_event = threading.Event()
        self.state = {"status": "stopped"}
        self.frame = None
        self.sequence = 0
        self.last_publish = 0

    def snapshot(self):
        with self.condition:
            return dict(self.state)

    def is_active(self):
        with self.condition:
            return self.thread is not None and self.thread.is_alive()

    def start(self, model="m1", source=0, threshold=None, window_seconds=5.0):
        if model not in ("m1", "m2", "m3"):
            raise ValueError("Mô hình không hợp lệ")
        if type(source) is not int or not 0 <= source <= 9:
            raise ValueError("Camera phải là chỉ số từ 0 đến 9")
        if threshold is not None and (type(threshold) not in (int, float) or not 0 <= threshold <= 1):
            raise ValueError("Ngưỡng phải nằm trong [0, 1]")
        if type(window_seconds) not in (int, float) or not 0.5 <= window_seconds <= 30:
            raise ValueError("Cửa sổ M3 phải nằm trong [0.5, 30] giây")
        weights = self.project / f"results/{model}/best.pt"
        yolo = self.project / "yolo11n.pt"
        if not weights.is_file():
            raise ValueError(f"Thiếu results/{model}/best.pt")
        if model == "m3" and not yolo.is_file():
            raise ValueError("M3 cần trọng số yolo11n.pt")
        with self.condition:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("Camera đang chạy hoặc đang dừng")
            self.stop_event = threading.Event()
            self.frame = None
            self.last_publish = 0
            self.state = {"status": "starting", "source": source, "model": model, "window_seconds": window_seconds}
            args = Namespace(
                source=str(source), model=model, weights=str(weights), yolo_weights=str(yolo),
                image_size=224, threshold=threshold, seq_len=16, stride=2,
                window_seconds=window_seconds,
                save=None, output_dir=str(self.root), device=None, metrics=None,
                camera_backend="auto", camera_width=640, camera_height=480, camera_mjpg=True,
                no_voice=True, voice_cooldown=10, alert_hold=1.0, pre_seconds=2.5,
                no_boxes=False, headless=True,
            )
            self.thread = threading.Thread(target=self._run, args=(args,), daemon=True)
            self.thread.start()
            return dict(self.state)

    def _run(self, args):
        try:
            runner = self.runner
            if runner is None:
                from inference import run_inference
                runner = run_inference
            runner(args, on_frame=self._publish, stop_event=self.stop_event)
            with self.condition:
                self.state["status"] = "stopped"
        except Exception as exc:
            with self.condition:
                self.state.update(status="failed", error=str(exc))
        finally:
            with self.condition:
                self.frame = None
                self.state.update(active=False, pending=False, ready=False)
                self.condition.notify_all()

    def _publish(self, frame, telemetry):
        now = time.monotonic()
        if now - self.last_publish < .1 or self.stop_event.is_set():
            return
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self.condition:
            if self.stop_event.is_set():
                return
            self.frame = encoded.tobytes()
            self.sequence += 1
            self.last_publish = now
            self.state.update(telemetry, status="running")
            self.condition.notify_all()

    def next_frame(self, previous, timeout=1):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous or self.state["status"] in ("stopped", "failed"),
                timeout=timeout,
            )
            return self.sequence, self.frame, self.state["status"]

    def stop(self, wait=False):
        with self.condition:
            self.stop_event.set()
            thread = self.thread
            if thread is not None and thread.is_alive():
                self.state["status"] = "stopping"
            self.condition.notify_all()
        if wait and thread is not None:
            thread.join(timeout=8)
        return self.snapshot()
