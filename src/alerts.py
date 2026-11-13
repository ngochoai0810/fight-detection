"""Vietnamese camera overlay and non-blocking Windows speech alerts."""
import base64
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class AlertConfirmation:
    """Require consecutive above-threshold predictions before raising an alert."""
    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.since = None

    def update(self, above_threshold, now=None):
        now = time.monotonic() if now is None else now
        if not above_threshold:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.seconds


class VoiceAlert:
    def __init__(self, enabled=True, cooldown=10):
        self.enabled = enabled and os.name == "nt"
        self.cooldown = cooldown
        self.last_spoken = float("-inf")
        self.process = None
        if enabled and not self.enabled:
            print("[!] Voice alerts currently require Windows.")

    def update(self, active):
        if not active:
            self.close()
            return
        if not self.enabled:
            return
        if self.process is not None:
            result = self.process.poll()
            if result is None:
                return
            self.process = None
            if result != 0:
                print("[!] Khong phat duoc giong noi Windows. Kiem tra loa va Speech voices.")
                self.enabled = False
                return
        now = time.monotonic()
        if not active or now - self.last_spoken < self.cooldown:
            return
        # Fixed script only; no shell interpolation of user-provided text.
        script = """$ErrorActionPreference = 'Stop'
$s = New-Object -ComObject SAPI.SpVoice
$v = $s.GetVoices() | Where-Object { $_.GetAttribute('Language') -match '(^|;)42a(;|$)' } | Select-Object -First 1
if ($v) {
    $s.Voice = $v
    [void]$s.Speak('Cảnh báo! Phát hiện dấu hiệu đánh nhau. Vui lòng kiểm tra camera.')
} else {
    [void]$s.Speak('Warning! Possible fighting detected. Please check the camera.')
}
"""
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            self.process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.last_spoken = now
        except OSError as exc:
            print(f"[!] Voice alert unavailable: {exc}")
            self.enabled = False

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


class AlertOverlay:
    def __init__(self):
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
        try:
            self.font = ImageFont.truetype(str(font_path), 20)
        except OSError:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.alert_since = None
        self.events = 0
        self.last_active = False

    def draw(self, frame, active, ready, score, threshold, model, fps, pending=False):
        now = time.monotonic()
        if active and not self.last_active:
            self.alert_since = now
            self.events += 1
        if not active:
            self.alert_since = None
        self.last_active = active
        color = (255, 70, 70) if active else (90, 240, 130)
        status = "CẢNH BÁO: CÓ DẤU HIỆU ĐÁNH NHAU!" if active else "TRẠNG THÁI: CHƯA PHÁT HIỆN ĐÁNH NHAU"
        if not ready:
            status = "ĐANG CHỜ ĐỦ DỮ LIỆU ĐỂ DỰ ĐOÁN..."
        elif pending:
            status = "ĐANG XÁC NHẬN DẤU HIỆU ĐÁNH NHAU..."
            color = (255, 210, 70)
        lines = [status,
                 f"Điểm Fight: {score * 100:.3f}% | Ngưỡng: {threshold * 100:.3f}%",
                 f"Mô hình: {model.upper()} | FPS: {fps:.1f}"]
        if active:
            lines += [f"Cảnh báo liên tục: {now - self.alert_since:.1f} giây",
                      "Vui lòng kiểm tra tình huống trên camera."]
        lines.append(f"Số đợt cảnh báo: {self.events} | Nhấn Q để thoát")
        # Render only the information panel, retaining the original camera resolution.
        panel_width = min(frame.shape[1], 650)
        panel_height = min(frame.shape[0], 18 + 29 * len(lines))
        panel = Image.new("RGB", (panel_width, panel_height), (15, 15, 20))
        draw = ImageDraw.Draw(panel)
        for i, line in enumerate(lines):
            draw.text((12, 8 + i * 29), line, font=self.font, fill=color if i == 0 else (240, 240, 240))
        frame[:panel_height, :panel_width] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        if active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
        return frame
