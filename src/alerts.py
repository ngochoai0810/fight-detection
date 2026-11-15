"""Vietnamese camera tactical HUD overlay, detection border lines, and Windows speech alerts."""
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


def _draw_corner_box(frame, x1, y1, x2, y2, color_bgr, thickness=2, corner_ratio=0.18, fill_alpha=0.0):
    """Ve border line kieu camera giam sat chuyen dung voi 4 goc L-bracket."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, min(w - 1, int(x1))), max(0, min(h - 1, int(y1)))
    x2, y2 = max(0, min(w - 1, int(x2))), max(0, min(h - 1, int(y2)))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return
    if fill_alpha > 0:
        roi = frame[y1:y2, x1:x2]
        tint = np.empty_like(roi)
        tint[:] = color_bgr
        cv2.addWeighted(tint, fill_alpha, roi, 1.0 - fill_alpha, 0, dst=roi)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, max(1, thickness - 1), cv2.LINE_AA)
    bw, bh = x2 - x1, y2 - y1
    cl = max(8, min(32, int(min(bw, bh) * corner_ratio)))
    ct = thickness + 2
    for cx, cy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color_bgr, ct, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * cl), color_bgr, ct, cv2.LINE_AA)


class AlertOverlay:
    def __init__(self):
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
        try:
            self.font = ImageFont.truetype(str(font_path), 17)
            self.font_bold = ImageFont.truetype(str(font_path), 19)
            self.font_tag = ImageFont.truetype(str(font_path), 14)
        except OSError:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 17)
            self.font_bold = ImageFont.truetype("DejaVuSans.ttf", 19)
            self.font_tag = ImageFont.truetype("DejaVuSans.ttf", 14)
        self.alert_since = None
        self.events = 0
        self.last_active = False

    def _draw_pill_label(self, frame, x, y, text, bg_rgb, text_rgb=(255, 255, 255)):
        h, w = frame.shape[:2]
        dummy = Image.new("RGB", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=self.font_tag)
        tw, th = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 8
        if tw >= w or th >= h:
            return
        px = max(2, min(w - tw - 2, int(x)))
        py = int(y) - th - 2
        if py < 2:
            py = min(h - th - 2, int(y) + 4)
        chip = Image.new("RGB", (tw, th), bg_rgb)
        ImageDraw.Draw(chip).text((8, 3), text, font=self.font_tag, fill=text_rgb)
        chip_bgr = cv2.cvtColor(np.asarray(chip), cv2.COLOR_RGB2BGR)
        roi = frame[py : py + th, px : px + tw]
        cv2.addWeighted(chip_bgr, 0.88, roi, 0.12, 0, dst=roi)

    def _draw_detections(self, frame, detections, active, pending, score):
        if not detections:
            return
        h, w = frame.shape[:2]
        persons = detections.get("persons") or []
        zone = detections.get("zone")

        for idx, p in enumerate(persons, start=1):
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            involved = bool(p.get("involved"))
            if active and involved:
                p_color = (60, 75, 255)  # BGR red-coral
                _draw_corner_box(frame, x1, y1, x2, y2, p_color, thickness=2, fill_alpha=0.05)
            elif pending and involved:
                p_color = (20, 185, 255)  # BGR amber
                _draw_corner_box(frame, x1, y1, x2, y2, p_color, thickness=2)
            else:
                p_color = (235, 190, 56)  # BGR cyan-slate
                _draw_corner_box(frame, x1, y1, x2, y2, p_color, thickness=1)

        if zone and len(zone) >= 4 and (active or pending or score >= 0.25):
            zx1, zy1, zx2, zy2 = zone[0] * w, zone[1] * h, zone[2] * w, zone[3] * h
            if active:
                z_bgr = (48, 59, 255)
                label = f"KHOANH VÙNG ĐÁNH NHAU • {score * 100:.1f}%"
                bg_rgb = (210, 38, 38)
                fill_alpha = 0.10
                thick = 3
            elif pending:
                z_bgr = (0, 175, 255)
                label = f"NGHI VẤN XÔ XÁT • {score * 100:.1f}%"
                bg_rgb = (205, 130, 10)
                fill_alpha = 0.06
                thick = 2
            else:
                z_bgr = (220, 180, 45)
                label = f"VÙNG QUAN SÁT • {score * 100:.1f}%"
                bg_rgb = (25, 95, 135)
                fill_alpha = 0.0
                thick = 1
            _draw_corner_box(frame, zx1, zy1, zx2, zy2, z_bgr, thickness=thick, corner_ratio=0.22, fill_alpha=fill_alpha)
            self._draw_pill_label(frame, zx1, zy1, label, bg_rgb)

    def draw(self, frame, active, ready, score, threshold, model, fps, pending=False, detections=None):
        now = time.monotonic()
        if active and not self.last_active:
            self.alert_since = now
            self.events += 1
        if not active:
            self.alert_since = None
        self.last_active = active

        # Draw detection border lines first so top-left HUD stays readable above boxes
        self._draw_detections(frame, detections, active, pending, score)

        color = (255, 82, 82) if active else (86, 232, 142)
        status = "CẢNH BÁO: PHÁT HIỆN ĐÁNH NHAU!" if active else "GIÁM SÁT: AN TOÀN / BÌNH THƯỜNG"
        if not ready:
            status = "ĐANG TÍCH LŨY KHUNG HÌNH..."
            color = (148, 195, 255)
        elif pending:
            status = "ĐANG XÁC NHẬN HÀNH VI XÔ XÁT..."
            color = (255, 205, 65)

        lines = [
            status,
            f"Điểm Fight: {score * 100:.1f}%  |  Ngưỡng: {threshold * 100:.1f}%",
            f"Mô hình: {model.upper()}  |  FPS: {fps:.1f}  |  Đợt CB: {self.events}",
        ]
        if active:
            lines.append(f"● REC BẰNG CHỨNG: {now - self.alert_since:.1f}s  (Nhấn Q để thoát)")

        fh, fw = frame.shape[:2]
        panel_width = min(fw - 16 if fw > 40 else fw, 470)
        panel_height = min(fh - 16 if fh > 40 else fh, 26 + 24 * len(lines))
        if panel_width > 10 and panel_height > 10:
            ox, oy = (8, 8) if fw > 40 and fh > 40 else (0, 0)
            panel = Image.new("RGB", (panel_width, panel_height), (12, 17, 28))
            draw = ImageDraw.Draw(panel)
            # Status accent bar on left edge
            draw.rectangle((0, 0, 4, panel_height - 1), fill=color)
            for i, line in enumerate(lines):
                draw.text(
                    (14, 7 + i * 23),
                    line,
                    font=self.font_bold if i == 0 else self.font,
                    fill=color if i == 0 else (228, 235, 245),
                )
            # Confidence progress bar at bottom of HUD
            bar_y = panel_height - 10
            if bar_y > 16 and panel_width > 40:
                bar_x1, bar_x2 = 14, panel_width - 14
                draw.rectangle((bar_x1, bar_y, bar_x2, bar_y + 4), fill=(38, 48, 66))
                fill_x = bar_x1 + int((bar_x2 - bar_x1) * max(0.0, min(1.0, score)))
                if fill_x > bar_x1:
                    draw.rectangle((bar_x1, bar_y, fill_x, bar_y + 4), fill=color)
                th_x = bar_x1 + int((bar_x2 - bar_x1) * max(0.0, min(1.0, threshold)))
                draw.rectangle((th_x - 1, bar_y - 2, th_x + 1, bar_y + 6), fill=(255, 255, 255))

            panel_bgr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
            roi = frame[oy : oy + panel_height, ox : ox + panel_width]
            cv2.addWeighted(panel_bgr, 0.76, roi, 0.24, 0, dst=roi)

        if active:
            cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (36, 48, 245), 5)
        elif pending:
            cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 180, 255), 2)
        return frame
