"""Optional Telegram text alerts; credentials come from the process environment."""
from datetime import datetime
import json
import os
import threading
import time
from urllib.request import Request, urlopen


def send_message(token, chat_id, text):
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=10) as response:
        if not json.load(response).get("ok"):
            raise RuntimeError("Telegram rejected the message")


class TelegramAlert:
    def __init__(self, token="", chat_id="", cooldown=60):
        self.token, self.chat_id = token, chat_id
        self.cooldown = cooldown
        self.last_sent = float("-inf")
        self.previous = False
        self.worker = None

    @classmethod
    def from_env(cls):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if token and chat_id:
            print("[*] Telegram alerts enabled (minimum interval: 60 seconds).")
        else:
            print("[!] Telegram disabled: set both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return cls(token, chat_id)

    def update(self, active, score=0, model="", source="", now=None):
        rising = active and not self.previous
        self.previous = bool(active)
        now = time.monotonic() if now is None else now
        if not (rising and self.token and self.chat_id):
            return
        if now - self.last_sent < self.cooldown or (self.worker and self.worker.is_alive()):
            return
        self.last_sent = now
        # Never include camera URLs, which can contain embedded passwords.
        camera = str(source) if str(source).isdigit() else "Video / camera"
        text = (
            "⚠️ CẢNH BÁO: Phát hiện dấu hiệu đánh nhau\n"
            f"Nguồn: {camera}\n"
            f"Thời gian: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"Điểm mô hình: {score:.1%} ({model.upper()})\n"
            "Vui lòng kiểm tra camera ngay."
        )
        self.worker = threading.Thread(target=self._send, args=(text,), daemon=True)
        self.worker.start()

    def _send(self, text):
        try:
            send_message(self.token, self.chat_id, text)
            print("[*] Telegram alert sent.")
        except Exception:
            # Network exception strings may contain the bot token in the URL.
            print("[!] Telegram alert failed. Check network, bot token and chat ID.")
