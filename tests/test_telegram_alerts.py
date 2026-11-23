import telegram_alerts


def test_alert_edges_cooldown_and_redaction(monkeypatch):
    sent = []
    class Worker:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)
        def is_alive(self):
            return False
    monkeypatch.setattr(telegram_alerts.threading, "Thread", Worker)
    monkeypatch.setattr(telegram_alerts, "send_message", lambda *args: sent.append(args))
    alert = telegram_alerts.TelegramAlert("secret", "123")
    alert.update(True, .9, "m1", "rtsp://user:password@host", now=0)
    alert.update(True, now=70)
    alert.update(False, now=71)
    alert.update(True, now=72)
    alert.update(False, now=73)
    alert.update(True, now=74)
    assert len(sent) == 2
    assert "90.0%" in sent[0][2]
    assert "password" not in sent[0][2]


def test_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    alert = telegram_alerts.TelegramAlert.from_env()
    alert.update(True)
    assert alert.worker is None


def test_failure_does_not_print_token(monkeypatch, capsys):
    def fail(*args):
        raise OSError("https://api.telegram.org/botSECRET/sendMessage")
    monkeypatch.setattr(telegram_alerts, "send_message", fail)
    telegram_alerts.TelegramAlert("SECRET", "123")._send("test")
    output = capsys.readouterr().out
    assert "failed" in output
    assert "SECRET" not in output


def test_config_status_does_not_expose_credentials(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "PRIVATE_ID")
    telegram_alerts.TelegramAlert.from_env()
    output = capsys.readouterr().out
    assert "enabled" in output
    assert "SECRET" not in output
    assert "PRIVATE_ID" not in output
