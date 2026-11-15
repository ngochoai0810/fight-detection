import numpy as np

import alerts


def test_confirmation_rejects_spikes_and_stops_immediately():
    gate = alerts.AlertConfirmation(seconds=1)
    assert not gate.update(True, now=0)
    assert not gate.update(True, now=0.9)
    assert not gate.update(False, now=0.95)
    assert not gate.update(True, now=1)
    assert gate.update(True, now=2)
    assert not gate.update(False, now=2.1)
    assert not gate.update(True, now=2.2)


def test_zero_hold_preserves_immediate_alerts():
    gate = alerts.AlertConfirmation(seconds=0)
    assert gate.update(True, now=0)
    assert not gate.update(False, now=0.1)


def test_voice_cooldown_no_overlap_and_disabled_normal(monkeypatch):
    calls = []
    clock = [100.0]
    class Process:
        result = None
        def poll(self):
            return self.result
    def spawn(*args, **kwargs):
        p = Process()
        calls.append(p)
        return p
    monkeypatch.setattr(alerts.os, "name", "nt")
    monkeypatch.setattr(alerts.subprocess, "Popen", spawn)
    monkeypatch.setattr(alerts.time, "monotonic", lambda: clock[0])
    voice = alerts.VoiceAlert(cooldown=10)
    voice.update(False)
    assert not calls
    voice.update(True)
    clock[0] = 120
    voice.update(True)
    assert len(calls) == 1  # Speech still playing: never queue another message.
    calls[0].result = 0
    clock[0] = 105
    voice.update(True)
    assert len(calls) == 1
    clock[0] = 111
    voice.update(True)
    assert len(calls) == 2


def test_overlay_counts_only_new_alerts_and_keeps_resolution():
    overlay = alerts.AlertOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for active in [False, True, True, False, True]:
        result = overlay.draw(frame.copy(), active, True, 0.6, 0.5, "m1", 20)
        assert result.shape == frame.shape
    assert overlay.events == 2


def test_overlay_renders_detection_border_lines():
    overlay = alerts.AlertOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = {
        "zone": [0.2, 0.25, 0.75, 0.85],
        "zone_kind": "interaction",
        "persons": [
            {"box": [0.22, 0.28, 0.48, 0.82], "conf": 0.9, "involved": True},
            {"box": [0.46, 0.30, 0.72, 0.83], "conf": 0.87, "involved": True},
        ],
    }
    out = overlay.draw(frame.copy(), True, True, 0.91, 0.5, "m3", 25.0, detections=detections)
    assert out.shape == frame.shape
    # Top-left corner of the conflict zone (x=128, y=120) should have non-zero border pixels
    assert out[120, 128].sum() > 0


def test_voice_stops_on_normal_and_can_alert_again(monkeypatch):
    monkeypatch.setattr(alerts.os, "name", "nt")
    class Process:
        stopped = False
        def poll(self):
            return None
        def terminate(self):
            self.stopped = True
        def wait(self, timeout):
            return 0
    voice = alerts.VoiceAlert()
    process = Process()
    voice.process = process
    voice.update(False)
    assert process.stopped
    assert voice.process is None
    assert voice.enabled
    # Repeated normal frames and application shutdown are safe.
    voice.update(False)
    voice.close()
