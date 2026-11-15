import hashlib
import json
from html.parser import HTMLParser
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from dashboard import build_events_csv, handler, load_dashboard_state, render_dashboard_html
from monitoring import Monitor, connect, review_event


def test_dashboard_video_upload_runs_headless_and_serves_result(tmp_path, monkeypatch):
    import dashboard
    from types import SimpleNamespace

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        source = command[command.index("--source") + 1]
        assert open(source, "rb").read() == b"test-video"
        output = command[command.index("--save") + 1]
        with open(output, "wb") as video:
            video.write(b"result-video")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/api/detection?model=m1", data=b"test-video")
        assert urlopen(request).status == 202
        for _ in range(100):
            job = json.load(urlopen(base + "/api/detection"))
            if job["status"] != "running":
                break
            time.sleep(.01)
        assert job["status"] == "completed"
        assert urlopen(base + job["video"]).read() == b"result-video"
        assert "--headless" in commands[0] and "--no-voice" in commands[0]
        assert not list(tmp_path.glob("*.upload"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_evidence_roundtrip_and_review(tmp_path):
    monitor = Monitor(tmp_path, {"model": "m3"}, post_seconds=0)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    monitor.record(frame, True, .8, 20, 30)
    monitor.record(frame, True, .9, 40, 50)
    monitor.record(frame, False, .2, None, 10)
    monitor.close()
    db = connect(tmp_path)
    event = db.execute("SELECT * FROM events").fetchone()
    assert event[4] > 0 and event[5] == .9
    cap = cv2.VideoCapture(str(tmp_path / event[3]))
    assert cap.read()[0]
    cap.release()
    summary = json.loads(db.execute("SELECT summary FROM sessions").fetchone()[0])
    assert summary["predictions"] == 2
    assert summary["inference_mean_ms"] == 30
    assert summary["pipeline_fps"] > 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert "m3" in urlopen(base).read().decode()
        for name in ("dashboard.css", "detection.js", "camera.js", "evidence.js"):
            with urlopen(base + "/assets/" + name) as asset:
                assert asset.status == 200
                expected_type = "text/css" if name.endswith(".css") else "text/javascript"
                assert asset.headers.get_content_type() == expected_type
                assert len(asset.read()) > 100
        data = urlencode({"id": event[0], "review": "false_alert"}).encode()
        assert "100.0%" in urlopen(Request(base + "/review", data=data)).read().decode()
        assert db.execute("SELECT review FROM events").fetchone()[0] == "false_alert"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        db.close()


def test_shutdown_finalizes_active_clip_and_empty_session(tmp_path):
    monitor = Monitor(tmp_path, {})
    monitor.record(np.zeros((64, 64, 3), dtype=np.uint8), True, .7, None, 2)
    monitor.close()
    db = connect(tmp_path)
    assert db.execute("SELECT duration FROM events").fetchone()[0] > 0
    db.close()
    empty = Monitor(tmp_path, {})
    empty.close()


def test_preroll_bbox_timeline_hash_and_audit_logs(tmp_path):
    cfg = {"source": "cam_lobby", "model": "m3", "threshold": 0.6, "source_fps": 10.0}
    monitor = Monitor(tmp_path, cfg, post_seconds=0, pre_seconds=2.0)

    # 2 pre-roll frames before alert confirmation
    f1 = np.full((80, 120, 3), 30, dtype=np.uint8)
    f2 = np.full((80, 120, 3), 60, dtype=np.uint8)
    f3 = np.full((80, 120, 3), 180, dtype=np.uint8)
    det_peak = {
        "zone": [0.15, 0.2, 0.75, 0.85],
        "zone_kind": "interaction",
        "persons": [
            {"box": [0.18, 0.22, 0.45, 0.82], "conf": 0.91, "involved": True},
            {"box": [0.42, 0.25, 0.72, 0.84], "conf": 0.88, "involved": True},
        ],
    }

    monitor.record(f1, False, 0.25, 12.0, 15.0)
    monitor.record(f2, False, 0.68, 14.0, 16.0, detections=det_peak, pending=True)
    monitor.record(f3, True, 0.93, 15.0, 18.0, detections=det_peak, pending=False)
    monitor.record(f1, False, 0.20, 11.0, 14.0)
    monitor.close()

    state = load_dashboard_state(tmp_path)
    assert state["stats"]["total_events"] == 1
    ev = state["events"][0]
    assert ev["frames"] == 4  # 2 pre-roll frames + 1 alert frame + 1 closing post-frame
    assert ev["pre_roll"] > 0
    assert ev["severity"] == "critical"
    assert ev["thumbnail"] is not None and (tmp_path / ev["thumbnail"]).is_file()
    assert ev["bbox"]["zone"] == [0.15, 0.2, 0.75, 0.85]
    assert len(ev["timeline"]) >= 2

    # Verify SHA-256 matches actual MP4 bytes on disk
    clip_bytes = (tmp_path / ev["clip"]).read_bytes()
    assert ev["file_size"] == len(clip_bytes)
    assert ev["sha256"] == hashlib.sha256(clip_bytes).hexdigest()

    # Review with operator notes & check audit trail + CSV/HTML export
    assert review_event(tmp_path, ev["id"], "true_alert", notes="Xo xat khu vuc sakh A")
    state_after = load_dashboard_state(tmp_path)
    assert state_after["events"][0]["review"] == "true_alert"
    assert state_after["events"][0]["notes"] == "Xo xat khu vuc sakh A"
    assert state_after["stats"]["precision_ratio"] == "100.0%"

    codes = {log["code"] for log in state_after["logs"]}
    assert {"SESSION_START", "ALERT_PENDING", "ALERT_CONFIRMED", "CLIP_SAVED", "ALERT_CLEARED", "SESSION_END", "EVENT_REVIEWED"} <= codes
    assert (tmp_path / "alert_logs.jsonl").is_file()

    csv_text = build_events_csv(state_after).decode("utf-8-sig")
    assert "cam_lobby" in csv_text and "Xo xat khu vuc sakh A" in csv_text

    html_page = render_dashboard_html(state_after).decode("utf-8")
    assert "DASHBOARD" in html_page and ev["id"] in html_page


def test_external_dashboard_template_preserves_json_with_html_notes(tmp_path):
    state = load_dashboard_state(tmp_path)
    state["notes"] = '</script><script>alert("test")</script> & Tiếng Việt'
    page = render_dashboard_html(state).decode("utf-8")

    class InitialStateParser(HTMLParser):
        capture = False
        data = ""
        def handle_starttag(self, tag, attrs):
            self.capture = tag == "script" and dict(attrs).get("id") == "initialState"
        def handle_data(self, data):
            if self.capture:
                self.data += data
        def handle_endtag(self, tag):
            if tag == "script":
                self.capture = False

    parser = InitialStateParser()
    parser.feed(page)
    assert json.loads(parser.data) == state
    assert '<script>alert("test")</script>' not in page
