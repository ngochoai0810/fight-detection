import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from dashboard import handler
from live_camera import LiveCamera


def fixture_camera(tmp_path, runner):
    camera = LiveCamera(tmp_path, runner=runner)
    camera.project = tmp_path
    weights = tmp_path / "results/m1/best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    return camera


def frame_runner(args, on_frame, stop_event):
    on_frame(np.zeros((48, 64, 3), dtype=np.uint8), {
        "score": .8, "threshold": .5, "active": True, "pending": False,
        "ready": True, "fps": 12.5, "model": args.model,
    })
    stop_event.wait(5)


def wait_status(camera, expected):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if camera.snapshot()["status"] == expected:
            return camera.snapshot()
        time.sleep(.01)
    raise AssertionError(camera.snapshot())


def test_latest_frame_camera_stop_and_restart(tmp_path):
    camera = fixture_camera(tmp_path, frame_runner)
    camera.start("m1", 0, .5)
    try:
        assert wait_status(camera, "running")["active"]
        sequence, frame, status = camera.next_frame(-1)
        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (48, 64, 3)
        with pytest.raises(RuntimeError):
            camera.start("m1")
        camera.stop(wait=True)
        assert camera.snapshot()["status"] == "stopped"
        assert camera.frame is None
        camera.start("m1", 1)
        wait_status(camera, "running")
        assert camera.next_frame(sequence)[0] > sequence
    finally:
        camera.stop(wait=True)


def test_camera_failure_and_validation(tmp_path):
    def failing_runner(*args, **kwargs):
        raise RuntimeError("Camera unplugged")
    camera = fixture_camera(tmp_path, failing_runner)
    for model, source, threshold in [("bad", 0, None), ("m1", -1, None), ("m1", "0", None), ("m1", 0, 2), ("m1", 0, float("nan"))]:
        with pytest.raises(ValueError):
            camera.start(model, source, threshold)
    camera.start("m1")
    assert wait_status(camera, "failed")["error"] == "Camera unplugged"
    camera.stop(wait=True)
    assert camera.frame is None


def test_camera_http_stream_and_controls(tmp_path, monkeypatch):
    import dashboard
    camera = fixture_camera(tmp_path, frame_runner)
    monkeypatch.setattr(dashboard, "LiveCamera", lambda root: camera)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body=None, headers=None):
        return urlopen(Request(base + path, data=json.dumps(body or {}).encode(), headers=headers or {}), timeout=3)

    try:
        assert post("/api/camera/start", {"model": "m1", "source": 0}).status == 202
        wait_status(camera, "running")
        assert json.load(urlopen(base + "/api/camera"))["score"] == .8
        with urlopen(base + "/api/camera/stream", timeout=3) as response:
            assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            assert b"Content-Type: image/jpeg" in response.read(150)
        with pytest.raises(HTTPError) as duplicate:
            post("/api/camera/start", {"model": "m1"})
        assert duplicate.value.code == 409
        with pytest.raises(HTTPError) as upload:
            urlopen(Request(base + "/api/detection?model=m1", data=b"video"), timeout=3)
        assert upload.value.code == 409
        with pytest.raises(HTTPError) as foreign:
            post("/api/camera/stop", headers={"Origin": "https://example.com"})
        assert foreign.value.code == 403
        assert post("/api/camera/stop").status == 200
        wait_status(camera, "stopped")
    finally:
        camera.stop(wait=True)
        server.shutdown()
        server.server_close()
        thread.join()
