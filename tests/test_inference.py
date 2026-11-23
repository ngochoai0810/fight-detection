from argparse import Namespace
import json

import pytest
import torch

from inference import load_model, resolve_threshold
from models import CNNLSTM, FrameCNN


@pytest.mark.parametrize("kind,key,arch", [("m1", "state_dict", "mobilenet_v2"), ("m1", "model", "resnet18"), ("m3", "state_dict", "resnet18")])
def test_load_training_and_legacy_checkpoints(tmp_path, kind, key, arch):
    model = FrameCNN(arch, pretrained=False) if kind == "m1" else CNNLSTM(arch, hidden_size=8, pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({key: model.state_dict(), "config": {"arch": arch}}, path)
    loaded = load_model(Namespace(model=kind, weights=str(path)), torch.device("cpu"))
    assert not loaded.training
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())


def test_threshold_uses_validation_metrics_and_allows_override(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"threshold": 0.08}))
    args = Namespace(weights=str(tmp_path / "best.pt"), metrics=None, threshold=None)
    assert resolve_threshold(args) == pytest.approx(0.08)
    args.threshold = 0.7
    assert resolve_threshold(args) == 0.7


def test_camera_falls_back_after_first_frame_failure(monkeypatch):
    import inference
    import numpy as np
    monkeypatch.setattr(inference.os, "name", "nt")
    calls = []
    class Capture:
        released = False
        def __init__(self, source, backend):
            self.backend = backend
            calls.append(self)
        def isOpened(self):
            return True
        def read(self):
            return (False, None) if self.backend == inference.cv2.CAP_DSHOW else (True, np.zeros((2, 2, 3)))
        def release(self):
            self.released = True
        def getBackendName(self):
            return "mock"
    monkeypatch.setattr(inference.cv2, "VideoCapture", Capture)
    cap, frame = inference.open_capture(0)
    assert calls[0].released
    assert cap is calls[1]
    assert frame.shape == (2, 2, 3)


def test_dashboard_pipeline_callback_stop_releases_camera_and_finalizes_session(tmp_path, monkeypatch):
    import inference
    import numpy as np
    import threading
    from monitoring import connect
    from live_camera import LiveCamera

    class Capture:
        released = False
        def get(self, prop):
            return 10.
        def read(self):
            return True, np.zeros((96, 128, 3), dtype=np.uint8)
        def release(self):
            self.released = True

    cap = Capture()
    monkeypatch.setattr(inference, "open_capture", lambda *args: (cap, cap.read()[1]))
    monkeypatch.setattr(inference, "load_model", lambda *args: lambda tensor: torch.tensor([-10.]))
    seen = []

    def runner(args, on_frame, stop_event):
        args.device = "cpu"
        args.stride = 1
        args.no_boxes = True
        def callback(frame, telemetry):
            seen.append(telemetry)
            on_frame(frame, telemetry)
            if len(seen) == 2:
                stop_event.set()
        inference.run_inference(args, callback, stop_event)

    camera = LiveCamera(tmp_path, runner=runner)
    camera.project = tmp_path
    weights = tmp_path / "results/m1/best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    camera.start("m1")
    camera.thread.join(timeout=10)
    assert not camera.thread.is_alive()
    assert len(seen) == 2 and seen[-1]["ready"]
    assert cap.released
    assert camera.snapshot()["status"] == "stopped"
    with connect(tmp_path) as db:
        assert db.execute("SELECT status FROM sessions").fetchone()[0] == "completed"


def test_m3_file_clock_waits_for_five_second_window_in_fast_batch(tmp_path, monkeypatch):
    import inference
    import numpy as np
    from live_camera import LiveCamera
    from threading import Event

    class Capture:
        index = 0
        released = False
        def get(self, prop):
            return 30.
        def read(self):
            self.index += 1
            return self.index <= 180, np.full((64, 64, 3), self.index - 1, dtype=np.uint8)
        def release(self):
            self.released = True

    camera = LiveCamera(tmp_path)
    camera.project = tmp_path
    weights = tmp_path / "results/m3/best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    (tmp_path / "yolo11n.pt").touch()
    source = tmp_path / "input.mp4"
    source.touch()
    cap, stop = Capture(), Event()
    inputs = []

    def model(tensor):
        inputs.append(tensor.clone())
        return torch.tensor([-10.])

    monkeypatch.setattr(inference, "load_model", lambda *args: model)
    monkeypatch.setattr(inference, "open_capture", lambda *args: (cap, cap.read()[1]))
    monkeypatch.setattr(inference, "make_person_locator", lambda **kwargs: lambda image: [])
    monkeypatch.setattr(inference, "preprocess_frame", lambda image, size: torch.tensor(float(image[0, 0, 0])).reshape(1, 1, 1))

    def runner(args, on_frame, stop_event):
        args.source = str(source)
        args.device = "cpu"
        args.threshold = .5
        def callback(frame, state):
            if state["ready"]:
                assert state["window_coverage"] >= 4.8
                stop_event.set()
        inference.run_inference(args, callback, stop_event)

    camera.runner = runner
    camera.start("m3")
    camera.thread.join(timeout=15)
    assert not camera.thread.is_alive()
    assert camera.snapshot()["status"] == "stopped", camera.snapshot()
    assert cap.released and len(inputs) == 1
    selected = inputs[0].flatten().tolist()
    assert len(selected) == 16
    assert (max(selected) - min(selected)) / 30 >= 4.4
    assert cap.index >= 145
