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
