import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fight_data import ClipDataset, FrameDataset, list_clips
from models import CNNLSTM, build_cnn
from train_utils import fit, predict_clips


def write_frames(root, split, label, video, n=4, value=100):
    folder = root / split / label / video
    folder.mkdir(parents=True)
    for i in range(n):
        image = np.full((40, 60, 3), value, dtype=np.uint8)
        image[:, :30] = 0  # nua trai den de phat hien phep lat ngang
        cv2.imwrite(str(folder / f"frame_{i:02d}.jpg"), image)


@pytest.fixture
def frames_root(tmp_path):
    write_frames(tmp_path, "train", "Fight", "f1")
    write_frames(tmp_path, "train", "NonFight", "n1")
    write_frames(tmp_path, "val", "Fight", "f2")
    return tmp_path


def test_list_clips_maps_fight_to_positive_label(frames_root):
    clips = {c["video"]: c for c in list_clips(frames_root, "train")}
    assert clips["f1"]["label"] == 1
    assert clips["n1"]["label"] == 0
    assert len(clips["f1"]["frames"]) == 4


def test_list_clips_never_mixes_splits(frames_root):
    assert {c["video"] for c in list_clips(frames_root, "val")} == {"f2"}


def test_frame_dataset_yields_one_item_per_frame(frames_root):
    ds = FrameDataset(frames_root, "train", image_size=32)
    assert len(ds) == 8
    image, label, video = ds[0]
    assert image.shape == (3, 32, 32)
    assert video in {"f1", "n1"}


def test_clip_dataset_yields_sequence_for_each_clip(frames_root):
    ds = ClipDataset(frames_root, "train", image_size=32)
    assert len(ds) == 2
    clip, label, video = ds[0]
    assert clip.shape == (4, 3, 32, 32)


def test_clip_augmentation_is_identical_across_frames(frames_root):
    ds = ClipDataset(frames_root, "train", image_size=32, augment=True)
    for _ in range(10):
        clip, _, _ = ds[0]
        for frame in clip[1:]:
            assert torch.allclose(frame, clip[0])


def test_cnn_outputs_one_logit_per_image():
    model = build_cnn("resnet18", pretrained=False)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2,)


def test_cnn_lstm_outputs_one_logit_per_clip():
    model = CNNLSTM("mobilenet_v2", hidden_size=8, pretrained=False)
    assert model(torch.zeros(2, 4, 3, 32, 32)).shape == (2,)


def test_frozen_backbone_does_not_change_during_training():
    model = CNNLSTM("resnet18", hidden_size=8, pretrained=False, freeze_backbone=True)
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    loss = model(torch.rand(2, 3, 3, 32, 32)).sum()
    loss.backward()
    optimizer.step()
    for key, value in model.backbone.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_predict_clips_averages_frames_of_a_clip(frames_root):
    class Half(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(len(x))

    ds = FrameDataset(frames_root, "train", image_size=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    clips = predict_clips(Half(), loader, "cpu")
    assert sorted(clips["video"]) == ["f1", "n1"]
    assert clips["prob"].tolist() == pytest.approx([0.5, 0.5])


def test_fit_stops_early_and_keeps_best_checkpoint(frames_root, tmp_path):
    ds = FrameDataset(frames_root, "train", image_size=16)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    model = build_cnn("mobilenet_v2", pretrained=False)
    out = tmp_path / "run"
    history = fit(model, loader, loader, out, epochs=20, patience=1, lr=0.0, device="cpu")
    # lr = 0 nen F1 khong doi: dung sau epoch 2 thay vi chay du 20 epoch.
    assert len(history) == 2
    assert (out / "best.pt").exists()
    assert (out / "history.csv").exists()
