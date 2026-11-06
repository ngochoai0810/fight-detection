import cv2
import numpy as np

from crop_persons import crop_frames_dir, union_box


def test_union_box_covers_all_people_with_margin():
    boxes = [(10, 10, 20, 20), (40, 30, 50, 60)]
    assert union_box(boxes, width=100, height=100, margin=0.1) == (6, 5, 54, 65)


def test_union_box_is_clipped_to_image():
    assert union_box([(0, 0, 100, 100)], width=100, height=100, margin=0.5) == (0, 0, 100, 100)


def test_union_box_none_without_people():
    assert union_box([], width=100, height=100) is None


def _write_frame(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((100, 200, 3), dtype=np.uint8))


def test_crop_uses_detector_box_and_resizes(tmp_path):
    _write_frame(tmp_path / "in" / "frame_00.jpg")
    stats = crop_frames_dir(
        tmp_path / "in", tmp_path / "out", detector=lambda img: [(50, 20, 100, 80)], size=64
    )
    assert cv2.imread(str(tmp_path / "out" / "frame_00.jpg")).shape == (64, 64, 3)
    assert stats == {"frames": 1, "no_person": 0}


def test_crop_falls_back_to_full_frame_without_people(tmp_path):
    _write_frame(tmp_path / "in" / "frame_00.jpg")
    stats = crop_frames_dir(tmp_path / "in", tmp_path / "out", detector=lambda img: [], size=64)
    assert (tmp_path / "out" / "frame_00.jpg").exists()
    assert stats == {"frames": 1, "no_person": 1}


def test_crop_skips_frames_already_cached(tmp_path):
    _write_frame(tmp_path / "in" / "frame_00.jpg")
    calls = []

    def detector(img):
        calls.append(1)
        return []

    crop_frames_dir(tmp_path / "in", tmp_path / "out", detector=detector, size=64)
    crop_frames_dir(tmp_path / "in", tmp_path / "out", detector=detector, size=64)
    assert len(calls) == 1
