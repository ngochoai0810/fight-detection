import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_video(path, kind, n_frames=20, size=(96, 72), fps=10, seed=0):
    """Video tong hop: 'Fight' co hai khoi sang nhay ngau nhien, 'NonFight' co mot khoi troi cham."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    rng = np.random.default_rng(seed)
    width, height = size
    for i in range(n_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        if kind == "Fight":
            for color in ((0, 0, 255), (0, 255, 255)):
                x, y = rng.integers(0, width - 24), rng.integers(0, height - 24)
                frame[y : y + 24, x : x + 24] = color
        else:
            x = (seed * 7 + i) % (width - 24)
            frame[20:44, x : x + 24] = (255, 0, 0)
        writer.write(frame)
    writer.release()


@pytest.fixture
def make_video():
    return write_video


@pytest.fixture
def rwf_root(tmp_path):
    """Cay thu muc giong RWF-2000: 16 clip train va 4 clip val moi lop."""
    root = tmp_path / "RWF-2000"
    for label in ("Fight", "NonFight"):
        for i in range(8):
            write_video(root / "train" / label / f"{label}src{i}_0.avi", label, seed=i)
        for i in range(2):
            write_video(root / "val" / label / f"{label}test{i}_0.avi", label, seed=100 + i)
    return root
