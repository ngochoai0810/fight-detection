"""Kiem tra RWF-2000, chia train/val/test theo video va trich xuat frame.

    python src/prepare_data.py --root D:/RWF-2000 --splits splits/splits.csv --frames data/frames
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("Fight", "NonFight")
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def read_image(path):
    """Doc anh, ho tro ca duong dan co dau tieng Viet tren Windows."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"Khong ma hoa duoc anh: {path}")
    buffer.tofile(str(path))


def scan_dataset(root):
    """Liet ke clip theo cau truc <root>/{train,val}/{Fight,NonFight}/*."""
    root = Path(root)
    rows = []
    for source_split in ("train", "val"):
        for label in LABELS:
            folder = root / source_split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Thieu thu muc: {folder}")
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in VIDEO_EXTS:
                    rows.append(
                        {
                            "video": path.stem,
                            "label": label,
                            "source_split": source_split,
                            "relpath": path.relative_to(root).as_posix(),
                        }
                    )
    return pd.DataFrame(rows)


def source_id(video):
    """Bo chi so clip o cuoi ten: 'abc_12' -> 'abc'. Cac clip cung nguon co the rat giong nhau."""
    return re.sub(r"_\d+$", "", video)


def make_splits(df, val_fraction=0.1, seed=42):
    """Giu 'val' goc lam test; tach val tu 'train' goc, phan tang va khong tach clip cung nguon."""
    out = df.copy()
    out["split"] = "train"
    out.loc[out["source_split"] == "val", "split"] = "test"

    pool = out[out["split"] == "train"]
    n_val = round(val_fraction * len(out))
    folds = StratifiedGroupKFold(n_splits=max(2, round(len(pool) / n_val)), shuffle=True, random_state=seed)
    _, val_positions = next(folds.split(pool, pool["label"], groups=pool["video"].map(source_id)))
    out.loc[pool.index[val_positions], "split"] = "val"
    return out


def check_no_leak(df):
    """Bao loi neu ro ri trong train/val; tra ve cac nguon dung chung voi test de ghi vao bao cao."""
    per_video = df.groupby(["video", "label"])["split"].nunique()
    repeated = per_video[per_video > 1]
    if len(repeated):
        raise ValueError(f"Video nam o nhieu tap: {sorted(v for v, _ in repeated.index)[:10]}")

    sources = df["video"].map(source_id)
    develop = df["split"].isin(["train", "val"])
    per_source = df[develop].groupby(sources[develop])["split"].nunique()
    shared = per_source[per_source > 1]
    if len(shared):
        raise ValueError(f"Nguon video nam o ca train va val: {sorted(shared.index)[:10]}")

    test_sources = set(sources[df["split"] == "test"])
    return sorted(test_sources & set(sources[develop]))


def sample_indices(n_frames, k):
    """Chon k vi tri cach deu nhau, lay diem giua moi doan."""
    return ((np.arange(k) + 0.5) * n_frames / k).astype(int)


def probe_video(path):
    capture = cv2.VideoCapture(str(path))
    info = {"readable": False, "fps": 0.0, "n_frames": 0, "width": 0, "height": 0, "duration_s": 0.0}
    if capture.isOpened():
        readable, _ = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {
            "readable": bool(readable) and n_frames > 0,
            "fps": fps,
            "n_frames": n_frames,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": n_frames / fps if fps > 0 else 0.0,
        }
    capture.release()
    return info


def read_frames(path, size=None):
    """Doc toan bo frame vi so frame trong metadata co the sai; size la canh anh vuong, None giu nguyen."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (size, size)) if size else frame)
    capture.release()
    return frames


def extract_frames(video_path, out_dir, num_frames=16, size=None):
    frames = read_frames(video_path, size)
    if not frames:
        return 0
    for position, index in enumerate(sample_indices(len(frames), num_frames)):
        write_image(Path(out_dir) / f"frame_{position:02d}.jpg", frames[index])
    return num_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Thu muc RWF-2000")
    parser.add_argument("--splits", default="splits/splits.csv")
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    index = scan_dataset(root)
    probes = pd.DataFrame([probe_video(root / rel) for rel in index["relpath"]])
    index = pd.concat([index, probes], axis=1)

    bad = index[~index["readable"]]
    if len(bad):
        print(f"Bo {len(bad)} clip khong doc duoc: {bad['relpath'].tolist()}")
    index = index[index["readable"]].reset_index(drop=True)

    splits = make_splits(index, args.val_fraction, args.seed)
    overlap = check_no_leak(splits)
    if overlap:
        print(f"CANH BAO: {len(overlap)} nguon video xuat hien o ca test va train/val: {overlap[:10]}")

    splits_path = Path(args.splits)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(splits_path, index=False)
    print(splits.groupby(["split", "label"]).size())

    for row in splits.itertuples():
        out_dir = Path(args.frames) / row.split / row.label / row.video
        if len(list(out_dir.glob("*.jpg"))) < args.num_frames:
            extract_frames(root / row.relpath, out_dir, args.num_frames)


if __name__ == "__main__":
    main()
