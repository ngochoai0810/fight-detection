"""Dung YOLO11 phat hien nguoi, cat vung bao tat ca moi nguoi trong frame (dau vao cua M3).

    python src/crop_persons.py --frames data/frames --out data/crops
"""
import argparse
from pathlib import Path

import cv2

from prepare_data import read_image, write_image


def union_box(boxes, width, height, margin=0.1):
    """Hop cac box (x1, y1, x2, y2), noi rong theo margin va gioi han trong anh."""
    if len(boxes) == 0:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, round(x1 - pad_x)),
        max(0, round(y1 - pad_y)),
        min(width, round(x2 + pad_x)),
        min(height, round(y2 + pad_y)),
    )


def crop_frames_dir(in_dir, out_dir, detector, size=224, margin=0.1):
    """Cat moi frame trong mot clip; khong co nguoi thi giu ca frame. Bo qua frame da cat truoc do."""
    stats = {"frames": 0, "no_person": 0}
    for frame_path in sorted(Path(in_dir).glob("*.jpg")):
        target = Path(out_dir) / frame_path.name
        if target.exists():
            continue
        image = read_image(frame_path)
        height, width = image.shape[:2]
        box = union_box(detector(image), width, height, margin)
        stats["frames"] += 1
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            stats["no_person"] += 1
        else:
            image = image[box[1] : box[3], box[0] : box[2]]
        write_image(target, cv2.resize(image, (size, size)))
    return stats


def make_yolo_detector(weights="yolo11n.pt", conf=0.25, imgsz=640):
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(image):
        result = model.predict(image, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        return result.boxes.xyxy.cpu().tolist()

    return detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--out", default="data/crops")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector = make_yolo_detector(args.weights, args.conf)
    frames_root = Path(args.frames)
    total = {"frames": 0, "no_person": 0}
    for clip_dir in sorted(p for p in frames_root.glob("*/*/*") if p.is_dir()):
        stats = crop_frames_dir(clip_dir, Path(args.out) / clip_dir.relative_to(frames_root), detector, args.size)
        total = {key: total[key] + stats[key] for key in total}
    print(f"Da cat {total['frames']} frame, {total['no_person']} frame khong co nguoi (giu nguyen ca frame)")


if __name__ == "__main__":
    main()
