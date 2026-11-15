"""Dung YOLO11 phat hien nguoi, cat vung bao tat ca moi nguoi trong frame (dau vao cua M3)
va khoanh vung khu vuc tuong tac / hanh vi danh nhau cho giam sat truc tiep.

    python src/crop_persons.py --frames data/frames --out data/crops
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

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


def make_person_locator(weights="yolo11n.pt", conf=0.25, imgsz=640, device=None):
    """Tra ve ham phat hien nguoi kem do tin cay [(x1, y1, x2, y2, conf), ...]."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def locate(image):
        kwargs = dict(classes=[0], conf=conf, imgsz=imgsz, verbose=False)
        if device is not None:
            kwargs["device"] = str(device)
        result = model.predict(image, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        return [(*box, float(c)) for box, c in zip(xyxy, confs)]

    return locate


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _boxes_proximate(a, b, frame_w):
    """Kiem tra 2 nguoi dung gan hoac giao nhau (co kha nang tuong tac / xo xat)."""
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    pad_x = max(0.35 * (aw + bw), 0.06 * frame_w)
    pad_y = 0.25 * (ah + bh)
    return not (
        a[2] + pad_x < b[0]
        or b[2] + pad_x < a[0]
        or a[3] + pad_y < b[1]
        or b[3] + pad_y < a[1]
    )


def _region_motion(diff_map, box, width, height):
    if diff_map is None:
        return 0.0
    dh, dw = diff_map.shape[:2]
    sx, sy = dw / max(width, 1), dh / max(height, 1)
    x1 = max(0, min(dw - 1, int(box[0] * sx)))
    y1 = max(0, min(dh - 1, int(box[1] * sy)))
    x2 = max(x1 + 1, min(dw, int(round(box[2] * sx))))
    y2 = max(y1 + 1, min(dh, int(round(box[3] * sy))))
    roi = diff_map[y1:y2, x1:x2]
    return float(roi.mean()) if roi.size else 0.0


def _motion_fallback_box(diff_map, width, height):
    """Tim vung chuyen dong manh nhat khi YOLO khong bat duoc nguoi hoac bi che khuat."""
    if diff_map is None or diff_map.size == 0:
        return None
    _, mask = cv2.threshold(diff_map, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    dh, dw = diff_map.shape[:2]
    min_area = 0.012 * dw * dh
    active = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    if not active:
        return None
    active.sort(key=lambda r: r[2] * r[3], reverse=True)
    top = active[:3]
    x1 = min(r[0] for r in top) / dw * width
    y1 = min(r[1] for r in top) / dh * height
    x2 = max(r[0] + r[2] for r in top) / dw * width
    y2 = max(r[1] + r[3] for r in top) / dh * height
    return union_box([(x1, y1, x2, y2)], width, height, margin=0.06)


def localize_behavior(person_boxes, width, height, diff_map=None, margin=0.07):
    """Khoanh vung khu vuc hanh vi va danh sach doi tuong chuan hoa [0..1].

    Uu tien cum >= 2 nguoi dung gan nhau co chuyen dong cao nhat; neu khong co cum
    tuong tac thi chon doi tuong co chuyen dong manh nhat hoac fallback sang vung chuyen dong.
    """
    if width <= 0 or height <= 0:
        return {"zone": None, "zone_kind": None, "persons": []}

    parsed = []
    for item in person_boxes or []:
        x1, y1, x2, y2 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
        conf = float(item[4]) if len(item) >= 5 else 1.0
        x1, y1 = max(0.0, min(width, x1)), max(0.0, min(height, y1))
        x2, y2 = max(0.0, min(width, x2)), max(0.0, min(height, y2))
        if x2 > x1 + 1 and y2 > y1 + 1:
            motion = _region_motion(diff_map, (x1, y1, x2, y2), width, height)
            area = (x2 - x1) * (y2 - y1)
            parsed.append({"xyxy": (x1, y1, x2, y2), "conf": conf, "motion": motion, "area": area})

    n = len(parsed)
    involved_indices = set()
    zone_px = None
    zone_kind = None

    if n >= 2:
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            comp = [i]
            visited[i] = True
            queue = [i]
            while queue:
                cur = queue.pop()
                for j in range(n):
                    if not visited[j] and _boxes_proximate(parsed[cur]["xyxy"], parsed[j]["xyxy"], width):
                        visited[j] = True
                        comp.append(j)
                        queue.append(j)
            clusters.append(comp)

        multi = [c for c in clusters if len(c) >= 2]
        if multi:
            best_cluster = max(
                multi,
                key=lambda c: (
                    sum(parsed[k]["motion"] for k in c) / len(c) * 2.0
                    + len(c) * 5.0
                    + sum(parsed[k]["area"] for k in c) / (width * height) * 10.0
                ),
            )
            involved_indices = set(best_cluster)
            zone_px = union_box([parsed[k]["xyxy"] for k in best_cluster], width, height, margin=margin)
            zone_kind = "interaction"

    if zone_px is None and n >= 1:
        best_idx = max(
            range(n),
            key=lambda k: parsed[k]["motion"] * 2.0 + parsed[k]["conf"] + parsed[k]["area"] / (width * height),
        )
        involved_indices = {best_idx}
        zone_px = union_box([parsed[best_idx]["xyxy"]], width, height, margin=margin)
        zone_kind = "person"

    if zone_px is None:
        fallback = _motion_fallback_box(diff_map, width, height)
        if fallback is not None:
            zone_px = fallback
            zone_kind = "motion"

    def norm_box(b):
        return [
            round(max(0.0, min(1.0, b[0] / width)), 4),
            round(max(0.0, min(1.0, b[1] / height)), 4),
            round(max(0.0, min(1.0, b[2] / width)), 4),
            round(max(0.0, min(1.0, b[3] / height)), 4),
        ]

    persons_out = [
        {
            "box": norm_box(p["xyxy"]),
            "conf": round(p["conf"], 3),
            "involved": idx in involved_indices,
        }
        for idx, p in enumerate(parsed)
    ]

    return {
        "zone": norm_box(zone_px) if zone_px is not None else None,
        "zone_kind": zone_kind,
        "persons": persons_out,
    }


class BehaviorTracker:
    """Theo doi chuyen dong giua cac frame va lam muot border line khoanh vung (EMA)."""

    def __init__(self, smooth=0.55):
        self.smooth = smooth
        self.prev_gray = None
        self.prev_zone = None

    def update(self, frame_bgr, person_boxes=None, recompute=True, cached=None):
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (min(w, 240), min(h, 180)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, self.prev_gray) if self.prev_gray is not None and self.prev_gray.shape == gray.shape else None
        self.prev_gray = gray

        if not recompute and cached is not None:
            return cached

        det = localize_behavior(person_boxes, w, h, diff_map=diff)
        zone = det["zone"]
        if zone is not None and self.prev_zone is not None and _box_iou(zone, self.prev_zone) >= 0.2:
            a = self.smooth
            zone = [round(a * z + (1.0 - a) * pz, 4) for z, pz in zip(zone, self.prev_zone)]
            det["zone"] = zone
        self.prev_zone = zone
        return det


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
