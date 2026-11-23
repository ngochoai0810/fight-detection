"""Inference truc tiep tu webcam hoac file video de phat hien danh nhau (Fight Detection).

Ho tro ca 3 mo hinh:
  - m1: ResNet18 Frame-level (nhanh nhat)
  - m2: YOLO11-cls
  - m3: YOLO11 Person Crop + ResNet18 + LSTM (chinh xac nhat)

Cach dung:
  # Chay truc tiep tren webcam:
  python src/inference.py --source 0 --model m1 --weights path/to/m1/best.pt

  # Chay tren file video:
  python src/inference.py --source test_video.mp4 --model m3 --weights path/to/m3/best.pt

  # Luu video ket qua:
  python src/inference.py --source test_video.mp4 --model m1 --weights path/to/m1/best.pt --save out.mp4
"""
import argparse
import collections
import json
import hashlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from models import FrameCNN, CNNLSTM
from alerts import AlertOverlay, VoiceAlert, AlertConfirmation
from telegram_alerts import TelegramAlert
from monitoring import Monitor
from crop_persons import BehaviorTracker, make_person_locator, union_box
from temporal import TemporalWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Fight Detection Inference on Video/Webcam")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) hoac duong dan toi video file")
    parser.add_argument("--model", type=str, default="m1", choices=["m1", "m2", "m3"], help="Kien truc mo hinh")
    parser.add_argument("--weights", type=str, required=True, help="Duong dan toi file checkpoint (.pt)")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt", help="Trong so YOLO phat hien nguoi va khoanh vung")
    parser.add_argument("--image-size", type=int, default=224, help="Kich thuoc frame dua vao model")
    parser.add_argument("--threshold", type=float, default=None, help="Nguong xac suat de canh bao Fight (0.0 - 1.0)")
    parser.add_argument("--seq-len", type=int, default=16, help="So luong frame trong mot window (cho M3 hoac smooth M1)")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Thoi luong cua so M3; mac dinh 5 giay nhu clip huan luyen")
    parser.add_argument("--stride", type=int, default=2, help="Lay mau moi N frame de tiet kiem tai tinh toan")
    parser.add_argument("--save", type=str, default=None, help="Duong dan luu video ket qua (neu can)")
    parser.add_argument("--output-dir", default="runs/live", help="Thu muc clip bang chung, log va hieu nang")
    parser.add_argument("--device", type=str, default=None, help="cuda hoac cpu (mac dinh tu chon)")
    parser.add_argument("--metrics", help="metrics.json chua threshold validation; mac dinh canh checkpoint")
    parser.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"], default="auto")
    parser.add_argument("--camera-width", type=int, default=1280, help="Do rong camera mong muon")
    parser.add_argument("--camera-height", type=int, default=720, help="Do cao camera mong muon")
    parser.add_argument("--camera-mjpg", action="store_true", help="Thu dinh dang MJPG neu camera USB ho tro")
    parser.add_argument("--no-voice", action="store_true", help="Tat giong noi canh bao")
    parser.add_argument("--voice-cooldown", type=float, default=10, help="So giay nghi giua cac lan doc canh bao")
    parser.add_argument("--alert-hold", type=float, default=1.0, help="So giay diem phai lien tuc vuot nguong truoc khi canh bao; 0 de tat")
    parser.add_argument("--pre-seconds", type=float, default=2.5, help="So giay luu dem truoc khi canh bao duoc xac nhan")
    parser.add_argument("--no-boxes", action="store_true", help="Tat YOLO khoanh vung doi tuong bo sung cho M1/M2")
    parser.add_argument("--headless", action="store_true", help="Chay khong mo cua so cv2.imshow (phu hop xu ly batch/server)")
    args = parser.parse_args()
    if not 0 <= args.alert_hold <= 60:
        parser.error("alert-hold phai trong [0, 60]")
    if not 0 <= args.pre_seconds <= 15:
        parser.error("pre-seconds phai trong [0, 15]")
    if not 1 <= args.voice_cooldown <= 3600:
        parser.error("voice-cooldown phai trong [1, 3600]")
    if args.camera_width < 1 or args.camera_height < 1:
        parser.error("camera-width va camera-height phai lon hon 0")
    if args.stride < 1 or args.seq_len < 1 or args.image_size < 1:
        parser.error("stride, seq-len va image-size phai lon hon 0")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("threshold phai nam trong [0, 1]")
    if not 0.5 <= args.window_seconds <= 30:
        parser.error("window-seconds phai nam trong [0.5, 30]")
    return args


def load_model(args, device):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Khong tim thay file trong so: {weights_path}")

    if args.model == "m2":
        from ultralytics import YOLO
        return YOLO(str(weights_path))

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    batch_counts = [v.item() for k, v in state_dict.items() if k.endswith("num_batches_tracked")]
    if "dummy" in weights_path.name.lower() or (batch_counts and all(n == 0 for n in batch_counts)):
        print("[!] CANH BAO: checkpoint co dau hieu chua huan luyen (dummy/BatchNorm chua thay batch).")
        print("[!] Can best.pt tu lan train that; metrics.json hien co chua duoc xac minh cho checkpoint nay.")
    arch = config.get("arch")
    if arch is None:
        arch = "resnet18" if "backbone.conv1.weight" in state_dict else "mobilenet_v2"
    if args.model == "m1":
        model = FrameCNN(arch=arch, pretrained=False)
    else:
        hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
        model = CNNLSTM(arch=arch, hidden_size=hidden_size, pretrained=False)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def resolve_threshold(args):
    if args.threshold is not None:
        return args.threshold
    metrics = Path(args.metrics) if args.metrics else Path(args.weights).with_name("metrics.json")
    if metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        expected_hash = payload.get("checkpoint_sha256")
        if expected_hash and hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Threshold/metrics khong thuoc checkpoint nay")
        value = float(payload["threshold"])
        if not 0 <= value <= 1:
            raise ValueError("Threshold trong metrics.json phai nam trong [0, 1]")
        return value
    print("[*] Khong co metrics.json; dung threshold 0.5")
    return 0.5


def preprocess_frame(bgr_image, image_size):
    """Chuyen doi frame BGR -> RGB Tensor chuan hoa ImageNet."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor


def crop_from_boxes(image, boxes, margin=0.1):
    """Crop vung hop nhat tu danh sach box nguoi da phat hien."""
    h, w = image.shape[:2]
    crop_box = union_box(boxes, w, h, margin=margin)
    if crop_box is None or crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return image
    return image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]


def crop_person_union(image, detector, margin=0.1):
    """Phat hien nguoi bang YOLO va crop vung hop nhat chua cac nguoi cho M3."""
    return crop_from_boxes(image, detector(image), margin=margin)


def open_capture(source, backend="auto", width=None, height=None, mjpg=False):
    """Validate a first frame before choosing a backend; release failed devices."""
    backends = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    candidates = [backend]
    if backend == "auto" and isinstance(source, int) and os.name == "nt":
        candidates = ["dshow", "msmf", "auto"]
    for name in candidates:
        cap = cv2.VideoCapture(source, backends[name])
        if cap.isOpened():
            if isinstance(source, int):
                if mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"[*] Camera/video backend: {cap.getBackendName()}")
                print(f"[*] Do phan giai thuc te: {frame.shape[1]}x{frame.shape[0]}")
                return cap, frame
        cap.release()
        print(f"[!] Khong doc duoc frame voi backend {name}")
    raise RuntimeError(
        f"Khong doc duoc camera/video: {source}. Dong ung dung dang dung camera, "
        "kiem tra quyen Camera cua Windows hoac thu --source 1."
    )


def run_inference(args, on_frame=None, stop_event=None):
    """Run the shared CLI/dashboard pipeline; release camera and evidence on stop."""
    args.threshold = resolve_threshold(args)
    print(f"[*] Threshold: {args.threshold:.6f}")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Dang chay tren thiet bi: {device}")

    print(f"[*] Dang tai mo hinh {args.model.upper()} tu {args.weights}...")
    model = load_model(args, device)

    person_locator = None
    need_yolo = args.model == "m3" or (not args.no_boxes and Path(args.yolo_weights).is_file())
    if need_yolo:
        print(f"[*] Dang tai YOLO khoanh vung doi tuong tu {args.yolo_weights}...")
        person_locator = make_person_locator(weights=args.yolo_weights, conf=0.25, device=device)

    if stop_event is not None and stop_event.is_set():
        return

    tracker = BehaviorTracker()
    overlay = AlertOverlay()
    confirmation = AlertConfirmation(args.alert_hold)
    is_fight = False
    voice = VoiceAlert(enabled=not args.no_voice, cooldown=args.voice_cooldown)
    telegram = TelegramAlert.from_env()

    source = int(args.source) if args.source.isdigit() else args.source
    cap, first_frame = open_capture(source, args.camera_backend, args.camera_width, args.camera_height, args.camera_mjpg)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0
    file_source = isinstance(source, str) and Path(source).is_file()
    args.window_seconds = getattr(args, "window_seconds", 5.0)

    writer = None
    if args.save:
        from monitoring import open_evidence_writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer, save_size, _ = open_evidence_writer(args.save, source_fps, (width, height))
        print(f"[*] Se luu video ket qua vao: {args.save}")

    window = TemporalWindow(args.seq_len, args.window_seconds)
    prob_history = collections.deque(maxlen=10)

    frame_idx = 0
    fight_prob = 0.0
    fps_display = 0.0
    detections = {"zone": None, "zone_kind": None, "persons": []}
    prev_time = stream_started = time.monotonic()

    print("[*] Bat dau nhan dien. Nhan 'q' tren cua so hien thi de dung.")

    monitor = None
    try:
        session_cfg = dict(vars(args), source_fps=round(float(source_fps), 2))
        monitor = Monitor(args.output_dir, session_cfg, pre_seconds=args.pre_seconds)
        while stop_event is None or not stop_event.is_set():
            frame_started = time.perf_counter()
            inference_ms = None
            if first_frame is not None:
                ret, raw_frame = True, first_frame
                first_frame = None
            else:
                ret, raw_frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    raise RuntimeError("Camera bi ngat khi doc frame. Thu backend khac hoac kiem tra ket noi USB.")
                break

            frame_idx += 1
            curr_time = time.monotonic()
            frame_time = (frame_idx - 1) / source_fps if file_source else curr_time - stream_started
            fps_display = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            if frame_idx % args.stride == 0:
                predicted = False
                inference_started = time.perf_counter()
                person_boxes = person_locator(raw_frame) if person_locator is not None else []
                detections = tracker.update(raw_frame, person_boxes=person_boxes, recompute=True)

                if args.model == "m1":
                    tensor = preprocess_frame(raw_frame, args.image_size).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logit = model(tensor)
                        fight_prob = torch.sigmoid(logit).item()
                    prob_history.append(fight_prob)
                    predicted = True
                    smoothed_prob = sum(prob_history) / len(prob_history)

                elif args.model == "m2":
                    results = model.predict(raw_frame, imgsz=args.image_size, device=str(device), verbose=False)[0]
                    names = results.names
                    fight_idx = 1
                    for idx, name in names.items():
                        if "fight" in name.lower() and "non" not in name.lower():
                            fight_idx = idx
                            break
                    probs = results.probs.data.cpu().numpy()
                    fight_prob = float(probs[fight_idx])
                    prob_history.append(fight_prob)
                    predicted = True
                    smoothed_prob = sum(prob_history) / len(prob_history)

                elif args.model == "m3":
                    cropped = crop_from_boxes(raw_frame, person_boxes)
                    tensor = preprocess_frame(cropped, args.image_size)
                    window.add(frame_time, tensor)
                    sequence = window.sample()

                    if sequence is not None:
                        clip_tensor = torch.stack(sequence).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logit = model(clip_tensor)
                            fight_prob = torch.sigmoid(logit).item()
                        prob_history.append(fight_prob)
                        predicted = True
                    else:
                        prob_history.clear()
                    smoothed_prob = sum(prob_history) / len(prob_history) if prob_history else fight_prob

                if predicted:
                    inference_ms = (time.perf_counter() - inference_started) * 1000
            else:
                detections = tracker.update(raw_frame, recompute=False, cached=detections)
                smoothed_prob = sum(prob_history) / len(prob_history) if prob_history else fight_prob

            ready = bool(prob_history)
            above_threshold = ready and smoothed_prob >= args.threshold
            if frame_idx % args.stride == 0:
                is_fight = confirmation.update(above_threshold, now=frame_time)
            pending = above_threshold and not is_fight
            voice.update(is_fight)
            telegram.update(is_fight, smoothed_prob, args.model, args.source)

            display_frame = overlay.draw(
                raw_frame.copy(),
                is_fight,
                ready,
                smoothed_prob,
                args.threshold,
                args.model,
                fps_display,
                pending=pending,
                detections=detections,
            )

            if writer is not None:
                writer.write(display_frame[: save_size[1], : save_size[0]])

            key = 0
            if not args.headless:
                cv2.imshow("Fight Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF

            processing_ms = (time.perf_counter() - frame_started) * 1000
            monitor.record(
                raw_frame,
                is_fight,
                smoothed_prob,
                inference_ms,
                processing_ms,
                detections=detections,
                pending=pending,
            )
            if on_frame is not None:
                on_frame(display_frame, {
                    "score": smoothed_prob, "threshold": args.threshold,
                    "active": is_fight, "pending": pending, "ready": ready,
                    "fps": fps_display, "model": args.model,
                    "window_seconds": args.window_seconds if args.model == "m3" else None,
                    "window_coverage": round(window.coverage, 2) if args.model == "m3" else None,
                })
            if key == ord("q"):
                break

    finally:
        voice.close()
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if monitor is not None:
            monitor.close()
    print("[*] Da ket thuc chuong trinh.")


def main():
    run_inference(parse_args())


if __name__ == "__main__":
    main()
