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
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from models import FrameCNN, CNNLSTM
from alerts import AlertOverlay, VoiceAlert, AlertConfirmation


def parse_args():
    parser = argparse.ArgumentParser(description="Fight Detection Inference on Video/Webcam")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) hoac duong dan toi video file")
    parser.add_argument("--model", type=str, default="m1", choices=["m1", "m2", "m3"], help="Kien truc mo hinh")
    parser.add_argument("--weights", type=str, required=True, help="Duong dan toi file checkpoint (.pt)")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt", help="Trọng so YOLO cho M3 crop hoac M2 cls")
    parser.add_argument("--image-size", type=int, default=224, help="Kich thuoc frame dua vao model")
    parser.add_argument("--threshold", type=float, default=None, help="Nguong xac suat de canh bao Fight (0.0 - 1.0)")
    parser.add_argument("--seq-len", type=int, default=16, help="So luong frame trong mot window (cho M3 hoac smooth M1)")
    parser.add_argument("--stride", type=int, default=2, help="Lay mau moi N frame de tiet kiem tai tinh toan")
    parser.add_argument("--save", type=str, default=None, help="Duong dan luu video ket qua (neu can)")
    parser.add_argument("--device", type=str, default=None, help="cuda hoac cpu (mac dinh tu chon)")
    parser.add_argument("--metrics", help="metrics.json chua threshold validation; mac dinh canh checkpoint")
    parser.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"], default="auto")
    parser.add_argument("--camera-width", type=int, default=1280, help="Do rong camera mong muon")
    parser.add_argument("--camera-height", type=int, default=720, help="Do cao camera mong muon")
    parser.add_argument("--camera-mjpg", action="store_true", help="Thu dinh dang MJPG neu camera USB ho tro")
    parser.add_argument("--no-voice", action="store_true", help="Tat giong noi canh bao")
    parser.add_argument("--voice-cooldown", type=float, default=10, help="So giay nghi giua cac lan doc canh bao")
    parser.add_argument("--alert-hold", type=float, default=1.0, help="So giay diem phai lien tuc vuot nguong truoc khi canh bao; 0 de tat")
    args = parser.parse_args()
    if not 0 <= args.alert_hold <= 60:
        parser.error("alert-hold phai trong [0, 60]")
    if not 1 <= args.voice_cooldown <= 3600:
        parser.error("voice-cooldown phai trong [1, 3600]")
    if args.camera_width < 1 or args.camera_height < 1:
        parser.error("camera-width va camera-height phai lon hon 0")
    if args.stride < 1 or args.seq_len < 1 or args.image_size < 1:
        parser.error("stride, seq-len va image-size phai lon hon 0")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("threshold phai nam trong [0, 1]")
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
        value = float(json.loads(metrics.read_text(encoding="utf-8"))["threshold"])
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


def crop_person_union(image, detector, margin=0.1):
    """Phat hien nguoi bang YOLO va crop vung hop nhat chua cac nguoi cho M3."""
    h, w = image.shape[:2]
    boxes = detector(image)
    if len(boxes) == 0:
        return image
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    crop_box = (
        max(0, round(x1 - pad_x)),
        max(0, round(y1 - pad_y)),
        min(w, round(x2 + pad_x)),
        min(h, round(y2 + pad_y)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return image
    return image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]


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


def main():
    args = parse_args()
    args.threshold = resolve_threshold(args)
    print(f"[*] Threshold: {args.threshold:.6f}")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Dang chay tren thiet bi: {device}")

    # Load model phan loai
    print(f"[*] Dang tai mo hinh {args.model.upper()} tu {args.weights}...")
    model = load_model(args, device)

    # Neu dung M3, can them YOLO detector de crop nguoi
    detector = None
    if args.model == "m3":
        print(f"[*] Dang tai YOLO detector nguoi tu {args.yolo_weights}...")
        from crop_persons import make_yolo_detector
        detector = make_yolo_detector(weights=args.yolo_weights, conf=0.25)

    overlay = AlertOverlay()
    confirmation = AlertConfirmation(args.alert_hold)
    is_fight = False
    voice = VoiceAlert(enabled=not args.no_voice, cooldown=args.voice_cooldown)

    # Mo video / webcam source
    source = int(args.source) if args.source.isdigit() else args.source
    cap, first_frame = open_capture(source, args.camera_backend, args.camera_width, args.camera_height, args.camera_mjpg)

    # Thiet lap video writer neu luu file
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"[*] Se luu video ket qua vao: {args.save}")

    # Buffer luu tru frame
    frame_queue = collections.deque(maxlen=args.seq_len)
    prob_history = collections.deque(maxlen=10)

    frame_idx = 0
    fight_prob = 0.0
    fps_display = 0.0
    prev_time = time.time()

    print("[*] Bat dau nhan dien. Nhan 'q' tren cua so hien thi de dung.")

    try:
        while True:
            if first_frame is not None:
                ret, frame = True, first_frame
                first_frame = None
            else:
                ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    raise RuntimeError("Camera bi ngat khi doc frame. Thu backend khac hoac kiem tra ket noi USB.")
                break

            frame_idx += 1
            curr_time = time.time()
            fps_display = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Suy luan theo tung loai mo hinh
            if frame_idx % args.stride == 0:
                if args.model == "m1":
                    # ResNet18: du doan tung frame
                    tensor = preprocess_frame(frame, args.image_size).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logit = model(tensor)
                        fight_prob = torch.sigmoid(logit).item()
                    prob_history.append(fight_prob)
                    smoothed_prob = sum(prob_history) / len(prob_history)

                elif args.model == "m2":
                    # YOLO11-cls
                    results = model.predict(frame, imgsz=args.image_size, device=str(device), verbose=False)[0]
                    # Gia su class Fight la index 1 hoac lay tu names
                    names = results.names
                    fight_idx = 1
                    for idx, name in names.items():
                        if "fight" in name.lower() and "non" not in name.lower():
                            fight_idx = idx
                            break
                    probs = results.probs.data.cpu().numpy()
                    fight_prob = float(probs[fight_idx])
                    prob_history.append(fight_prob)
                    smoothed_prob = sum(prob_history) / len(prob_history)

                elif args.model == "m3":
                    # M3: Crop nguoi -> dua vao queue -> du doan qua CNN+LSTM
                    cropped = crop_person_union(frame, detector)
                    tensor = preprocess_frame(cropped, args.image_size)
                    frame_queue.append(tensor)

                    if len(frame_queue) == args.seq_len:
                        clip_tensor = torch.stack(list(frame_queue)).unsqueeze(0).to(device)  # (1, T, C, H, W)
                        with torch.no_grad():
                            logit = model(clip_tensor)
                            fight_prob = torch.sigmoid(logit).item()
                        prob_history.append(fight_prob)
                    smoothed_prob = sum(prob_history) / len(prob_history) if prob_history else fight_prob
            else:
                smoothed_prob = sum(prob_history) / len(prob_history) if prob_history else fight_prob

            # Ve overlay canh bao len man hinh
            ready = bool(prob_history)
            above_threshold = ready and smoothed_prob >= args.threshold
            if frame_idx % args.stride == 0:
                is_fight = confirmation.update(above_threshold)
            voice.update(is_fight)
            frame = overlay.draw(frame, is_fight, ready, smoothed_prob,
                                 args.threshold, args.model, fps_display,
                                 pending=above_threshold and not is_fight)

            if writer:
                writer.write(frame)

            cv2.imshow("Fight Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        voice.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print("[*] Da ket thuc chuong trinh.")


if __name__ == "__main__":
    main()

