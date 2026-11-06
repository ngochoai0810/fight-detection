"""Danh gia mot mo hinh tu checkpoint o cap clip: chon nguong tren val, tinh chi so tren test.

    python src/evaluate.py --model m1
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from fight_data import ClipDataset, FrameDataset, list_clips
from metrics import aggregate_clips, choose_threshold, clip_metrics, comparison_table, errors_table
from models import CNNLSTM, build_cnn
from prepare_data import read_image
from train_utils import default_device, predict_clips


def per_second(fn, repeats=30):
    fn()  # lan dau thuong cham hon do khoi tao
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return repeats / (time.perf_counter() - start)


def load_torch_model(name, ckpt, device):
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if name == "m1":
        model = build_cnn(config["arch"], pretrained=False)
    else:
        model = CNNLSTM(config["arch"], config["hidden_size"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def evaluate_torch(args, device):
    model = load_torch_model(args.model, args.ckpt, device)
    dataset_cls = FrameDataset if args.model == "m1" else ClipDataset

    def predict(split):
        loader = DataLoader(dataset_cls(args.frames, split, args.image_size), args.batch_size)
        return predict_clips(model, loader, device)

    sample = dataset_cls(args.frames, "test", args.image_size)[0][0].unsqueeze(0).to(device)

    @torch.no_grad()
    def forward():
        model(sample)

    frames_per_call = sample.shape[1] if args.model == "m3" else 1
    return predict("val"), predict("test"), per_second(forward) * frames_per_call


def evaluate_yolo(args, device):
    from ultralytics import YOLO

    model = YOLO(str(args.ckpt))
    fight = next(i for i, name in model.names.items() if name == "Fight")

    def predict(split):
        rows = []
        for clip in list_clips(args.frames, split):
            results = model.predict(
                [str(p) for p in clip["frames"]], imgsz=args.image_size, device=device, verbose=False
            )
            rows += [
                {"video": clip["video"], "label": clip["label"], "prob": float(r.probs.data[fight])} for r in results
            ]
        return aggregate_clips(pd.DataFrame(rows))

    image = read_image(list_clips(args.frames, "test")[0]["frames"][0])
    fps = per_second(lambda: model.predict(image, imgsz=args.image_size, device=device, verbose=False))
    return predict("val"), predict("test"), fps


def add_detector_time(fps, args, device):
    """M3 con phai chay YOLO phat hien nguoi tren moi frame, nen tinh ca thoi gian do."""
    if not args.yolo_weights:
        print("CANH BAO: khong co --yolo-weights, FPS cua M3 chua tinh buoc phat hien nguoi")
        return fps
    from ultralytics import YOLO

    detector = YOLO(str(args.yolo_weights))
    image = read_image(list_clips(args.full_frames, "test")[0]["frames"][0])
    detector_fps = per_second(lambda: detector.predict(image, classes=[0], device=device, verbose=False))
    return 1 / (1 / fps + 1 / detector_fps)


def save_plots(clips, metrics, out):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, value, ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NonFight", "Fight"], yticklabels=["NonFight", "Fight"],
           xlabel="Du doan", ylabel="Thuc te")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(clips["label"], clips["prob"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["m1", "m2", "m3"])
    parser.add_argument("--ckpt")
    parser.add_argument("--frames", help="Mac dinh data/crops voi m3, data/frames voi m1 va m2")
    parser.add_argument("--full-frames", default="data/frames", help="Frame goc, de do toc do phat hien nguoi cua m3")
    parser.add_argument("--results", default="results")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--yolo-weights", help="Trong so phat hien nguoi, chi dung voi m3")
    args = parser.parse_args()

    out = Path(args.results) / args.model
    out.mkdir(parents=True, exist_ok=True)
    args.ckpt = args.ckpt or out / "best.pt"
    args.frames = args.frames or ("data/crops" if args.model == "m3" else "data/frames")
    device = default_device()

    if args.model == "m2":
        val_clips, test_clips, fps = evaluate_yolo(args, device)
    else:
        val_clips, test_clips, fps = evaluate_torch(args, device)
    if args.model == "m3":
        fps = add_detector_time(fps, args, device)

    threshold = choose_threshold(val_clips["label"], val_clips["prob"])
    metrics = clip_metrics(test_clips["label"], test_clips["prob"], threshold)
    metrics.update(
        model=args.model,
        fps=round(fps, 1),
        size_mb=round(Path(args.ckpt).stat().st_size / 1e6, 1),
        device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        test_clips=len(test_clips),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    test_clips.to_csv(out / "preds_test.csv", index=False)
    errors_table(test_clips, threshold).to_csv(out / "errors.csv", index=False)
    save_plots(test_clips, metrics, out)

    table = comparison_table(args.results)
    table.to_csv(Path(args.results) / "comparison.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
