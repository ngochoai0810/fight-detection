"""Vong huan luyen dung chung cho M1 va M3: dung som theo F1 cap clip tren validation."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import aggregate_clips, clip_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_clips(model, loader, device):
    model.eval()
    rows = []
    for inputs, labels, videos in loader:
        probs = torch.sigmoid(model(inputs.to(device))).cpu().tolist()
        rows += [
            {"video": video, "label": int(label), "prob": prob} for video, label, prob in zip(videos, labels, probs)
        ]
    return aggregate_clips(pd.DataFrame(rows))


def fit(model, train_loader, val_loader, out_dir, epochs, patience, lr, device, weight_decay=1e-4, config=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    history, best_f1, stale = [], -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)

        clips = predict_clips(model, val_loader, device)
        val = clip_metrics(clips["label"], clips["prob"])
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "val_f1": val["f1"], "val_acc": val["accuracy"]})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(history[-1], flush=True)

        if val["f1"] > best_f1:
            best_f1, stale = val["f1"], 0
            torch.save({"state_dict": model.state_dict(), "config": config or {}}, out / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    return history


def train_parser(default_frames, default_out, batch_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=default_frames)
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_training(args, model, dataset_cls, config):
    set_seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "config.json").write_text(json.dumps(vars(args), indent=2))
    train_set = dataset_cls(args.frames, "train", args.image_size, augment=True)
    val_set = dataset_cls(args.frames, "val", args.image_size)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers)
    return fit(model, train_loader, val_loader, args.out, args.epochs, args.patience, args.lr, default_device(), config=config)
