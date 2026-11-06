"""Chi so cap clip, dung chung cho ca ba mo hinh. Lop duong (1) la Fight."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score


def aggregate_clips(frames):
    """Gop xac suat tung frame thanh mot xac suat cho moi clip."""
    return frames.groupby(["video", "label"], as_index=False)["prob"].mean()


def clip_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def choose_threshold(y, p):
    """Chon nguong tot nhat theo F2 de uu tien Recall. Chi goi tren tap validation."""
    candidates = np.unique(p)
    scores = [fbeta_score(y, p >= t, beta=2, zero_division=0) for t in candidates]
    # Khi hoa diem, lay nguong cao nhat de bot bao dong gia.
    return float(candidates[len(scores) - 1 - int(np.argmax(scores[::-1]))])


def errors_table(clips, threshold):
    pred = (clips["prob"] >= threshold).astype(int)
    wrong = clips[pred != clips["label"]].copy()
    wrong["error"] = np.where(wrong["label"] == 1, "FN", "FP")
    return wrong.sort_values("prob", ascending=False)


def comparison_table(results_dir):
    rows = [json.loads(p.read_text()) for p in Path(results_dir).glob("*/metrics.json")]
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
