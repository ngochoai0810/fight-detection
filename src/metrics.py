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


def threshold_sweep(y, p):
    """All distinct operating points. Call on validation to select a threshold."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p) or set(y.tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both classes in validation")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    rows = []
    for threshold in np.unique(np.r_[0., .5, 1., p]):
        row = clip_metrics(y, p, threshold)
        precision, recall = row["precision"], row["recall"]
        row["f2"] = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.
        row["fpr"] = row["fp"] / (row["fp"] + row["tn"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_operating_point(sweep, strategy="f2", min_recall=.9):
    if strategy not in ("f2", "f1", "precision_at_recall"):
        raise ValueError("Unknown threshold strategy")
    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be within [0, 1]")
    candidates = sweep
    columns = [strategy, "threshold"]
    if strategy == "precision_at_recall":
        candidates = sweep[sweep["recall"] >= min_recall]
        columns = ["precision", "recall", "threshold"]
    if candidates.empty:
        raise ValueError("No validation threshold satisfies the requested recall")
    return candidates.sort_values(columns, ascending=False).iloc[0].to_dict()


def errors_table(clips, threshold):
    pred = (clips["prob"] >= threshold).astype(int)
    wrong = clips[pred != clips["label"]].copy()
    wrong["error"] = np.where(wrong["label"] == 1, "FN", "FP")
    return wrong.sort_values("prob", ascending=False)


def comparison_table(results_dir):
    rows = [json.loads(p.read_text()) for p in Path(results_dir).glob("*/metrics.json")]
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
