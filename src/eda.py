"""EDA cho bo du lieu video. Doc splits.csv tu prepare_data.py, ghi bang va bieu do vao --out.

    python src/eda.py --root D:/RWF-2000 --splits splits/splits.csv --out results/eda
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_data import read_frames, sample_indices

LABEL_COLORS = {"Fight": "#c8553d", "NonFight": "#3d6fc8"}


def frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _motion(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(grays, grays[1:])])) if len(grays) > 1 else 0.0


def motion_score(path):
    """Chenh lech trung binh giua hai frame lien tiep, xap xi do lon chuyen dong cua clip."""
    return _motion(read_frames(path, size=160))


def dhash(frame, hash_size=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def find_duplicates(hashes):
    groups = defaultdict(list)
    for video, value in hashes.items():
        groups[value].append(video)
    return [sorted(videos) for videos in groups.values() if len(videos) > 1]


def summarize_videos(df):
    return df.groupby(["split", "label"]).agg(clips=("duration_s", "size"), duration_mean=("duration_s", "mean"))


def analyze_clips(df, root, n_samples=3, detector=None):
    """Tinh chat luong anh, chuyen dong, dHash va so nguoi cho tung clip."""
    rows = []
    for row in df.itertuples():
        frames = read_frames(Path(root) / row.relpath)
        chosen = [frames[i] for i in sample_indices(len(frames), n_samples)]
        quality = pd.DataFrame([frame_quality(f) for f in chosen]).mean().to_dict()
        record = {
            "video": row.video,
            "label": row.label,
            "split": row.split,
            "motion": _motion([cv2.resize(f, (160, 160)) for f in frames]),
            "dhash": dhash(frames[len(frames) // 2]),
            **quality,
        }
        if detector:
            detections = [detector(f) for f in chosen]
            areas = [
                (b[2] - b[0]) * (b[3] - b[1]) / (f.shape[0] * f.shape[1])
                for boxes, f in zip(detections, chosen)
                for b in boxes
            ]
            record["persons_mean"] = float(np.mean([len(boxes) for boxes in detections]))
            record["no_person_rate"] = float(np.mean([len(boxes) == 0 for boxes in detections]))
            record["person_area_mean"] = float(np.mean(areas)) if areas else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def plot_by_label(clips, column, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.histogram_bin_edges(clips[column], bins=30)
    for label, color in LABEL_COLORS.items():
        ax.hist(clips.loc[clips["label"] == label, column], bins=bins, alpha=0.55, label=label, color=color)
    ax.set(xlabel=column, ylabel="So clip", title=f"Phan bo {column} theo lop")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_samples(df, root, out_path, per_class=4, seed=42):
    fig, axes = plt.subplots(2, per_class, figsize=(per_class * 2.5, 5.2), squeeze=False)
    for axes_row, label in zip(axes, LABEL_COLORS):
        subset = df[df["label"] == label]
        picks = subset.sample(min(per_class, len(subset)), random_state=seed)
        for ax in axes_row:
            ax.axis("off")
        for ax, row in zip(axes_row, picks.itertuples()):
            frames = read_frames(Path(root) / row.relpath)
            ax.imshow(cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB))
            ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eda(root, splits_csv, out_dir, detector=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(splits_csv)

    summary = summarize_videos(df)
    summary.to_csv(out / "summary.csv")
    df[["fps", "width", "height", "n_frames"]].describe().to_csv(out / "video_properties.csv")

    clips = analyze_clips(df, root, detector=detector)
    clips.to_csv(out / "clips.csv", index=False)

    duplicates = find_duplicates(dict(zip(clips["video"] + "/" + clips["label"], clips["dhash"])))
    pd.DataFrame({"videos": [" | ".join(group) for group in duplicates]}).to_csv(out / "duplicates.csv", index=False)

    columns = ["motion", "brightness", "contrast", "sharpness"]
    if detector:
        columns += ["persons_mean", "no_person_rate", "person_area_mean"]
    for column in columns:
        plot_by_label(clips, column, out / f"{column}.png")
    plot_samples(df, root, out / "samples.png")

    by_label = clips.groupby("label")[columns].mean()
    by_label.to_csv(out / "by_label.csv")
    return {"summary": summary, "clips": clips, "by_label": by_label, "duplicates": duplicates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", default="splits/splits.csv")
    parser.add_argument("--out", default="results/eda")
    parser.add_argument("--yolo-weights", help="Vd yolo11n.pt; bo trong de khong thong ke so nguoi")
    args = parser.parse_args()

    detector = None
    if args.yolo_weights:
        from crop_persons import make_yolo_detector

        detector = make_yolo_detector(args.yolo_weights)
    result = run_eda(args.root, args.splits, args.out, detector)
    print(result["summary"])
    print(result["by_label"])
    print(f"So nhom clip trung nhau: {len(result['duplicates'])}")


if __name__ == "__main__":
    main()
