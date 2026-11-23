"""Analyze saved predictions and compare policies selected on validation only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import clip_metrics, errors_table, select_operating_point, threshold_sweep


def read_predictions(path):
    clips = pd.read_csv(path)
    if not {"video", "label", "prob"} <= set(clips):
        raise ValueError(f"{path}: required columns are video, label, prob")
    if clips.empty or clips[["video", "label", "prob"]].isna().any().any():
        raise ValueError(f"{path}: predictions must be nonempty and complete")
    clips["label"] = pd.to_numeric(clips["label"], errors="raise")
    clips["prob"] = pd.to_numeric(clips["prob"], errors="raise")
    if not clips["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    if not np.isfinite(clips["prob"]).all() or not clips["prob"].between(0, 1).all():
        raise ValueError(f"{path}: probabilities must be within [0, 1]")
    if clips.duplicated(["video", "label"]).any():
        raise ValueError(f"{path}: duplicate clip predictions")
    clips["label"] = clips["label"].astype(int)
    return clips


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_policies(validation, test, min_recall=.9):
    # Test scores are never used to choose a policy or its operating threshold.
    shared = set(zip(validation.video, validation.label)) & set(zip(test.video, test.label))
    if shared:
        raise ValueError("Validation and test contain overlapping clips")
    sweep = threshold_sweep(validation.label, validation.prob)
    rows = []
    for strategy in ("f2", "f1", "precision_at_recall"):
        point = select_operating_point(sweep, strategy, min_recall)
        row = {"strategy": strategy, "threshold": point["threshold"], "selected_on": "val"}
        row.update({f"val_{key}": value for key, value in point.items() if key != "threshold"})
        row.update({f"test_{key}": value for key, value in clip_metrics(test.label, test.prob, point["threshold"]).items() if key != "threshold"})
        rows.append(row)
    return sweep, pd.DataFrame(rows)


def error_analysis(clips, threshold, metadata=None):
    errors = errors_table(clips, threshold)
    errors["pred"] = (errors.prob >= threshold).astype(int)
    errors["distance_from_threshold"] = (errors.prob - threshold).abs()
    errors = errors.sort_values(["error", "distance_from_threshold"], ascending=[True, False])
    if metadata is not None:
        metadata = metadata.copy()
        metadata["label"] = metadata.label.replace({"Fight": 1, "NonFight": 0}).astype(int)
        if metadata.duplicated(["video", "label"]).any():
            raise ValueError("Duplicate clips in split metadata")
        extras = [key for key in ("video", "label", "relpath", "split", "fps", "width", "height", "duration_s") if key in metadata]
        errors = errors.merge(metadata[extras], on=["video", "label"], how="left", validate="one_to_one")
    errors["review_reason"] = ""
    errors["review_notes"] = ""
    errors["source_id"] = errors.video.astype(str).str.replace(r"_\d+$", "", regex=True)
    summary = clip_metrics(clips.label, clips.prob, threshold)
    summary["fpr"] = summary["fp"] / (summary["fp"] + summary["tn"]) if summary["fp"] + summary["tn"] else None
    for kind in ("FP", "FN"):
        scores = errors.loc[errors.error == kind, "prob"]
        summary[kind.lower() + "_score"] = {
            "count": len(scores), "min": float(scores.min()) if len(scores) else None,
            "median": float(scores.median()) if len(scores) else None,
            "max": float(scores.max()) if len(scores) else None,
        }
    return errors, summary


def save_error_samples(errors, frames_root, out):
    """Optional manual-review grids on Kaggle, never infer semantic error causes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prepare_data import read_image

    for kind in ("FP", "FN"):
        rows = []
        for error in errors[errors.error == kind].itertuples():
            folder = Path(frames_root) / "test" / ("Fight" if error.label else "NonFight") / error.video
            files = sorted(folder.glob("*.jpg"))
            if files:
                rows.append((error, files))
            if len(rows) == 6:
                break
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.5 * len(rows)), squeeze=False)
        for row_index, (error, files) in enumerate(rows):
            for ax, index in zip(axes[row_index], (0, len(files) // 2, len(files) - 1)):
                image = read_image(files[index])
                if image is not None:
                    ax.imshow(image[:, :, ::-1])
                ax.set_title(f"{error.video[:35]} | p={error.prob:.3f} | f{index}", fontsize=8)
                ax.axis("off")
        fig.suptitle(f"{kind}: manual review samples (test set)")
        fig.tight_layout()
        fig.savefig(Path(out) / f"{kind.lower()}_samples.png", dpi=100)
        plt.close(fig)


def analyze_model(model_dir, out, metadata=None, min_recall=.9, frames=None):
    model_dir, out = Path(model_dir), Path(out)
    test_path = model_dir / "preds_test.csv"
    test = read_predictions(test_path)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    errors, summary = error_analysis(test, float(metrics["threshold"]), metadata)
    summary.update(model=model_dir.name, split="test", predictions_sha256=file_hash(test_path))
    out.mkdir(parents=True, exist_ok=True)
    review_path = out / "errors_review.csv"
    if review_path.is_file():
        previous = pd.read_csv(review_path).fillna("")
        review_keys = ["video", "label", "review_reason", "review_notes"]
        if set(review_keys) <= set(previous) and not previous.duplicated(["video", "label"]).any():
            errors = errors.drop(columns=["review_reason", "review_notes"]).merge(previous[review_keys], on=["video", "label"], how="left", validate="one_to_one")
            errors[["review_reason", "review_notes"]] = errors[["review_reason", "review_notes"]].fillna("")
    errors.to_csv(out / "errors_review.csv", index=False)
    groups = test.assign(
        source_id=test.video.astype(str).str.replace(r"_\d+$", "", regex=True),
        fp=(test.label == 0) & (test.prob >= metrics["threshold"]),
        fn=(test.label == 1) & (test.prob < metrics["threshold"]),
    ).groupby(["source_id", "label"]).agg(clips=("video", "size"), fp=("fp", "sum"), fn=("fn", "sum"))
    groups["error_rate"] = (groups.fp + groups.fn) / groups.clips
    groups.sort_values(["error_rate", "clips"], ascending=False).to_csv(out / "error_groups.csv")
    if frames:
        save_error_samples(errors, frames, out)
    (out / "error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    val_path = model_dir / "preds_val.csv"
    if val_path.is_file():
        sweep, policies = compare_policies(read_predictions(val_path), test, min_recall)
        sweep.to_csv(out / "thresholds_val.csv", index=False)
        policies.to_csv(out / "policy_comparison.csv", index=False)
        for row in policies.to_dict("records"):
            target = out / row["strategy"]
            target.mkdir(exist_ok=True)
            policy = dict(row, validation_predictions_sha256=file_hash(val_path), model=model_dir.name, min_recall=min_recall)
            if (model_dir / "best.pt").is_file():
                policy["checkpoint_sha256"] = file_hash(model_dir / "best.pt")
            (target / "metrics.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
        summary["threshold_comparison"] = "ready"
    else:
        summary["threshold_comparison"] = "missing preds_val.csv; run evaluation on Kaggle"
    (out / "analysis_status.json").write_text(json.dumps({"threshold_comparison": summary["threshold_comparison"]}), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--model", choices=("m1", "m2", "m3", "all"), default="all")
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--splits", default="results/splits.csv")
    parser.add_argument("--min-recall", type=float, default=.9)
    parser.add_argument("--frames", help="Optional extracted frame root for manual-review image grids")
    args = parser.parse_args()
    if not 0 <= args.min_recall <= 1:
        parser.error("min-recall must be within [0, 1]")
    metadata = pd.read_csv(args.splits) if Path(args.splits).is_file() else None
    models = ("m1", "m2", "m3") if args.model == "all" else (args.model,)
    summaries = []
    for model in models:
        frame_root = args.frames
        if frame_root and args.model == "all":
            parser.error("Use --model m1/m2/m3 with --frames to select the matching frame/crop root")
        summary = analyze_model(Path(args.results) / model, Path(args.out) / model, metadata, args.min_recall, frame_root)
        summaries.append({k: v for k, v in summary.items() if not isinstance(v, dict)})
        print(f"{model}: FP={summary['fp']}, FN={summary['fn']}; {summary['threshold_comparison']}")
    pd.DataFrame(summaries).to_csv(Path(args.out) / "summary.csv", index=False)


if __name__ == "__main__":
    main()
