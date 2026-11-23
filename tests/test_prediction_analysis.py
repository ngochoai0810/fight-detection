import json

import pandas as pd
import pytest

from metrics import choose_threshold, threshold_sweep, select_operating_point
from prediction_analysis import analyze_model, compare_policies, error_analysis, read_predictions


def predictions(prefix, labels, scores):
    return pd.DataFrame({"video": [f"{prefix}{i}" for i in range(len(labels))], "label": labels, "prob": scores})


def test_validation_policies_trade_precision_and_recall_without_using_test():
    val = predictions("v", [1, 1, 1, 1, 0, 0, 0, 0], [.9, .8, .7, .3, .4, .1, .1, .1])
    test = predictions("t", [1, 0], [.2, .8])
    sweep, policies = compare_policies(val, test, min_recall=.75)
    selected = policies.set_index("strategy")
    assert selected.loc["f2", "threshold"] == choose_threshold(val.label, val.prob)
    assert selected.loc["f2", "threshold"] == .3
    assert selected.loc["precision_at_recall", "threshold"] == .7
    changed_test = test.assign(prob=[.99, .01])
    _, changed = compare_policies(val, changed_test, min_recall=.75)
    assert policies.threshold.tolist() == changed.threshold.tolist()
    assert not policies.test_f1.equals(changed.test_f1)
    with pytest.raises(ValueError, match="overlapping"):
        compare_policies(val, val)
    with pytest.raises(ValueError):
        threshold_sweep([1, 1], [.2, .3])
    with pytest.raises(ValueError):
        select_operating_point(sweep, "precision_at_recall", 1.1)


def test_error_review_contains_only_errors_and_joins_source_metadata():
    clips = predictions("clip", [1, 1, 0, 0], [.9, .2, .8, .1])
    metadata = pd.DataFrame({"video": clips.video, "label": ["Fight", "Fight", "NonFight", "NonFight"], "relpath": [f"source/{i}.avi" for i in range(4)]})
    errors, summary = error_analysis(clips, .5, metadata)
    assert set(errors.error) == {"FP", "FN"}
    assert set(errors.relpath) == {"source/1.avi", "source/2.avi"}
    assert summary["fp"] == summary["fn"] == 1
    assert summary["fpr"] == .5
    assert summary["fp_score"]["median"] == .8
    assert errors.review_reason.tolist() == ["", ""]


def test_missing_validation_never_selects_on_test_and_keeps_human_notes(tmp_path):
    model = tmp_path / "m1"
    model.mkdir()
    clips = predictions("test", [1, 0], [.2, .8])
    clips.to_csv(model / "preds_test.csv", index=False)
    (model / "metrics.json").write_text(json.dumps({"threshold": .5}))
    original = (model / "metrics.json").read_bytes()
    out = tmp_path / "analysis"
    summary = analyze_model(model, out)
    assert "missing preds_val" in summary["threshold_comparison"]
    assert not (out / "policy_comparison.csv").exists()
    review = pd.read_csv(out / "errors_review.csv").fillna("")
    review.loc[0, "review_notes"] = "Checked manually"
    review.to_csv(out / "errors_review.csv", index=False)
    analyze_model(model, out)
    assert pd.read_csv(out / "errors_review.csv").iloc[0].review_notes == "Checked manually"
    predictions("val", [1, 0], [.7, .3]).to_csv(model / "preds_val.csv", index=False)
    analyze_model(model, out)
    policy = json.loads((out / "f1/metrics.json").read_text())
    assert policy["selected_on"] == "val"
    assert policy["threshold"] == .7
    assert len(pd.read_csv(out / "policy_comparison.csv")) == 3
    assert (model / "metrics.json").read_bytes() == original


def test_prediction_input_rejects_invalid_scores_and_duplicate_clips(tmp_path):
    path = tmp_path / "preds.csv"
    for clips in (predictions("v", [1, 0], [float("nan"), .2]), predictions("v", [1, 0], [1.1, .2]), pd.DataFrame({"video": ["x", "x"], "label": [1, 1], "prob": [.2, .8]})):
        clips.to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_predictions(path)
