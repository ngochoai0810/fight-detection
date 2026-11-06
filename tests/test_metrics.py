import numpy as np
import pandas as pd
import pytest

from metrics import aggregate_clips, choose_threshold, clip_metrics, comparison_table, errors_table


def test_aggregate_averages_frame_probabilities_per_clip():
    frames = pd.DataFrame(
        {"video": ["a", "a", "b"], "label": [1, 1, 0], "prob": [0.2, 0.8, 0.1]}
    )
    clips = aggregate_clips(frames).set_index("video")
    assert clips.loc["a", "prob"] == pytest.approx(0.5)
    assert clips.loc["b", "label"] == 0
    assert len(clips) == 2


def test_metrics_match_hand_computed_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.6, 0.1])
    m = clip_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_threshold_is_lowered_to_favour_recall():
    # Nguong 0.5 bo sot mot clip Fight; F2 uu tien bat duoc clip do du them mot bao dong gia.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.4, 0.1, 0.1, 0.1])
    threshold = choose_threshold(y, p)
    assert threshold == pytest.approx(0.3)
    assert clip_metrics(y, p, threshold)["recall"] == 1.0
    assert clip_metrics(y, p, threshold)["fp"] == 1


def test_errors_table_lists_only_wrong_clips_with_type():
    clips = pd.DataFrame(
        {"video": ["tp", "fn", "fp", "tn"], "label": [1, 1, 0, 0], "prob": [0.9, 0.2, 0.7, 0.1]}
    )
    errors = errors_table(clips, threshold=0.5).set_index("video")
    assert errors["error"].to_dict() == {"fn": "FN", "fp": "FP"}


def test_comparison_table_has_one_row_per_model_sorted_by_f1(tmp_path):
    for name, f1 in (("m1", 0.7), ("m3", 0.9)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(f'{{"model": "{name}", "f1": {f1}}}')
    table = comparison_table(tmp_path)
    assert table["model"].tolist() == ["m3", "m1"]
