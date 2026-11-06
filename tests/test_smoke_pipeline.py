"""Chay toan bo quy trinh tren video tong hop, gom tai trong so YOLO. Chay bang: pytest -m smoke"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "src"
WEIGHTS = Path(os.environ.get("FIGHT_WEIGHTS_DIR", "."))
pytestmark = pytest.mark.smoke


def run(script, *args, cwd):
    subprocess.run([sys.executable, str(SRC / script), *map(str, args)], check=True, cwd=cwd)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    from conftest import write_video

    work = tmp_path_factory.mktemp("smoke")
    root = work / "RWF-2000"
    for label in ("Fight", "NonFight"):
        for i in range(10):
            write_video(root / "train" / label / f"{label}s{i}_0.avi", label, n_frames=24, seed=i)
        for i in range(3):
            write_video(root / "val" / label / f"{label}t{i}_0.avi", label, n_frames=24, seed=100 + i)
    run("prepare_data.py", "--root", root, "--num-frames", 8, cwd=work)
    return work


def read_metrics(workspace, model):
    return json.loads((workspace / "results" / model / "metrics.json").read_text())


def test_split_has_all_three_sets(workspace):
    splits = pd.read_csv(workspace / "splits" / "splits.csv")
    assert set(splits["split"]) == {"train", "val", "test"}


def test_eda_writes_tables_and_charts(workspace):
    run("eda.py", "--root", workspace / "RWF-2000", "--out", "results/eda", cwd=workspace)
    assert (workspace / "results" / "eda" / "motion.png").exists()
    by_label = pd.read_csv(workspace / "results" / "eda" / "by_label.csv", index_col=0)
    assert by_label.loc["Fight", "motion"] > by_label.loc["NonFight", "motion"]


def test_m1_cnn_overfits_tiny_dataset(workspace):
    run("train_cnn.py", "--no-pretrained", "--image-size", 64, "--epochs", 8, "--batch-size", 16,
        "--num-workers", 0, "--patience", 8, cwd=workspace)
    run("evaluate.py", "--model", "m1", "--image-size", 64, cwd=workspace)
    assert read_metrics(workspace, "m1")["f1"] >= 0.8


def test_m2_yolo_cls_trains_and_evaluates(workspace):
    run("train_yolo_cls.py", "--weights", WEIGHTS / "yolo11n-cls.pt", "--image-size", 64, "--epochs", 5,
        "--batch-size", 16, "--workers", 0, cwd=workspace)
    run("evaluate.py", "--model", "m2", "--image-size", 64, cwd=workspace)
    assert 0.0 <= read_metrics(workspace, "m2")["f1"] <= 1.0


def test_m3_crop_then_cnn_lstm_overfits_tiny_dataset(workspace):
    run("crop_persons.py", "--weights", WEIGHTS / "yolo11n.pt", "--size", 64, cwd=workspace)
    run("train_cnn_lstm.py", "--no-pretrained", "--image-size", 64, "--epochs", 10, "--batch-size", 4,
        "--num-workers", 0, "--patience", 10, cwd=workspace)
    run("evaluate.py", "--model", "m3", "--image-size", 64, "--yolo-weights", WEIGHTS / "yolo11n.pt", cwd=workspace)
    assert read_metrics(workspace, "m3")["f1"] >= 0.8


def test_evaluate_is_reproducible_from_checkpoint(workspace):
    first = read_metrics(workspace, "m1")
    run("evaluate.py", "--model", "m1", "--image-size", 64, cwd=workspace)
    second = read_metrics(workspace, "m1")
    for key in ("accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "threshold"):
        assert first[key] == second[key], key


def test_comparison_lists_every_model(workspace):
    table = pd.read_csv(workspace / "results" / "comparison.csv")
    assert set(table["model"]) == {"m1", "m2", "m3"}
    assert {"fps", "size_mb", "device"} <= set(table.columns)
