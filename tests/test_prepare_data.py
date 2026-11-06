import cv2
import pandas as pd
import pytest

from prepare_data import (
    check_no_leak,
    extract_frames,
    make_splits,
    probe_video,
    sample_indices,
    scan_dataset,
    source_id,
)


def fake_index(n_train_per_class=80, n_test_per_class=20):
    rows = []
    for label in ("Fight", "NonFight"):
        for i in range(n_train_per_class):
            rows.append({"video": f"{label}{i}_0", "label": label, "source_split": "train"})
        for i in range(n_test_per_class):
            rows.append({"video": f"{label}t{i}_0", "label": label, "source_split": "val"})
    return pd.DataFrame(rows)


def test_scan_finds_every_clip_with_label_and_source_split(rwf_root):
    df = scan_dataset(rwf_root)
    assert len(df) == 20
    assert df.groupby(["source_split", "label"]).size().to_dict() == {
        ("train", "Fight"): 8,
        ("train", "NonFight"): 8,
        ("val", "Fight"): 2,
        ("val", "NonFight"): 2,
    }


def test_scan_rejects_folder_without_rwf_layout(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path)


def test_source_id_strips_clip_index():
    assert source_id("abc_DEF_12") == "abc_DEF"
    assert source_id("noindex") == "noindex"


def test_splits_are_70_10_20_and_stratified():
    out = make_splits(fake_index(), seed=1)
    assert out["split"].value_counts().to_dict() == {"train": 140, "test": 40, "val": 20}
    for split in ("train", "val", "test"):
        counts = out[out["split"] == split]["label"].value_counts()
        assert counts["Fight"] == counts["NonFight"]


def test_official_val_folder_becomes_test_untouched():
    out = make_splits(fake_index(), seed=1)
    assert set(out[out["split"] == "test"]["source_split"]) == {"val"}
    assert set(out[out["split"] != "test"]["source_split"]) == {"train"}


def test_clips_from_same_source_stay_in_one_split():
    rows = [
        {"video": f"{label}{s}_{c}", "label": label, "source_split": "train"}
        for label in ("Fight", "NonFight")
        for s in range(40)
        for c in range(2)
    ]
    rows += [{"video": f"t{i}_0", "label": "Fight", "source_split": "val"} for i in range(20)]
    out = make_splits(pd.DataFrame(rows), seed=3)
    assert out.groupby(out["video"].map(source_id))["split"].nunique().max() == 1
    assert (out["split"] == "val").sum() > 0


def test_splits_are_reproducible_with_same_seed():
    first = make_splits(fake_index(), seed=7)
    second = make_splits(fake_index(), seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_check_no_leak_rejects_video_in_two_splits():
    df = pd.DataFrame(
        [
            {"video": "a_0", "label": "Fight", "split": "train"},
            {"video": "a_0", "label": "Fight", "split": "test"},
        ]
    )
    with pytest.raises(ValueError, match="a_0"):
        check_no_leak(df)


def test_check_no_leak_rejects_source_shared_by_train_and_val():
    df = pd.DataFrame(
        [
            {"video": "a_0", "label": "Fight", "split": "train"},
            {"video": "a_1", "label": "Fight", "split": "val"},
        ]
    )
    with pytest.raises(ValueError, match="a"):
        check_no_leak(df)


def test_check_no_leak_only_warns_for_source_shared_with_official_test():
    df = pd.DataFrame(
        [
            {"video": "a_0", "label": "Fight", "split": "train"},
            {"video": "a_1", "label": "Fight", "split": "test"},
        ]
    )
    assert check_no_leak(df) == ["a"]


def test_sample_indices_are_uniform_and_in_range():
    assert sample_indices(160, 16).tolist() == [5 + 10 * i for i in range(16)]
    assert sample_indices(4, 8).min() >= 0 and sample_indices(4, 8).max() <= 3
    assert len(sample_indices(4, 8)) == 8


def test_probe_reads_video_metadata(make_video, tmp_path):
    path = tmp_path / "v.avi"
    make_video(path, "Fight", n_frames=20, size=(96, 72), fps=10)
    info = probe_video(path)
    assert info["readable"] is True
    assert (info["width"], info["height"]) == (96, 72)
    assert info["n_frames"] == 20
    assert info["duration_s"] == pytest.approx(2.0)


def test_probe_marks_corrupt_file_unreadable(tmp_path):
    path = tmp_path / "bad.avi"
    path.write_bytes(b"not a video")
    assert probe_video(path)["readable"] is False


def test_extract_frames_saves_requested_count_and_size(make_video, tmp_path):
    video = tmp_path / "v.avi"
    make_video(video, "Fight", n_frames=20)
    out_dir = tmp_path / "frames"
    assert extract_frames(video, out_dir, num_frames=8, size=32) == 8
    files = sorted(out_dir.glob("*.jpg"))
    assert len(files) == 8
    assert cv2.imread(str(files[0])).shape == (32, 32, 3)


def test_extract_frames_writes_to_non_ascii_folder(make_video, tmp_path):
    video = tmp_path / "v.avi"
    make_video(video, "Fight", n_frames=20)
    out_dir = tmp_path / "chuyên ngành"
    assert extract_frames(video, out_dir, num_frames=4, size=32) == 4
    assert len(list(out_dir.glob("*.jpg"))) == 4
