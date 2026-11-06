import numpy as np

from eda import dhash, find_duplicates, frame_quality, motion_score, summarize_videos


def test_quality_reports_brighter_frame_as_brighter():
    dark = np.full((32, 32, 3), 20, dtype=np.uint8)
    bright = np.full((32, 32, 3), 200, dtype=np.uint8)
    assert frame_quality(bright)["brightness"] > frame_quality(dark)["brightness"]


def test_quality_reports_sharp_frame_above_flat_frame():
    flat = np.full((32, 32, 3), 100, dtype=np.uint8)
    sharp = np.indices((32, 32)).sum(axis=0) % 2 * 255
    sharp = np.stack([sharp] * 3, axis=-1).astype(np.uint8)
    assert frame_quality(sharp)["sharpness"] > frame_quality(flat)["sharpness"]


def test_motion_score_higher_for_fast_synthetic_clip(make_video, tmp_path):
    fast, slow = tmp_path / "fast.avi", tmp_path / "slow.avi"
    make_video(fast, "Fight")
    make_video(slow, "NonFight")
    assert motion_score(fast) > motion_score(slow)


def test_dhash_identical_for_same_image_and_differs_for_other():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert dhash(a) == dhash(a.copy())
    assert dhash(a) != dhash(b)


def test_find_duplicates_groups_clips_with_same_hash():
    hashes = {"a": 1, "b": 1, "c": 2}
    assert find_duplicates(hashes) == [["a", "b"]]


def test_summarize_counts_clips_per_split_and_label():
    import pandas as pd

    df = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "label": ["Fight", "NonFight", "Fight"],
            "duration_s": [5.0, 5.0, 4.0],
        }
    )
    summary = summarize_videos(df)
    assert summary.loc[("train", "Fight"), "clips"] == 1
    assert summary.loc[("test", "Fight"), "duration_mean"] == 4.0
