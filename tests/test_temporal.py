import pytest

from temporal import TemporalWindow


@pytest.mark.parametrize("fps", [10, 15, 30, 60, 120])
def test_window_covers_training_duration_independently_of_fps(fps):
    window = TemporalWindow(16, 5)
    first_ready = None
    for i in range(fps * 7):
        timestamp = i / fps
        window.add(timestamp, timestamp)
        sequence = window.sample()
        if sequence is not None:
            first_ready = first_ready or timestamp
            assert len(sequence) == 16
            assert sequence == sorted(sequence)
            assert sequence[-1] - sequence[0] >= 4.4
            assert sequence[0] >= timestamp - 5.1
        assert len(window.frames) <= 34
    assert 4.8 <= first_ready <= 5.1


def test_camera_gap_restarts_warmup_and_rejects_backwards_time():
    window = TemporalWindow(16, 5)
    for i in range(61):
        window.add(i / 10, i)
    assert window.sample() is not None
    window.add(20, "after-gap")
    assert window.sample() is None
    assert window.coverage == 0
    assert len(window.frames) == 1
    with pytest.raises(ValueError):
        window.add(19, "older")


def test_window_length_configuration_and_sparse_input():
    window = TemporalWindow(8, 2)
    for timestamp in (0, .5, 1, 1.5, 2):
        window.add(timestamp, timestamp)
    sequence = window.sample()
    assert len(sequence) == 8
    assert min(sequence) >= 0 and max(sequence) <= 2
    with pytest.raises(ValueError):
        TemporalWindow(16, float("nan"))
