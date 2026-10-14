import numpy as np

from race_dashboard.backend.replay import Replay


def test_window_is_half_open_and_aligned(replay):
    w = replay.window(10.0, 20.0)
    assert w["t0"] == 10.0 and w["hz"] == 4.0 and w["frames"] == 40
    assert len(w["lap"]) == 40 and len(w["lap_time"]) == 40
    d = w["drivers"]["A"]
    assert all(len(d[k]) == 40 for k in ("x", "y", "pos", "gap", "down", "status"))


def test_adjacent_windows_do_not_overlap(replay):
    a, b = replay.window(0.0, 60.0), replay.window(60.0, 120.0)
    assert a["frames"] + b["frames"] == len(replay.clock)  # duration 110 s -> 440 frames


def test_window_past_end_is_empty_and_before_start_clamps(replay):
    assert replay.window(500.0, 560.0)["frames"] == 0
    w = replay.window(-5.0, 2.0)
    assert w["t0"] == 0.0 and w["frames"] == 8


def test_save_load_roundtrip(replay, tmp_path):
    p = tmp_path / "replay.npz"
    replay.save(p)
    back = Replay.load(p)
    assert back.meta == replay.meta
    np.testing.assert_array_equal(back.pos, replay.pos)
    np.testing.assert_array_equal(back.x, replay.x)
    np.testing.assert_allclose(back.gap, replay.gap)
    assert back.window(0, 5) == replay.window(0, 5)
