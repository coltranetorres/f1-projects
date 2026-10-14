import numpy as np

from race_dashboard.backend.replay import arc_lengths, last_moving_time, unwrap_progress


def test_arc_lengths_closed_square():
    outline = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    cum, L = arc_lengths(outline)
    assert cum.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert L == 40.0


def test_unwrap_progress_handles_grid_offset_and_line_crossing():
    L = 6283.0
    s = np.array([6200.0, 6250.0, 20.0, 70.0])  # starts behind the line, then crosses it
    p = unwrap_progress(s, L)
    np.testing.assert_allclose(p * L, [-83.0, -33.0, 20.0, 70.0], atol=1e-6)


def test_unwrap_progress_never_decreases():
    L = 1000.0
    s = np.array([100.0, 110.0, 105.0, 120.0])  # projection jitter backwards
    p = unwrap_progress(s, L)
    assert np.all(np.diff(p) >= 0)


def test_last_moving_time_finds_stop():
    t = np.arange(0, 10, 1.0)
    x = np.array([0, 100, 200, 300, 300, 300, 300, 300, 300, 300], float)
    y = np.zeros(10)
    assert last_moving_time(t, x, y) == 3.0


def test_last_moving_time_never_moved_returns_first_sample():
    t = np.arange(5, 10, 1.0)
    assert last_moving_time(t, np.zeros(5), np.zeros(5)) == 5.0


def test_first_moving_time_ignores_samples_before_after():
    from race_dashboard.backend.replay import first_moving_time
    t = np.arange(-5.0, 10.0, 1.0)
    x = np.where(t < -2, t * 100.0, -200.0)        # moves before t=0, stops, moves again at t=4
    x = np.where(t >= 4, -200.0 + (t - 3) * 50.0, x)
    y = np.zeros_like(x)
    assert first_moving_time(t, x, y, after=0.0) == 4.0
    assert first_moving_time(t, x, y, after=-10.0) == -4.0


def test_first_moving_time_never_moves_is_inf():
    from race_dashboard.backend.replay import first_moving_time
    t = np.arange(0.0, 5.0)
    assert first_moving_time(t, np.zeros(5), np.zeros(5)) == float("inf")


def test_project_to_outline_is_continuous_between_vertices():
    from race_dashboard.backend.replay import arc_lengths, project_to_outline
    outline = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    cum, L = arc_lengths(outline)
    s, dist = project_to_outline(outline, cum, L, np.array([[50.0, 3.0], [100.0, 40.0], [-30.0, 50.0]]))
    assert np.allclose(s, [50.0, 140.0, 350.0])
    assert np.allclose(dist, [3.0, 0.0, 30.0])
