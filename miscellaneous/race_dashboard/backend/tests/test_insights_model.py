from dataclasses import replace

import pytest

from race_dashboard.backend.tests.insights_fixtures import patch, series, table


def test_order_and_gap_ahead():
    t = table(series("A", 3, 90.0, pos=1), series("B", 3, 90.0, pos=2, offset=1.5))
    assert t.order(3) == ["A", "B"]
    assert t.gap_ahead("B", 3) == pytest.approx(1.5)
    assert t.gap_ahead("A", 3) is None


def test_retired_driver_drops_out_of_order():
    t = table(series("A", 3, 90.0, pos=1), series("B", 2, 90.0, pos=2))
    assert t.order(3) == ["A"]
    assert t.gap_ahead("B", 3) is None


def test_position_zero_is_the_grid():
    t = table(series("A", 2, 90.0), grid={"A": 4})
    assert t.position("A", 0) == 4 and t.position("A", 1) == 1 and t.position("Z", 1) is None


def test_recent_pace_means_last_three_clean_laps():
    t = table(series("A", 4, 90.0, step=1.0))  # 90, 91, 92, 93
    assert t.recent_pace("A", 4) == pytest.approx(92.0)


def test_recent_pace_needs_two_clean_laps():
    recs = [replace(r, clean=False) if r.lap < 4 else r for r in series("A", 4, 90.0)]
    assert table(recs).recent_pace("A", 4) is None


def test_tyre_and_missing_tyre():
    t = table(series("A", 3, 90.0, compound="SOFT", life0=5))
    assert t.tyre("A", 3) == ("SOFT", 7)
    assert t.tyre("A", 9) is None


def test_stint_laps_stop_at_a_stint_change():
    recs = series("A", 3, 90.0) + [replace(r, lap=r.lap + 3, stint=2) for r in series("A", 3, 90.0)]
    laps = table(recs).stint_laps("A", 6)
    assert [r.lap for r in laps] == [4, 5, 6] and [r.tyre_life for r in laps] == [1, 2, 3]


def test_pit_loss_uses_completed_stops_only():
    x = patch(series("X", 8, 90.0, pos=3), 3, pit_in=True, time_s=100.0, clean=False)
    x = patch(x, 4, pit_out=True, time_s=105.0, clean=False)
    t = table(x)
    assert t.pit_loss(8) == pytest.approx(25.0)
    assert t.pit_loss(3) is None  # the out-lap (4) has not happened yet
