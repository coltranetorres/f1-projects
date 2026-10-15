import pytest

from race_dashboard.backend.insights.facts import pit_candidates
from race_dashboard.backend.tests.insights_fixtures import patch, series, table


def test_stop_this_lap():
    a = patch(series("A", 6, 90.0, pos=1), 4, pit_in=True, position=3)
    t = table(a, series("B", 6, 90.0, pos=2, offset=5.0))
    (c,) = pit_candidates(t, 4)
    assert c.key == "stop:A:4" and c.drivers == ("A",)
    n = c.numbers
    assert (n["old_compound"], n["tyre_age"], n["pos_before"], n["pos_after"]) == ("MEDIUM", 4, 1, 3)
    assert c.score == pytest.approx(0.9)
    assert pit_candidates(t, 5) == []  # only the in-lap itself is a stop fact


def test_undercut_window_needs_a_known_pit_loss():
    x = patch(series("X", 8, 90.0, pos=3, offset=30.0), 3, pit_in=True, time_s=100.0, clean=False)
    x = patch(x, 4, pit_out=True, time_s=105.0, clean=False)
    pair = [series("A", 6, 90.0, pos=1), series("B", 6, 90.0, pos=2, offset=1.5, life0=14)]
    (c,) = pit_candidates(table(*pair, x, total_laps=20), 6)
    assert c.key == "undercut:A-B" and c.drivers == ("A", "B")
    n = c.numbers
    assert n["pit_loss_s"] == pytest.approx(25.0) and n["gap_s"] == pytest.approx(1.5)
    assert (n["behind_tyre_age"], n["ahead_tyre_age"]) == (19, 6)
    assert c.score == pytest.approx(0.63)
    assert pit_candidates(table(*pair), 6) == []  # nobody has stopped: pit loss unknown


def test_no_undercut_candidates_when_few_laps_remain():
    x = patch(series("X", 8, 90.0, pos=3, offset=30.0), 3, pit_in=True, time_s=100.0, clean=False)
    x = patch(x, 4, pit_out=True, time_s=105.0, clean=False)
    pair = [series("A", 6, 90.0, pos=1), series("B", 6, 90.0, pos=2, offset=1.5, life0=14)]
    assert pit_candidates(table(*pair, x, total_laps=8), 6) == []
