import pytest

from race_dashboard.backend.insights.facts import battle_candidates
from race_dashboard.backend.tests.insights_fixtures import series, table


def test_close_pair_with_closing_rate():
    t = table(series("A", 4, 90.0, pos=1), series("B", 4, 89.7, pos=2, offset=2.0),
              series("C", 4, 90.0, pos=3, offset=10.0))
    (c,) = battle_candidates(t, 4)  # B-C is ~9 s apart: not a battle
    assert c.key == "battle:A-B" and c.drivers == ("A", "B")
    n = c.numbers
    assert n["gap_s"] == pytest.approx(0.8) and n["closing_s_per_lap"] == pytest.approx(0.3)
    assert n["laps_to_close"] == 3 and n["ahead_tyre_age"] == 4 and n["behind_tyre_age"] == 4
    assert c.score == pytest.approx(0.648, abs=1e-3)


def test_gap_only_battle_without_pace_history():
    t = table(series("A", 1, 90.0, pos=1), series("B", 1, 90.0, pos=2, offset=1.0))
    (c,) = battle_candidates(t, 1)
    assert "closing_s_per_lap" not in c.numbers and c.score == pytest.approx(0.6 * (1 - 1.0 / 2.5))


def test_pulling_away_has_no_catch_projection():
    t = table(series("A", 4, 89.7, pos=1), series("B", 4, 90.0, pos=2, offset=1.0))
    (c,) = battle_candidates(t, 4)
    assert c.numbers["closing_s_per_lap"] < 0 and "laps_to_close" not in c.numbers


def test_no_battles_when_gaps_are_large():
    t = table(series("A", 3, 90.0, pos=1), series("B", 3, 90.0, pos=2, offset=8.0))
    assert battle_candidates(t, 3) == []


def test_no_close_projection_beyond_ten_laps():
    t = table(series("A", 6, 90.0, pos=1), series("B", 6, 89.95, pos=2, offset=1.8), total_laps=60)
    (c,) = battle_candidates(t, 6)
    assert c.numbers["closing_s_per_lap"] == pytest.approx(0.05) and "laps_to_close" not in c.numbers
