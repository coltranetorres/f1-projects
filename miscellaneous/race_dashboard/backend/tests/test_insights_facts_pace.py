import pytest

from race_dashboard.backend.insights.facts import pace_candidates
from race_dashboard.backend.tests.insights_fixtures import series, table


def by_kind(cands, kind):
    return [c for c in cands if c.numbers["kind"] == kind]


def test_fastest_driver_closing_on_the_car_ahead():
    t = table(series("A", 6, 90.0, pos=1), series("B", 6, 89.5, pos=2, offset=3.8))
    (c,) = by_kind(pace_candidates(t, 6), "fastest")
    n = c.numbers
    assert (n["driver"], n["next"], n["ahead"]) == ("B", "A", "A")
    assert n["margin_s"] == pytest.approx(0.5) and n["gap_s"] == pytest.approx(0.8)
    assert n["laps_to_catch"] == 2 and n["catch_lap"] == 8
    assert c.score == pytest.approx(0.825)


def test_fastest_leader_extends_the_lead():
    t = table(series("A", 6, 89.5, pos=1), series("B", 6, 90.0, pos=2, offset=3.0), total_laps=20)
    (c,) = by_kind(pace_candidates(t, 6), "fastest")
    n = c.numbers
    assert n["driver"] == "A" and n["leader"] is True
    assert n["lead_s"] == pytest.approx(6.0) and n["lead_in_5_laps_s"] == pytest.approx(8.5)
    (lead,) = by_kind(pace_candidates(t, 6), "leader")
    assert lead.numbers["gap_s"] == pytest.approx(6.0) and lead.score == 0.05


def test_no_projection_past_the_final_lap():
    t = table(series("A", 6, 89.5, pos=1), series("B", 6, 90.0, pos=2, offset=3.0), total_laps=10)
    (c,) = by_kind(pace_candidates(t, 6), "fastest")
    assert "lead_in_5_laps_s" not in c.numbers


def test_needs_two_drivers_with_pace():
    assert by_kind(pace_candidates(table(series("A", 6, 90.0)), 6), "fastest") == []


def test_no_catch_projection_beyond_ten_laps():
    t = table(series("A", 6, 90.0, pos=1), series("B", 6, 89.9, pos=2, offset=3.6), total_laps=60)
    (c,) = by_kind(pace_candidates(t, 6), "fastest")
    assert c.numbers["driver"] == "B" and "laps_to_catch" not in c.numbers and "catch_lap" not in c.numbers
