import pytest

from race_dashboard.backend.insights.facts import tyres_candidates
from race_dashboard.backend.tests.insights_fixtures import series, table


def four(d1_step=0.0, d4_life0=1):
    return table(series("D1", 6, 90.0, step=d1_step, pos=1), series("D2", 6, 90.0, pos=2),
                 series("D3", 6, 90.0, pos=3), series("D4", 6, 90.0, pos=4, life0=d4_life0))


def test_degrading_stint_is_flagged():
    cs = tyres_candidates(four(d1_step=0.1), 6)
    (c,) = [c for c in cs if c.key == "deg:D1"]
    assert c.numbers["deg_s_per_lap"] == pytest.approx(0.1) and c.numbers["tyre_age"] == 6
    assert c.score == pytest.approx(0.333, abs=1e-3)
    assert not [c for c in cs if c.numbers["kind"] == "age"]


def test_much_older_tyres_are_flagged():
    cs = tyres_candidates(four(d4_life0=10), 6)
    (c,) = [c for c in cs if c.key == "age:D4"]
    assert c.numbers["tyre_age"] == 15 and c.numbers["median_age"] == 6.0
    assert c.score == pytest.approx(0.6)


def test_nothing_notable_returns_nothing():
    assert tyres_candidates(four(), 6) == []


def test_too_few_drivers_with_tyre_data():
    assert tyres_candidates(table(series("A", 6, 90.0, step=0.2)), 6) == []
