import pytest

from race_dashboard.backend.insights.facts import candidates_for_lap, movers_candidates
from race_dashboard.backend.tests.insights_fixtures import patch, series, table

POS = {"P": [4, 4, 3, 2, 1, 1], "Q": [1, 1, 2, 3, 3, 3], "R": [2, 2, 1, 1, 2, 2], "S": [3, 3, 4, 4, 4, 4]}


def grid_table(**kw):
    return table(*[series(d, 6, 90.0, positions=p) for d, p in POS.items()],
                 grid={"P": 6, "Q": 1, "R": 2, "S": 3}, **kw)


def test_gainer_and_loser_over_five_laps():
    cs = {c.key: c for c in movers_candidates(grid_table(), 6)}
    assert set(cs) == {"move:P", "move:Q"}  # R is level, S moved only one place
    p, q = cs["move:P"].numbers, cs["move:Q"].numbers
    assert (p["kind"], p["positions"], p["from_pos"], p["to_pos"], p["cause"]) == ("gain", 3, 4, 1, "on_track")
    assert (q["kind"], q["positions"], p["window_laps"], p["from_grid"]) == ("loss", 2, 5, False)
    assert cs["move:P"].score == pytest.approx(0.5)


def test_lap_one_compares_with_the_grid():
    (c,) = movers_candidates(grid_table(), 1)
    n = c.numbers
    assert (n["driver"], n["positions"], n["from_pos"], n["to_pos"], n["from_grid"]) == ("P", 2, 6, 4, True)


def test_pit_stop_in_window_is_the_cause():
    recs = patch(series("P", 6, 90.0, positions=POS["P"]), 3, pit_in=True)
    t = table(recs, *[series(d, 6, 90.0, positions=p) for d, p in POS.items() if d != "P"])
    (c,) = [c for c in movers_candidates(t, 6) if c.key == "move:P"]
    assert c.numbers["cause"] == "pit_cycle"


def test_retired_driver_is_not_a_mover():
    t = table(series("P", 3, 90.0, positions=[1, 1, 1]), series("Q", 6, 90.0, positions=[2, 2, 2, 2, 2, 2]))
    assert all(c.drivers != ("P",) for c in movers_candidates(t, 6))


def test_lap_one_always_has_enough_candidates_to_fill_three():
    assert len(candidates_for_lap(grid_table(), 1)) >= 3
