import numpy as np
import pytest

from race_dashboard.backend.replay import DriverInput, RaceInput, compute_replay

R = 1000.0
N_OUTLINE = 200
TOTAL_LAPS = 5


def make_race() -> RaceInput:
    """Circular track; A leads, B follows, C is lapped once, D retires at t=30."""
    th = 2 * np.pi * np.arange(N_OUTLINE) / N_OUTLINE
    outline = np.column_stack([R * np.cos(th), R * np.sin(th)])
    t = np.arange(0.0, 120.0, 0.25)

    def car(code, lap_s, slot, stop_at=None, classified=None, finish_t=None, n_laps=0):
        tt = t if stop_at is None else np.minimum(t, stop_at)
        ang = -0.01 * (slot + 1) + 2 * np.pi * tt / lap_s  # grid slots are behind the line
        return DriverInput(code, str(slot + 1), f"Team {code}", "FF0000",
                           t.copy(), R * np.cos(ang), R * np.sin(ang), classified, finish_t, n_laps)

    drivers = [
        car("A", 20.0, 0, classified=1, finish_t=100.0, n_laps=5),
        car("B", 21.0, 1, classified=2, finish_t=105.0, n_laps=5),
        car("C", 26.0, 2, classified=3, finish_t=104.0, n_laps=4),
        car("D", 22.0, 3, stop_at=30.0, classified=None, finish_t=22.0, n_laps=1),
    ]
    return RaceInput("Test GP", drivers, outline, TOTAL_LAPS, ("A", 19.5))


@pytest.fixture
def race():
    return make_race()


@pytest.fixture
def replay(race):
    return compute_replay(race)
