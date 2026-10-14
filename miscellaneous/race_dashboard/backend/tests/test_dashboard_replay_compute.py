import numpy as np
import pytest

from race_dashboard.backend.replay import DNF, FINISHED, RUNNING, compute_replay
from race_dashboard.backend.tests.conftest import make_race


def idx(t, hz=4.0):
    return int(t * hz)


def order(rp, t):
    i = idx(t)
    codes = [d["code"] for d in rp.meta["drivers"]]
    return [c for _, c in sorted(zip(rp.pos[:, i], codes))]


def col(rp, code):
    return [d["code"] for d in rp.meta["drivers"]].index(code)


def test_running_order_by_progress(replay):
    assert order(replay, 10.0) == ["A", "B", "D", "C"]


def test_gap_is_time_since_leader_passed_same_point(replay):
    i, b = idx(10.0), col(replay, "B")
    assert 0.3 < replay.gap[b, i] < 0.7
    assert replay.gap[col(replay, "A"), i] < 0.05


def test_lapped_car_shows_laps_down(replay):
    assert replay.down[col(replay, "C"), idx(95.0)] == 1
    assert replay.down[col(replay, "B"), idx(95.0)] == 0


def test_dnf_freezes_and_ranks_last(replay):
    d = col(replay, "D")
    assert replay.status[d, idx(29.0)] == RUNNING
    assert replay.status[d, idx(60.0)] == DNF
    assert order(replay, 60.0)[-1] == "D"
    assert replay.x[d, idx(60.0)] == replay.x[d, idx(109.0)]


def test_classification_handoff_after_finish(replay):
    assert order(replay, 106.0) == ["A", "B", "C", "D"]
    assert replay.status[col(replay, "A"), idx(106.0)] == FINISHED
    assert replay.status[col(replay, "B"), idx(103.0)] == RUNNING


def test_finished_gap_is_crossing_time_difference(replay):
    i = idx(106.0)
    assert abs(replay.gap[col(replay, "B"), i] - 5.0) < 1e-3
    assert replay.down[col(replay, "C"), i] == 1


def test_lap_never_below_one_with_pole_behind_line(replay):
    assert replay.lap[0] == 1
    assert replay.lap_time[0] >= 0
    assert replay.lap.min() >= 1 and replay.lap.max() <= 5


def test_leader_lap_and_time_freeze_at_winner_finish(replay):
    assert replay.lap[idx(50.0)] == 3  # A is 2.5 laps in
    assert replay.lap[idx(109.0)] == 5
    assert replay.lap_time[idx(105.0)] == replay.lap_time[idx(109.0)]
    assert 19.0 <= replay.lap_time[idx(109.0)] <= 21.0


def test_finished_lapped_car_does_not_jump_ahead_of_running_car(replay):
    assert order(replay, 104.5) == ["A", "B", "C", "D"]


def test_classified_without_finish_time_is_dnf_and_last():
    race = make_race()
    race.drivers[3].classified_pos = 4
    race.drivers[3].finish_t = None
    rp = compute_replay(race)
    di = [x["code"] for x in rp.meta["drivers"]].index("D")
    assert rp.status[di, idx(60.0)] == DNF
    assert order(rp, 60.0)[-1] == "D"


def test_no_usable_drivers_raises():
    race = make_race()
    for d in race.drivers:
        d.t, d.x, d.y = d.t[:1], d.x[:1], d.y[:1]
    with pytest.raises(ValueError, match="no drivers with position data"):
        compute_replay(race)


def test_driver_with_too_few_samples_is_dropped():
    race = make_race()
    race.drivers[2].t = race.drivers[2].t[:1]
    race.drivers[2].x = race.drivers[2].x[:1]
    race.drivers[2].y = race.drivers[2].y[:1]
    rp = compute_replay(race)
    assert [d["code"] for d in rp.meta["drivers"]] == ["A", "B", "D"]


def test_driver_that_never_moves_ends_dnf_last():
    race = make_race()
    d = race.drivers[3]
    d.x = np.full_like(d.x, d.x[0])
    d.y = np.full_like(d.y, d.y[0])
    rp = compute_replay(race)
    di = [x["code"] for x in rp.meta["drivers"]].index("D")
    assert rp.status[di, idx(10.0)] == DNF
    assert order(rp, 10.0)[-1] == "D"


def test_meta_and_dtypes(replay):
    assert replay.meta["total_laps"] == 5
    assert replay.meta["best_lap"] == {"driver": "A", "time_s": 19.5}
    assert len(replay.meta["outline"]) == 200
    assert replay.x.dtype == np.int32 and replay.pos.dtype == np.int8
    assert replay.meta["duration"] == len(replay.clock) / 4.0


def max_rank_jump(rp, t0, t1):
    seg = rp.pos[:, idx(t0):idx(t1)].astype(int)
    return int(np.abs(np.diff(seg, axis=1)).max())


def test_rank_continuous_through_finish(replay):
    assert max_rank_jump(replay, 90.0, 115.0) < 3


def test_rank_continuous_with_several_lapped_cars_crossing_together():
    import copy
    race = make_race()
    c = race.drivers[2]
    for code, lap_s, pos in [("E", 26.2, 4), ("F", 26.4, 5), ("G", 26.6, 6)]:
        e = copy.deepcopy(c)
        e.code, e.number = code, code
        ang = np.arctan2(c.y[0], c.x[0]) + 2 * np.pi * e.t / lap_s
        e.x, e.y = 1000.0 * np.cos(ang), 1000.0 * np.sin(ang)
        e.classified_pos, e.finish_t, e.n_laps = pos, 4 * lap_s, 4
        race.drivers.append(e)
    rp = compute_replay(race)
    assert max_rank_jump(rp, 90.0, 115.0) < 3
    assert order(rp, 111.0) == ["A", "B", "C", "E", "F", "G", "D"]


def test_finished_gap_never_negative():
    race = make_race()
    race.drivers[2].finish_t = 60.0   # classified car that stopped before the winner finished
    rp = compute_replay(race)
    ci = [d["code"] for d in rp.meta["drivers"]].index("C")
    assert rp.gap[ci].min() >= 0.0


def test_pit_lane_starter_ranks_last_until_it_moves(race):
    import copy
    base = race.drivers[0]
    p = copy.deepcopy(base)
    p.code, p.number, p.classified_pos, p.finish_t, p.n_laps = "P", "9", None, None, 0
    ang = np.arctan2(base.y[0], base.x[0]) + 2 * np.pi * base.t / 20.5 + 0.1
    moving = base.t >= 10.0
    p.x = np.where(moving, 1000.0 * np.cos(ang), 1300.0 * np.cos(ang[0]))   # 300 units off the outline
    p.y = np.where(moving, 1000.0 * np.sin(ang), 1300.0 * np.sin(ang[0]))
    race.drivers.append(p)
    rp = compute_replay(race)
    for t in (0.5, 3.0, 9.5):
        assert order(rp, t)[-1] == "P"
    from race_dashboard.backend.replay import PIT_START, RUNNING
    pi = [d["code"] for d in rp.meta["drivers"]].index("P")
    assert (rp.status[pi, :idx(9.5) + 1] == PIT_START).all()
    assert rp.status[pi, idx(11.0)] == RUNNING
    assert not (np.delete(rp.status, pi, axis=0) == PIT_START).any()


def test_fixture_drivers_are_not_pit_starters(replay):
    from race_dashboard.backend.replay import PIT_START
    assert (replay.status != PIT_START).all()
    assert order(replay, 0.25)[0] == "A"


def test_nan_winner_finish_t_does_not_crash():
    race = make_race()
    race.drivers[0].finish_t = float("nan")   # winner A: classified but no valid finish time
    rp = compute_replay(race)
    assert np.isfinite(rp.clock).all()
    a = [d["code"] for d in rp.meta["drivers"]].index("A")
    assert rp.status[a, -1] != FINISHED   # handled as a non-finisher (DNF path), never as the winner
    assert np.isfinite(rp.gap).all()
