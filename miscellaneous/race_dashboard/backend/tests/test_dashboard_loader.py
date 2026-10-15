from types import SimpleNamespace

import pandas as pd
import pytest

from race_dashboard.backend import loader


def test_team_color_passes_valid_string():
    assert loader._team_color("FF8000") == "FF8000"


@pytest.mark.parametrize("bad", [float("nan"), None, ""])
def test_team_color_falls_back(bad):
    assert loader._team_color(bad) == "888888"


def test_seconds_valid_and_missing():
    t0 = pd.Timedelta(seconds=10)
    assert loader._seconds(pd.Timedelta(seconds=25.5), t0) == 15.5
    assert loader._seconds(pd.NaT, t0) is None
    assert loader._seconds(float("nan"), t0) is None


def test_lap_count():
    assert loader._lap_count(pd.Series([1.0, 5.0, 3.0])) == 5
    assert loader._lap_count(pd.Series([float("nan")])) == 0
    assert loader._lap_count(pd.Series([], dtype=float)) == 0


def test_get_replay_wraps_load_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)
    boom = OSError("no network")

    def fail(year, event):
        raise boom

    monkeypatch.setattr(loader, "load_session", fail)
    with pytest.raises(RuntimeError) as ei:
        loader.get_replay(2026, "Singapore")
    assert "2026" in str(ei.value) and "Singapore" in str(ei.value)
    assert ei.value.__cause__ is boom


def test_get_replay_corrupt_cache_still_rebuilds(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path)
    p = tmp_path / f"2026_Singapore_v{loader.REPLAY_VERSION}.npz"
    p.write_bytes(b"garbage")

    def fail(year, event):
        raise OSError("x")

    monkeypatch.setattr(loader, "load_session", fail)
    with pytest.raises(RuntimeError):
        loader.get_replay(2026, "Singapore")
    assert not p.exists()


def _fake_session():
    td = pd.to_timedelta
    laps = pd.DataFrame({
        "Driver": ["VER", "VER", "VER", "NOR"],
        "LapNumber": [1.0, 2.0, 3.0, 1.0],
        "LapTime": [td(95, unit="s"), td(90, unit="s"), pd.NaT, td(96, unit="s")],
        "Time": [td(100, unit="s"), td(190, unit="s"), td(300, unit="s"), td(101.5, unit="s")],
        "Position": [1.0, 1.0, float("nan"), 2.0],
        "Compound": ["SOFT", "SOFT", None, "MEDIUM"],
        "TyreLife": [1.0, 2.0, float("nan"), 1.0],
        "Stint": [1.0, 1.0, 1.0, 1.0],
        "PitInTime": [pd.NaT, pd.NaT, td(295, unit="s"), pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "TrackStatus": ["1", "1", "1", "12"],
        "IsAccurate": [True, True, float("nan"), True],
    })
    results = pd.DataFrame({"Abbreviation": ["VER", "NOR", "PIA"], "GridPosition": [1.0, 2.0, 0.0]})
    status = pd.DataFrame({"Status": ["Started"], "Time": [td(10, unit="s")]})
    return SimpleNamespace(laps=laps, results=results, session_status=status, total_laps=62)


def test_build_lap_table_maps_fastf1_columns():
    t = loader.build_lap_table(_fake_session())
    assert t.total_laps == 62 and t.grid == {"VER": 1, "NOR": 2}   # pit-lane starter (0) has no grid slot
    v2 = t.rec("VER", 2)
    assert (v2.time_s, v2.end_s, v2.position, v2.compound, v2.tyre_life, v2.clean) == (90.0, 180.0, 1, "SOFT", 2, True)
    assert not t.rec("VER", 1).clean                      # lap 1 is never clean
    v3 = t.rec("VER", 3)
    assert v3.time_s is None and v3.position is None and v3.compound is None and v3.tyre_life is None
    assert v3.pit_in and not v3.clean                     # NaN IsAccurate, in-lap: not clean
    assert not t.rec("NOR", 1).clean                      # yellow track status
