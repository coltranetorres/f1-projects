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
