from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from race_dashboard.backend.insights.model import LapRecord, LapTable
from race_dashboard.backend.replay import DriverInput, RaceInput, Replay, compute_replay

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
REPLAY_VERSION = 4  # bump when compute_replay output changes


def load_session(year: int = 2026, event: str = "Singapore"):
    (CACHE_DIR / "fastf1").mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR / "fastf1"))
    session = fastf1.get_session(year, event, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def _team_color(value) -> str:
    return value if isinstance(value, str) and value else "888888"


def _seconds(td, t0):
    """Seconds from t0 to td, or None when td is NaT/NaN."""
    if td is None or pd.isna(td):
        return None
    v = float((td - t0).total_seconds())
    return v if np.isfinite(v) else None


def _lap_count(series) -> int:
    m = series.max() if len(series) else float("nan")
    return 0 if pd.isna(m) else int(m)


def build_race_input(session) -> RaceInput:
    status = session.session_status
    t0 = status.loc[status["Status"] == "Started", "Time"].iloc[0]
    laps = session.laps
    outline = laps.pick_fastest().get_pos_data()[["X", "Y"]].to_numpy(float)[::2]

    drivers = []
    for _, r in session.results.iterrows():
        number = str(r["DriverNumber"])
        p = session.pos_data.get(number)
        if p is None or len(p) < 2:
            continue
        dl = laps.pick_drivers(number)
        cp = str(r["ClassifiedPosition"])
        drivers.append(DriverInput(
            code=r["Abbreviation"], number=number, team=r["TeamName"],
            color=_team_color(r["TeamColor"]),
            t=(p["SessionTime"] - t0).dt.total_seconds().to_numpy(),
            x=p["X"].to_numpy(float), y=p["Y"].to_numpy(float),
            classified_pos=int(float(cp)) if cp.replace(".", "", 1).isdigit() else None,
            finish_t=_seconds(dl["Time"].max(), t0) if len(dl) else None,
            n_laps=_lap_count(dl["LapNumber"]),
        ))
    best = laps.loc[laps["LapTime"].idxmin()]
    return RaceInput(session.event["EventName"], drivers, outline, int(session.total_laps),
                     (best["Driver"], float(best["LapTime"].total_seconds())))


def get_replay(year: int = 2026, event: str = "Singapore") -> Replay:
    path = CACHE_DIR / f"{year}_{event}_v{REPLAY_VERSION}.npz"
    if path.exists():
        try:
            return Replay.load(path)
        except Exception:
            path.unlink(missing_ok=True)  # corrupt cache: rebuild
    try:
        replay = compute_replay(build_race_input(load_session(year, event)))
    except Exception as e:
        raise RuntimeError(
            f"Could not load the {year} {event} race session from FastF1 (network or cache?): {e}") from e
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    replay.save(path)
    return replay


if __name__ == "__main__":
    rp = get_replay()
    m = rp.meta
    print(f"{m['event']}: {m['total_laps']} laps, {m['duration']:.0f}s, {len(m['drivers'])} drivers, "
          f"best lap {m['best_lap']}")
    last = len(rp.clock) - 1
    final = sorted(zip(rp.pos[:, last], [d["code"] for d in m["drivers"]], rp.status[:, last], rp.gap[:, last]))
    for p, code, st, gap in final:
        print(p, code, ["RUN", "FIN", "DNF", "PIT"][st], f"+{gap:.3f}")


def _num(v):
    return None if v is None or pd.isna(v) else v


def build_lap_table(session) -> LapTable:
    """Per-driver per-lap records for the insights pipeline. NaN/NaT become None, never 0."""
    status = session.session_status
    t0 = status.loc[status["Status"] == "Started", "Time"].iloc[0]
    records = []
    for r in session.laps.itertuples(index=False):
        lap = int(r.LapNumber)
        pit_in, pit_out = not pd.isna(r.PitInTime), not pd.isna(r.PitOutTime)
        accurate = (not pd.isna(r.IsAccurate)) and bool(r.IsAccurate)
        pos, life, stint = _num(r.Position), _num(r.TyreLife), _num(r.Stint)
        records.append(LapRecord(
            driver=r.Driver, lap=lap,
            time_s=None if pd.isna(r.LapTime) else float(r.LapTime.total_seconds()),
            end_s=None if pd.isna(r.Time) else float((r.Time - t0).total_seconds()),
            position=None if pos is None else int(pos),
            compound=_num(r.Compound),
            tyre_life=None if life is None else int(life),
            stint=None if stint is None else int(stint),
            pit_in=pit_in, pit_out=pit_out,
            clean=accurate and not pit_in and not pit_out and str(r.TrackStatus) == "1" and lap > 1,
        ))
    grid = {}
    for _, row in session.results.iterrows():
        g = _num(row["GridPosition"])
        if g is not None and int(g) > 0:
            grid[row["Abbreviation"]] = int(g)
    return LapTable(records, int(session.total_laps), grid)
