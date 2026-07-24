import fastf1
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

print("Loading 2026 British Grand Prix race session...")
session = fastf1.get_session(2026, "British Grand Prix", "R")
session.load(telemetry=True, laps=True, weather=False, messages=True)

laps = session.laps
results = session.results

# ---------------------------------------------------------------------------
# Full results + lap-by-lap summary (for context / gap chart baseline)
# ---------------------------------------------------------------------------
results_out = DATA_DIR / "R09_British_GP_results.csv"
if not results_out.exists():
    res = results.copy()
    for col in res.select_dtypes(include=["timedelta64[ns]"]).columns:
        res[col] = res[col].astype(str)
    res.to_csv(results_out, index=False)
    print(f"Saved results -> {results_out}")

laps_out = DATA_DIR / "R09_British_GP_laps.csv"
if not laps_out.exists():
    lp = laps.copy()
    for col in lp.select_dtypes(include=["timedelta64[ns]"]).columns:
        lp[col] = lp[col].astype(str)
    lp.to_csv(laps_out, index=False)
    print(f"Saved all laps -> {laps_out}")

# ---------------------------------------------------------------------------
# Highlight 1: Lap 1 launch, LEC vs ANT, lights-out to Turn 1
# ---------------------------------------------------------------------------
lec_lap1 = laps.pick_drivers("LEC").pick_laps(1)
ant_lap1 = laps.pick_drivers("ANT").pick_laps(1)

for name, lap_row in [("LEC", lec_lap1), ("ANT", ant_lap1)]:
    out = DATA_DIR / f"lap1_start_{name}_telemetry.csv"
    if out.exists():
        print(f"Already exists: {out}")
        continue
    lap = lap_row.iloc[0]
    tel = lap.get_telemetry()
    tel = tel[["Time", "Distance", "X", "Y", "Speed", "nGear", "Throttle", "Brake", "DRS"]].copy()
    tel["TimeSeconds"] = tel["Time"].dt.total_seconds()
    tel = tel.drop(columns=["Time"])
    # Turn 1 (Copse... actually first corner at Silverstone is Abbey/Turn1) is early in the lap;
    # keep first ~1200m which covers lights-out through Turn 1/2 (Village/The Loop) exit
    tel = tel[tel["Distance"] <= 1200].copy()
    tel.to_csv(out, index=False)
    print(f"Saved {len(tel)} rows -> {out}")

# ---------------------------------------------------------------------------
# Highlight 2: Gap-to-leader across laps 36-48, Lap 41 turning point
# ---------------------------------------------------------------------------
gap_out = DATA_DIR / "laps36_48_gap_and_pace.csv"
if not gap_out.exists():
    window = laps.pick_laps(range(36, 49)).copy()
    keep_cols = [
        "Driver", "DriverNumber", "LapNumber", "LapTime", "Position",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "TrackStatus", "IsAccurate", "Compound", "TyreLife",
    ]
    window = window[[c for c in keep_cols if c in window.columns]].copy()
    for col in window.select_dtypes(include=["timedelta64[ns]"]).columns:
        window[col] = window[col].astype(str)
    window = window.sort_values(["LapNumber", "Position"])
    window.to_csv(gap_out, index=False)
    print(f"Saved {len(window)} rows -> {gap_out}")

# session-time-based gap to race leader for every driver, laps 36-48
# (built from LapStartTime + accumulated laptime so we can chart the "gap dive")
gap_timeline_out = DATA_DIR / "laps36_48_gap_timeline.csv"
if not gap_timeline_out.exists():
    all_laps = laps.copy()
    window = all_laps[(all_laps["LapNumber"] >= 36) & (all_laps["LapNumber"] <= 48)].copy()
    window["LapStartSeconds"] = window["LapStartTime"].dt.total_seconds()
    window["TimeSeconds"] = window["Time"].dt.total_seconds()  # session time at lap completion
    leader_time_per_lap = window.groupby("LapNumber")["TimeSeconds"].min().rename("LeaderTimeSeconds")
    window = window.merge(leader_time_per_lap, on="LapNumber", how="left")
    window["GapToLeaderSeconds"] = window["TimeSeconds"] - window["LeaderTimeSeconds"]
    out_cols = ["Driver", "DriverNumber", "LapNumber", "Position", "TimeSeconds", "GapToLeaderSeconds"]
    window = window[out_cols].sort_values(["LapNumber", "Position"])
    window.to_csv(gap_timeline_out, index=False)
    print(f"Saved {len(window)} rows -> {gap_timeline_out}")

# ---------------------------------------------------------------------------
# Highlight 3: Leclerc's fastest lap, full telemetry + position for track map
# ---------------------------------------------------------------------------
fastest_out = DATA_DIR / "leclerc_fastest_lap_telemetry.csv"
fastest_pos_out = DATA_DIR / "leclerc_fastest_lap_pos10hz.csv"
if fastest_out.exists() and fastest_pos_out.exists():
    print(f"Already exists: {fastest_out}")
    print(f"Already exists: {fastest_pos_out}")
else:
    lec_laps = laps.pick_drivers("LEC")
    fastest = lec_laps.pick_fastest()
    print(f"Leclerc fastest lap: {fastest['LapTime']} (lap {int(fastest['LapNumber'])})")

    tel = fastest.get_telemetry()
    tel = tel[["Time", "Distance", "X", "Y", "Speed", "nGear", "Throttle", "Brake", "DRS"]].copy()
    tel["TimeSeconds"] = tel["Time"].dt.total_seconds()
    tel = tel.drop(columns=["Time"])
    tel.to_csv(fastest_out, index=False)
    print(f"Saved {len(tel)} telemetry rows -> {fastest_out}")

    pos = fastest.get_pos_data()
    pos = pos[["Time", "X", "Y"]].copy()
    pos["TimeSeconds"] = pos["Time"].dt.total_seconds()
    pos = pos.drop(columns=["Time"])
    pos.to_csv(fastest_pos_out, index=False)
    print(f"Saved {len(pos)} native 10Hz position rows -> {fastest_pos_out}")

# ---------------------------------------------------------------------------
# Highlight 2 (animated version): multi-lap telemetry + position for LEC,
# ANT, HAM across laps 36-48, concatenated on absolute session time so the
# three cars can be animated moving simultaneously around the track.
# ---------------------------------------------------------------------------
for drv in ["LEC", "ANT", "HAM"]:
    tel_out = DATA_DIR / f"laps36_48_{drv}_telemetry.csv"
    pos_out = DATA_DIR / f"laps36_48_{drv}_pos10hz.csv"
    if tel_out.exists() and pos_out.exists():
        print(f"Already exists: {tel_out}")
        print(f"Already exists: {pos_out}")
        continue

    drv_laps = laps.pick_drivers(drv).pick_laps(range(36, 49))
    tel_frames, pos_frames = [], []
    for _, lap in drv_laps.iterrows():
        lap_start_s = lap["LapStartTime"].total_seconds()

        t = lap.get_telemetry()
        t = t[["Time", "Distance", "X", "Y", "Speed", "nGear", "Throttle", "Brake", "DRS"]].copy()
        # get_telemetry() Time is lap-relative; add LapStartTime to get true
        # session-absolute seconds so laps concatenate onto one real timeline
        # (needed to animate LEC/ANT/HAM moving simultaneously).
        t["SessionSeconds"] = lap_start_s + t["Time"].dt.total_seconds()
        t["LapNumber"] = lap["LapNumber"]
        t = t.drop(columns=["Time"])
        tel_frames.append(t)

        p = lap.get_pos_data()
        p = p[["Time", "X", "Y"]].copy()
        p["SessionSeconds"] = lap_start_s + p["Time"].dt.total_seconds()
        p["LapNumber"] = lap["LapNumber"]
        p = p.drop(columns=["Time"])
        pos_frames.append(p)

    tel_all = pd.concat(tel_frames, ignore_index=True)
    tel_all.to_csv(tel_out, index=False)
    print(f"Saved {len(tel_all)} telemetry rows -> {tel_out}")

    pos_all = pd.concat(pos_frames, ignore_index=True)
    pos_all.to_csv(pos_out, index=False)
    print(f"Saved {len(pos_all)} native 10Hz position rows -> {pos_out}")

# ---------------------------------------------------------------------------
# Circuit corner reference (X, Y, distance, angle) for track-map annotations
# ---------------------------------------------------------------------------
corners_out = DATA_DIR / "silverstone_corners.csv"
if not corners_out.exists():
    ci = session.get_circuit_info()
    corners = ci.corners.copy()
    corners.to_csv(corners_out, index=False)
    print(f"Saved {len(corners)} corners -> {corners_out}")

# Named-corner lookup for Silverstone GP circuit, official turn numbers 1-18.
# FastF1 only supplies numbers/X/Y/distance, not names. Verified against
# Formula1.com's official corner-naming reference: Copse = Turn 9, the
# Maggotts/Becketts/Chapel complex = Turns 10-14, Stowe = Turn 15.
# (https://www.formula1.com/en/latest/article/explained-how-every-silverstone-corner-got-its-name)
SILVERSTONE_CORNER_NAMES = {
    1: "Abbey", 2: "Farm Curve", 3: "Village", 4: "The Loop", 5: "Aintree",
    6: "Brooklands", 7: "Luffield", 8: "Woodcote", 9: "Copse", 10: "Maggotts",
    11: "Becketts", 12: "Chapel", 13: "Chapel (2)", 14: "Hangar Straight kink",
    15: "Stowe", 16: "Vale", 17: "Club Corner", 18: "Club",
}
named_out = DATA_DIR / "silverstone_corners_named.csv"
if not named_out.exists():
    ci = session.get_circuit_info()
    corners = ci.corners.copy()
    corners["Name"] = corners["Number"].map(SILVERSTONE_CORNER_NAMES)
    corners.to_csv(named_out, index=False)
    print(f"Saved {len(corners)} named corners -> {named_out}")

# ---------------------------------------------------------------------------
# Race control messages around lap 41 (Antonelli incident) for annotation context
# ---------------------------------------------------------------------------
rc_out = DATA_DIR / "race_control_messages.csv"
if not rc_out.exists():
    rcm = session.race_control_messages.copy()
    for col in rcm.select_dtypes(include=["timedelta64[ns]"]).columns:
        rcm[col] = rcm[col].astype(str)
    rcm.to_csv(rc_out, index=False)
    print(f"Saved {len(rcm)} race control messages -> {rc_out}")

print("Done.")
