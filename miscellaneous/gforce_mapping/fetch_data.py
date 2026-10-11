import fastf1
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUT = DATA_DIR / "norris_2025_british_gp_fastest_lap.csv"
OUT_POS = DATA_DIR / "norris_2025_british_gp_fastest_lap_pos10hz.csv"

if OUT.exists() and OUT_POS.exists():
    print(f"Already exists: {OUT}")
    print(f"Already exists: {OUT_POS}")
else:
    print("Loading 2025 British Grand Prix race session...")
    session = fastf1.get_session(2025, "British Grand Prix", "R")
    session.load(telemetry=True, weather=False, messages=False)

    norris_laps = session.laps.pick_drivers("NOR")
    fastest = norris_laps.pick_fastest()
    print(f"Fastest lap: {fastest['LapTime']}  (lap {int(fastest['LapNumber'])})")

    tel = fastest.get_telemetry()
    tel = tel[["Time", "Distance", "X", "Y", "Speed", "nGear", "Throttle", "Brake", "DRS"]].copy()
    tel["TimeSeconds"] = tel["Time"].dt.total_seconds()
    tel = tel.drop(columns=["Time"])

    tel.to_csv(OUT, index=False)
    print(f"Saved {len(tel)} telemetry rows -> {OUT}")

    # Raw position data at native 10Hz (not linearly interpolated to 240Hz)
    pos = fastest.get_pos_data()
    pos = pos[["Time", "X", "Y"]].copy()
    pos["TimeSeconds"] = pos["Time"].dt.total_seconds()
    pos = pos.drop(columns=["Time"])

    pos.to_csv(OUT_POS, index=False)
    print(f"Saved {len(pos)} native 10Hz position rows -> {OUT_POS}")
