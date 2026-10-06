import fastf1
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

cache_dir = Path(__file__).parent / "cache"
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

SPRINT_ROUNDS = {2, 4, 5, 9}

schedule = fastf1.get_event_schedule(2026, include_testing=False)
print("2026 Schedule:")
print(schedule[["RoundNumber", "EventName", "EventDate"]].to_string())

hungarian_idx = schedule[
    schedule["EventName"].str.contains("Hungarian", case=False)
].index
if len(hungarian_idx) == 0:
    hungarian_idx = schedule[schedule["RoundNumber"] == 11].index
hungarian_round = schedule.loc[hungarian_idx[0], "RoundNumber"]
print(f"\nHungarian GP is Round {hungarian_round}")

pre_hungarian = schedule[schedule["RoundNumber"] < hungarian_round]
print(f"\nFetching data for {len(pre_hungarian)} races before Hungarian:")
print(pre_hungarian[["RoundNumber", "EventName"]].to_string())


def fetch_and_save(round_num, event_name_raw, session_type):
    """Fetch one session and save results + laps CSVs. Skips if both already exist."""
    event_name = event_name_raw.replace(" ", "_").replace("/", "-")
    prefix = f"R{round_num:02d}_{event_name}"

    if session_type == "sprint":
        results_path = DATA_DIR / f"{prefix}_sprint_results.csv"
        laps_path    = DATA_DIR / f"{prefix}_sprint_laps.csv"
        label        = "sprint"
    else:
        results_path = DATA_DIR / f"{prefix}_results.csv"
        laps_path    = DATA_DIR / f"{prefix}_laps.csv"
        label        = "race"

    if results_path.exists() and laps_path.exists():
        print(f"  Skipping {label} (already fetched)")
        return

    session = fastf1.get_session(2026, round_num, "S" if session_type == "sprint" else "R")
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results
    if results is not None and len(results) > 0:
        rc = results.copy()
        for col in rc.columns:
            if str(rc[col].dtype) == "timedelta64[ns]":
                rc[col] = rc[col].astype(str)
        rc.to_csv(results_path, index=False, encoding="utf-8")
        print(f"  Saved {label} results ({len(results)} rows) -> {results_path.name}")

    laps = session.laps
    if laps is not None and len(laps) > 0:
        lc = laps.copy()
        for col in lc.columns:
            if str(lc[col].dtype) == "timedelta64[ns]":
                lc[col] = lc[col].astype(str)
        lc.to_csv(laps_path, index=False, encoding="utf-8")
        print(f"  Saved {label} laps    ({len(laps)} rows) -> {laps_path.name}")


for _, event in pre_hungarian.iterrows():
    round_num = int(event["RoundNumber"])
    name      = event["EventName"]

    print(f"\n{'='*60}")
    print(f"Round {round_num}: {name}")

    try:
        fetch_and_save(round_num, name, "race")
    except Exception as e:
        print(f"  ERROR loading race: {e}")

    if round_num in SPRINT_ROUNDS:
        try:
            fetch_and_save(round_num, name, "sprint")
        except Exception as e:
            print(f"  ERROR loading sprint: {e}")

print(f"\nDone. Files saved to {DATA_DIR}")
