import fastf1
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

cache_dir = Path(__file__).parent / "cache"
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

schedule = fastf1.get_event_schedule(2026, include_testing=False)
print("2026 Schedule:")
print(schedule[["RoundNumber", "EventName", "EventDate"]].to_string())

monaco_idx = schedule[schedule["EventName"].str.contains("Monaco", case=False)].index
if len(monaco_idx) == 0:
    print("Monaco GP not found in schedule")
    exit(1)

monaco_round = schedule.loc[monaco_idx[0], "RoundNumber"]
print(f"\nMonaco GP is Round {monaco_round}")

pre_monaco = schedule[schedule["RoundNumber"] < monaco_round]
print(f"\nFetching data for {len(pre_monaco)} races before Monaco:")
print(pre_monaco[["RoundNumber", "EventName"]].to_string())

for _, event in pre_monaco.iterrows():
    round_num = event["RoundNumber"]
    event_name = event["EventName"].replace(" ", "_").replace("/", "-")
    print(f"\n{'='*60}")
    print(f"Round {round_num}: {event['EventName']}")

    try:
        session = fastf1.get_session(2026, round_num, "R")
        session.load(telemetry=False, weather=True, messages=False)

        prefix = f"R{round_num:02d}_{event_name}"

        # Results
        results = session.results
        if results is not None and len(results) > 0:
            results_clean = results.copy()
            for col in results_clean.columns:
                if hasattr(results_clean[col], 'dt') or str(results_clean[col].dtype) == 'timedelta64[ns]':
                    results_clean[col] = results_clean[col].astype(str)
            results_clean.to_csv(DATA_DIR / f"{prefix}_results.csv", index=False)
            print(f"  Saved results ({len(results)} rows)")

        # Laps
        laps = session.laps
        if laps is not None and len(laps) > 0:
            laps_clean = laps.copy()
            for col in laps_clean.columns:
                if str(laps_clean[col].dtype) == 'timedelta64[ns]':
                    laps_clean[col] = laps_clean[col].astype(str)
            laps_clean.to_csv(DATA_DIR / f"{prefix}_laps.csv", index=False)
            print(f"  Saved laps ({len(laps)} rows)")

        # Weather
        weather = session.weather_data
        if weather is not None and len(weather) > 0:
            weather_clean = weather.copy()
            for col in weather_clean.columns:
                if str(weather_clean[col].dtype) == 'timedelta64[ns]':
                    weather_clean[col] = weather_clean[col].astype(str)
            weather_clean.to_csv(DATA_DIR / f"{prefix}_weather.csv", index=False)
            print(f"  Saved weather ({len(weather)} rows)")

    except Exception as e:
        print(f"  ERROR loading Round {round_num}: {e}")

print(f"\nDone. Files saved to {DATA_DIR}")
