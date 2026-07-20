import fastf1
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

ROUNDS = range(1, 10)  # R01-R09, Australia through British GP


def _clean_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) == "timedelta64[ns]":
            out[col] = out[col].astype(str)
    return out


def fetch_round(round_num: int) -> None:
    schedule = fastf1.get_event_schedule(2026, include_testing=False)
    event = schedule[schedule["RoundNumber"] == round_num].iloc[0]
    event_name = event["EventName"].replace(" ", "_").replace("/", "-")
    prefix = f"R{round_num:02d}_{event_name}"

    race = fastf1.get_session(2026, round_num, "R")
    race.load(telemetry=False, weather=True, messages=False)

    laps = _clean_timedeltas(race.laps)
    laps.to_csv(DATA_DIR / f"{prefix}_laps.csv", index=False)
    print(f"Round {round_num}: saved {len(laps)} race laps")

    weather = _clean_timedeltas(race.weather_data)
    weather.to_csv(DATA_DIR / f"{prefix}_weather.csv", index=False)
    print(f"Round {round_num}: saved {len(weather)} weather rows")

    quali = fastf1.get_session(2026, round_num, "Q")
    quali.load(telemetry=False, weather=False, messages=False)
    quali_laps = _clean_timedeltas(quali.laps)
    quali_laps.to_csv(DATA_DIR / f"{prefix}_quali_laps.csv", index=False)
    print(f"Round {round_num}: saved {len(quali_laps)} qualifying laps")


if __name__ == "__main__":
    for round_num in ROUNDS:
        print(f"\n{'='*60}\nRound {round_num}\n{'='*60}")
        try:
            fetch_round(round_num)
        except Exception as e:
            print(f"  ERROR loading Round {round_num}: {e}")
    print(f"\nDone. Files saved to {DATA_DIR}")
