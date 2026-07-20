import fastf1
import numpy as np
import pandas as pd
from pathlib import Path
from driver_similarity.telemetry_metrics import compute_braking_metrics, compute_throttle_aggression

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

ROUNDS = range(1, 10)  # R01-R09, Australia through British GP

N_SAMPLED_LAPS = 5


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


def fetch_telemetry_features(round_num: int) -> None:
    schedule = fastf1.get_event_schedule(2026, include_testing=False)
    event = schedule[schedule["RoundNumber"] == round_num].iloc[0]
    event_name = event["EventName"].replace(" ", "_").replace("/", "-")
    prefix = f"R{round_num:02d}_{event_name}"

    race = fastf1.get_session(2026, round_num, "R")
    race.load(telemetry=True, weather=False, messages=False)

    rows = []
    for driver in race.laps["Driver"].unique():
        driver_laps = race.laps.pick_drivers(driver).pick_accurate()
        driver_laps = driver_laps[driver_laps["PitInTime"].isna() & driver_laps["PitOutTime"].isna()]
        if driver_laps.empty:
            continue
        fastest = driver_laps.sort_values("LapTime").head(N_SAMPLED_LAPS)

        brake_metrics = []
        throttle_scores = []
        for _, lap in fastest.iterrows():
            try:
                car_data = lap.get_car_data()
            except Exception:
                continue
            brake_metrics.append(compute_braking_metrics(car_data))
            throttle_scores.append(compute_throttle_aggression(car_data))

        if not brake_metrics:
            continue

        rows.append({
            "driver": driver,
            "brake_zone_count": float(np.mean([m["brake_zone_count"] for m in brake_metrics])),
            "mean_brake_duration": float(np.mean([m["mean_brake_duration"] for m in brake_metrics])),
            "mean_brake_onset_speed": float(np.mean([m["mean_brake_onset_speed"] for m in brake_metrics])),
            "throttle_aggression": float(np.mean(throttle_scores)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / f"{prefix}_telemetry_features.csv", index=False)
    print(f"Round {round_num}: saved telemetry features for {len(out)} drivers")


if __name__ == "__main__":
    for round_num in ROUNDS:
        print(f"\n{'='*60}\nRound {round_num}\n{'='*60}")
        try:
            fetch_round(round_num)
            fetch_telemetry_features(round_num)
        except Exception as e:
            print(f"  ERROR loading Round {round_num}: {e}")
    print(f"\nDone. Files saved to {DATA_DIR}")
