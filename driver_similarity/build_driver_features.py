from pathlib import Path
import numpy as np
import pandas as pd

from driver_similarity.features import (
    build_race_feature_row,
    compute_wet_performance,
    aggregate_driver_features,
    is_wet_session,
)

DATA_DIR = Path(__file__).parent / "data"
ROUNDS = range(1, 10)


def _event_prefix(round_num: int) -> str:
    matches = sorted(
        m for m in DATA_DIR.glob(f"R{round_num:02d}_*_laps.csv")
        if "_sprint_" not in m.name
    )
    if not matches:
        raise FileNotFoundError(f"No laps CSV found for round {round_num}")
    return matches[0].name.removesuffix("_laps.csv")


def main() -> pd.DataFrame:
    race_rows = []
    pace_rows = []

    for round_num in ROUNDS:
        prefix = _event_prefix(round_num)
        race_laps = pd.read_csv(DATA_DIR / f"{prefix}_laps.csv")
        quali_laps = pd.read_csv(DATA_DIR / f"{prefix}_quali_laps.csv")
        weather = pd.read_csv(DATA_DIR / f"{prefix}_weather.csv")
        telemetry_path = DATA_DIR / f"{prefix}_telemetry_features.csv"
        telemetry_features = pd.read_csv(telemetry_path) if telemetry_path.exists() else pd.DataFrame(
            columns=["driver", "brake_zone_count", "mean_brake_duration", "mean_brake_onset_speed", "throttle_aggression"]
        )

        wet = is_wet_session(weather)
        race_laps["LapTime"] = pd.to_timedelta(race_laps["LapTime"]).dt.total_seconds()
        quali_laps["LapTime"] = pd.to_timedelta(quali_laps["LapTime"]).dt.total_seconds()
        race_median = race_laps["LapTime"].median()

        for driver in race_laps["Driver"].dropna().unique():
            race_rows.append(build_race_feature_row(driver, race_laps, quali_laps, weather, telemetry_features, round_num))
            driver_pace = race_laps.loc[race_laps["Driver"] == driver, "LapTime"].mean()
            pace_rows.append({
                "driver": driver, "round": round_num, "is_wet": wet,
                "pace_delta": float(driver_pace - race_median) if pd.notna(driver_pace) else np.nan,
            })

    race_feature_rows = pd.DataFrame(race_rows)
    driver_race_pace = pd.DataFrame(pace_rows).dropna(subset=["pace_delta"])
    wet_perf = compute_wet_performance(driver_race_pace)

    race_feature_rows["wet_performance"] = race_feature_rows["driver"].map(wet_perf)

    driver_features = aggregate_driver_features(race_feature_rows)
    driver_features.to_csv(DATA_DIR / "driver_features.csv", index=False)
    print(f"Saved season-level features for {len(driver_features)} drivers to {DATA_DIR / 'driver_features.csv'}")
    return driver_features


if __name__ == "__main__":
    main()
