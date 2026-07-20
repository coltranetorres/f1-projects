import numpy as np
import pandas as pd


def qualifying_pace(quali_laps: pd.DataFrame, driver: str) -> float | None:
    best = quali_laps.groupby("Driver")["LapTime"].min()
    if driver not in best.index or best.empty:
        return None
    pole = best.min()
    if pole <= 0:
        return None
    return float((best[driver] - pole) / pole * 100)


def count_overtakes(race_laps: pd.DataFrame, driver: str) -> int:
    d = race_laps[race_laps["Driver"] == driver].sort_values("LapNumber")
    d = d[d["LapNumber"] > 1]
    d = d[d["PitInTime"].isna() & d["PitOutTime"].isna()]
    if len(d) < 2:
        return 0
    pos = d["Position"].astype(float).to_numpy()
    deltas = pos[:-1] - pos[1:]
    gains = deltas[deltas > 0]
    return int(gains.sum())


def tire_preservation_slope(race_laps: pd.DataFrame, driver: str) -> float | None:
    d = race_laps[(race_laps["Driver"] == driver) & (race_laps["IsAccurate"] == True)]
    slopes = []
    for _, stint_laps in d.groupby("Stint"):
        if len(stint_laps) < 3:
            continue
        x = stint_laps["TyreLife"].astype(float).to_numpy()
        y = stint_laps["LapTime"].astype(float).to_numpy()
        if np.std(x) == 0:
            continue
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    if not slopes:
        return None
    return float(np.mean(slopes))


def is_wet_session(weather: pd.DataFrame) -> bool:
    if weather.empty:
        return False
    return bool(weather["Rainfall"].astype(bool).mean() > 0.5)


def build_race_feature_row(
    driver: str,
    race_laps: pd.DataFrame,
    quali_laps: pd.DataFrame,
    weather: pd.DataFrame,
    telemetry_features: pd.DataFrame,
    round_num: int,
) -> dict:
    tf = telemetry_features[telemetry_features["driver"] == driver]
    tf_row = tf.iloc[0] if not tf.empty else None
    wet = is_wet_session(weather)

    return {
        "driver": driver,
        "round": round_num,
        "qualifying_pace": qualifying_pace(quali_laps, driver),
        "overtakes": count_overtakes(race_laps, driver),
        "tire_preservation": tire_preservation_slope(race_laps, driver),
        "brake_zone_count": float(tf_row["brake_zone_count"]) if tf_row is not None else None,
        "mean_brake_duration": float(tf_row["mean_brake_duration"]) if tf_row is not None else None,
        "mean_brake_onset_speed": float(tf_row["mean_brake_onset_speed"]) if tf_row is not None else None,
        "throttle_aggression": float(tf_row["throttle_aggression"]) if tf_row is not None else None,
        "is_wet": wet,
        # wet_performance is computed later at aggregation time (needs both
        # wet and dry race pace for the same driver); left None per-race.
        "wet_performance": None,
    }


AGG_COLUMNS = [
    "qualifying_pace", "overtakes", "tire_preservation",
    "brake_zone_count", "mean_brake_duration", "mean_brake_onset_speed",
    "throttle_aggression",
]


def compute_wet_performance(driver_race_pace: pd.DataFrame) -> pd.Series:
    """driver_race_pace: one row per (driver, round) with columns
    ['driver', 'is_wet', 'pace_delta'] where pace_delta is the driver's mean
    lap time delta to that race's median lap time (lower = faster relative
    to the field). Returns, per driver, dry_mean - wet_mean (positive = the
    driver is relatively better/faster in the wet); NaN if the driver has no
    wet or no dry races in the dataset.
    """
    def _delta(group: pd.DataFrame):
        dry = group.loc[~group["is_wet"], "pace_delta"]
        wet = group.loc[group["is_wet"], "pace_delta"]
        if dry.empty or wet.empty:
            return np.nan
        return float(dry.mean() - wet.mean())

    return driver_race_pace.groupby("driver").apply(_delta, include_groups=False)


def aggregate_driver_features(race_feature_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = race_feature_rows.groupby("driver")[AGG_COLUMNS].mean()

    wet_by_driver = race_feature_rows.groupby("driver")["wet_performance"].mean()
    has_wet = race_feature_rows.groupby("driver")["wet_performance"].apply(lambda s: s.notna().any())
    grid_wet_mean = wet_by_driver.mean(skipna=True)
    imputed_wet = wet_by_driver.fillna(grid_wet_mean if pd.notna(grid_wet_mean) else 0.0)

    grouped["wet_performance"] = imputed_wet
    grouped["has_wet_data"] = has_wet
    return grouped.reset_index()
