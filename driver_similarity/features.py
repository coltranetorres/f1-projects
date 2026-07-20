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
