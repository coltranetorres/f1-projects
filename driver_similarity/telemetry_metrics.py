import numpy as np
import pandas as pd


def compute_braking_metrics(telemetry: pd.DataFrame) -> dict:
    brake = telemetry["Brake"].astype(bool).to_numpy()
    speed = telemetry["Speed"].astype(float).to_numpy()

    zones = 0
    durations = []
    onset_speeds = []
    in_zone = False
    zone_len = 0

    for i, is_braking in enumerate(brake):
        if is_braking and not in_zone:
            in_zone = True
            zone_len = 1
            zones += 1
            onset_speeds.append(speed[i])
        elif is_braking and in_zone:
            zone_len += 1
        elif not is_braking and in_zone:
            in_zone = False
            durations.append(zone_len)
    if in_zone:
        durations.append(zone_len)

    return {
        "brake_zone_count": zones,
        "mean_brake_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_brake_onset_speed": float(np.mean(onset_speeds)) if onset_speeds else 0.0,
    }


def compute_throttle_aggression(telemetry: pd.DataFrame) -> float:
    throttle = telemetry["Throttle"].astype(float).to_numpy()
    if len(throttle) == 0:
        return 0.0
    return float(np.mean(throttle >= 99))
