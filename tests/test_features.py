import pandas as pd
import pytest
from driver_similarity.telemetry_metrics import compute_braking_metrics, compute_throttle_aggression


def test_compute_braking_metrics_counts_zones_and_onset_speed():
    telemetry = pd.DataFrame({
        "Brake": [False, False, True, True, True, False, False, True, True, False],
        "Speed": [300, 300, 280, 250, 220, 220, 300, 290, 260, 260],
    })
    result = compute_braking_metrics(telemetry)
    assert result["brake_zone_count"] == 2
    assert result["mean_brake_duration"] == pytest.approx((3 + 2) / 2)
    assert result["mean_brake_onset_speed"] == pytest.approx((280 + 290) / 2)


def test_compute_braking_metrics_no_braking_returns_zeros():
    telemetry = pd.DataFrame({"Brake": [False, False, False], "Speed": [300, 300, 300]})
    result = compute_braking_metrics(telemetry)
    assert result == {"brake_zone_count": 0, "mean_brake_duration": 0.0, "mean_brake_onset_speed": 0.0}


def test_compute_throttle_aggression_fraction_at_full_throttle():
    telemetry = pd.DataFrame({"Throttle": [100, 100, 50, 0, 99, 99.5]})
    assert compute_throttle_aggression(telemetry) == pytest.approx(4 / 6)


def test_compute_throttle_aggression_empty_returns_zero():
    telemetry = pd.DataFrame({"Throttle": []})
    assert compute_throttle_aggression(telemetry) == 0.0
