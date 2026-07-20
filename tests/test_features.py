import pandas as pd
import pytest
from driver_similarity.telemetry_metrics import compute_braking_metrics, compute_throttle_aggression
from driver_similarity.features import (
    qualifying_pace,
    count_overtakes,
    tire_preservation_slope,
    is_wet_session,
    build_race_feature_row,
    aggregate_driver_features,
    compute_wet_performance,
)


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


def test_qualifying_pace_delta_to_pole():
    quali_laps = pd.DataFrame({
        "Driver": ["VER", "VER", "HAM", "NOR"],
        "LapTime": [80.5, 79.8, 80.0, 81.0],
    })
    # pole is HAM/VER best = 79.8 (VER)
    assert qualifying_pace(quali_laps, "VER") == pytest.approx(0.0)
    assert qualifying_pace(quali_laps, "HAM") == pytest.approx((80.0 - 79.8) / 79.8 * 100)


def test_qualifying_pace_missing_driver_returns_none():
    quali_laps = pd.DataFrame({"Driver": ["VER"], "LapTime": [79.8]})
    assert qualifying_pace(quali_laps, "HAM") is None


def test_count_overtakes_sums_position_gains_excluding_lap1_and_pits():
    race_laps = pd.DataFrame({
        "Driver": ["VER"] * 5,
        "LapNumber": [1, 2, 3, 4, 5],
        "Position": [10, 8, 8, 5, 5],
        "PitInTime": [None, None, "0:01:00", None, None],
        "PitOutTime": [None, None, None, "0:01:05", None],
    })
    # lap1->2 excluded (lap1); lap2->3 no change; lap3 excluded (PitInTime set);
    # lap4 excluded (PitOutTime set); only lap4->5 comparison remains but lap4 is
    # filtered out entirely, so remaining laps after filtering are 2,3,5 -> gains: 8->8 (0), 8->5 (3)
    assert count_overtakes(race_laps, "VER") == 3


def test_count_overtakes_no_gains_returns_zero():
    race_laps = pd.DataFrame({
        "Driver": ["VER"] * 3,
        "LapNumber": [2, 3, 4],
        "Position": [5, 6, 7],
        "PitInTime": [None, None, None],
        "PitOutTime": [None, None, None],
    })
    assert count_overtakes(race_laps, "VER") == 0


def test_tire_preservation_slope_flat_stint_is_near_zero():
    race_laps = pd.DataFrame({
        "Driver": ["VER"] * 4,
        "IsAccurate": [True] * 4,
        "Stint": [1, 1, 1, 1],
        "TyreLife": [1, 2, 3, 4],
        "LapTime": [80.0, 80.0, 80.0, 80.0],
    })
    assert tire_preservation_slope(race_laps, "VER") == pytest.approx(0.0, abs=1e-9)


def test_tire_preservation_slope_degrading_stint_is_positive():
    race_laps = pd.DataFrame({
        "Driver": ["VER"] * 4,
        "IsAccurate": [True] * 4,
        "Stint": [1, 1, 1, 1],
        "TyreLife": [1, 2, 3, 4],
        "LapTime": [80.0, 80.5, 81.0, 81.5],
    })
    assert tire_preservation_slope(race_laps, "VER") == pytest.approx(0.5)


def test_tire_preservation_slope_insufficient_laps_returns_none():
    race_laps = pd.DataFrame({
        "Driver": ["VER"] * 2,
        "IsAccurate": [True] * 2,
        "Stint": [1, 1],
        "TyreLife": [1, 2],
        "LapTime": [80.0, 80.5],
    })
    assert tire_preservation_slope(race_laps, "VER") is None


def test_is_wet_session_majority_rainfall_true():
    weather = pd.DataFrame({"Rainfall": [True, True, True, False]})
    assert is_wet_session(weather) is True


def test_is_wet_session_majority_dry():
    weather = pd.DataFrame({"Rainfall": [True, False, False, False]})
    assert is_wet_session(weather) is False


def test_build_race_feature_row_combines_all_sources():
    race_laps = pd.DataFrame({
        "Driver": ["VER", "VER", "VER"],
        "LapNumber": [2, 3, 4],
        "Position": [5, 4, 4],
        "PitInTime": [None, None, None],
        "PitOutTime": [None, None, None],
        "IsAccurate": [True, True, True],
        "Stint": [1, 1, 1],
        "TyreLife": [1, 2, 3],
        "LapTime": [80.0, 80.2, 80.4],
    })
    quali_laps = pd.DataFrame({"Driver": ["VER", "HAM"], "LapTime": [79.0, 79.5]})
    weather = pd.DataFrame({"Rainfall": [False, False]})
    telemetry_features = pd.DataFrame({
        "driver": ["VER"],
        "brake_zone_count": [12.0],
        "mean_brake_duration": [8.0],
        "mean_brake_onset_speed": [270.0],
        "throttle_aggression": [0.6],
    })

    row = build_race_feature_row("VER", race_laps, quali_laps, weather, telemetry_features, round_num=1)

    assert row["driver"] == "VER"
    assert row["round"] == 1
    assert row["qualifying_pace"] == pytest.approx(0.0)
    assert row["overtakes"] == 1
    assert row["tire_preservation"] == pytest.approx(0.2)
    assert row["brake_zone_count"] == pytest.approx(12.0)
    assert row["is_wet"] is False
    assert row["wet_performance"] is None


def test_build_race_feature_row_missing_telemetry_driver_returns_none_fields():
    race_laps = pd.DataFrame({
        "Driver": ["HAM", "HAM"],
        "LapNumber": [2, 3],
        "Position": [5, 5],
        "PitInTime": [None, None],
        "PitOutTime": [None, None],
        "IsAccurate": [True, True],
        "Stint": [1, 1],
        "TyreLife": [1, 2],
        "LapTime": [81.0, 81.0],
    })
    quali_laps = pd.DataFrame({"Driver": ["HAM"], "LapTime": [79.5]})
    weather = pd.DataFrame({"Rainfall": [False]})
    telemetry_features = pd.DataFrame({
        "driver": ["VER"], "brake_zone_count": [12.0], "mean_brake_duration": [8.0],
        "mean_brake_onset_speed": [270.0], "throttle_aggression": [0.6],
    })

    row = build_race_feature_row("HAM", race_laps, quali_laps, weather, telemetry_features, round_num=1)
    assert row["brake_zone_count"] is None
    assert row["throttle_aggression"] is None


def test_aggregate_driver_features_averages_across_rounds_and_imputes_wet():
    race_feature_rows = pd.DataFrame([
        {"driver": "VER", "round": 1, "qualifying_pace": 0.0, "overtakes": 2, "tire_preservation": 0.1,
         "brake_zone_count": 10.0, "mean_brake_duration": 8.0, "mean_brake_onset_speed": 270.0,
         "throttle_aggression": 0.5, "is_wet": False, "wet_performance": None},
        {"driver": "VER", "round": 2, "qualifying_pace": 0.5, "overtakes": 0, "tire_preservation": 0.2,
         "brake_zone_count": 12.0, "mean_brake_duration": 9.0, "mean_brake_onset_speed": 260.0,
         "throttle_aggression": 0.6, "is_wet": True, "wet_performance": 1.2},
        {"driver": "HAM", "round": 1, "qualifying_pace": 1.0, "overtakes": 1, "tire_preservation": 0.05,
         "brake_zone_count": 9.0, "mean_brake_duration": 7.0, "mean_brake_onset_speed": 280.0,
         "throttle_aggression": 0.4, "is_wet": False, "wet_performance": None},
    ])

    result = aggregate_driver_features(race_feature_rows)
    result = result.set_index("driver")

    assert result.loc["VER", "qualifying_pace"] == pytest.approx(0.25)
    assert result.loc["VER", "overtakes"] == pytest.approx(1.0)
    assert result.loc["VER", "wet_performance"] == pytest.approx(1.2)
    assert bool(result.loc["VER", "has_wet_data"]) is True

    # HAM has no wet data -> imputed with grid mean of non-null wet_performance (1.2)
    assert result.loc["HAM", "wet_performance"] == pytest.approx(1.2)
    assert bool(result.loc["HAM", "has_wet_data"]) is False


def test_compute_wet_performance_positive_when_better_in_wet():
    driver_race_pace = pd.DataFrame([
        {"driver": "VER", "round": 1, "is_wet": False, "pace_delta": 1.0},
        {"driver": "VER", "round": 2, "is_wet": False, "pace_delta": 1.5},
        {"driver": "VER", "round": 3, "is_wet": True, "pace_delta": 0.2},
        {"driver": "HAM", "round": 1, "is_wet": False, "pace_delta": 0.5},
    ])
    result = compute_wet_performance(driver_race_pace)
    assert result["VER"] == pytest.approx(1.25 - 0.2)
    assert pd.isna(result["HAM"])
