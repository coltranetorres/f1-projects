"""Feature pipeline for the podium model, extracted from sepang/model.py."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data_pred" / "data"

RACES = [
    (1, "Australian_Grand_Prix"),
    (2, "Chinese_Grand_Prix"),
    (3, "Japanese_Grand_Prix"),
    (4, "Miami_Grand_Prix"),
    (5, "Canadian_Grand_Prix"),
    (6, "Monaco_Grand_Prix"),
    (7, "Barcelona_Grand_Prix"),
    (8, "Austrian_Grand_Prix"),
    (9, "British_Grand_Prix"),
    (10, "Belgian_Grand_Prix"),
    (11, "Hungarian_Grand_Prix"),
    (12, "Dutch_Grand_Prix"),
    (13, "Italian_Grand_Prix"),
    (14, "Spanish_Grand_Prix"),
    (15, "Azerbaijan_Grand_Prix"),
]
RACE_NAMES = {rn: name.replace("_", " ") for rn, name in RACES}
SPRINT_ROUNDS = {2, 4, 5, 9}

TARGET_ROUND = 16
QUALI_FILE = "R16_Bahrain_Grand_Prix_Qualifying_results.csv"

FEATURES = [
    "GridPosition", "positions_gained", "finished",
    "mean_lap_s", "best_lap_s", "lap_consistency",
    "mean_s1_s", "mean_s2_s", "mean_s3_s",
    "avg_speed_i1", "avg_speed_i2", "avg_speed_fl", "max_speed_st",
    "personal_best_count", "avg_race_position", "num_stints",
    "sprint_pos", "sprint_points", "sprint_grid", "has_sprint",
    "team_encoded", "driver_encoded",
]


def parse_td(series):
    def _to_sec(v):
        if pd.isna(v) or str(v) in ("NaT", ""):
            return np.nan
        try:
            return pd.to_timedelta(v).total_seconds()
        except Exception:
            return np.nan
    return series.apply(_to_sec)


def load_laps(path):
    df = pd.read_csv(path)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[col] = parse_td(df[col])
    return df


def aggregate_laps(laps_df):
    acc = laps_df[laps_df["IsAccurate"] == True].copy()
    grp = acc.groupby("Driver")
    agg = pd.DataFrame({
        "mean_lap_s":          grp["LapTime"].mean(),
        "best_lap_s":          grp["LapTime"].min(),
        "lap_consistency":     grp["LapTime"].std(),
        "mean_s1_s":           grp["Sector1Time"].mean(),
        "mean_s2_s":           grp["Sector2Time"].mean(),
        "mean_s3_s":           grp["Sector3Time"].mean(),
        "avg_speed_i1":        grp["SpeedI1"].mean(),
        "avg_speed_i2":        grp["SpeedI2"].mean(),
        "avg_speed_fl":        grp["SpeedFL"].mean(),
        "max_speed_st":        grp["SpeedST"].max(),
        "personal_best_count": grp["IsPersonalBest"].apply(
                                   lambda x: pd.to_numeric(x, errors="coerce").sum()),
        "race_avg_position_raw": grp["Position"].mean(),
    }).reset_index().rename(columns={"Driver": "Abbreviation"})

    stints = laps_df.groupby("Driver")["Stint"].nunique().reset_index()
    stints.columns = ["Abbreviation", "num_stints"]
    return agg.merge(stints, on="Abbreviation", how="left")


def build_training_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for rn, name in RACES:
        prefix = f"R{rn:02d}_{name}"
        results = pd.read_csv(data_dir / f"{prefix}_results.csv")
        laps_path = data_dir / f"{prefix}_laps.csv"
        if laps_path.exists():
            lap_agg = aggregate_laps(load_laps(laps_path))
        else:
            lap_agg = pd.DataFrame({"Abbreviation": results["Abbreviation"]})

        sprint_feats = pd.DataFrame({
            "Abbreviation":  results["Abbreviation"],
            "sprint_pos":    np.nan,
            "sprint_points": 0.0,
            "sprint_grid":   np.nan,
            "has_sprint":    0,
        })
        if rn in SPRINT_ROUNDS:
            sp = pd.read_csv(data_dir / f"{prefix}_sprint_results.csv")
            sprint_feats = sp[["Abbreviation", "Position", "Points", "GridPosition"]].copy()
            sprint_feats.columns = ["Abbreviation", "sprint_pos", "sprint_points", "sprint_grid"]
            sprint_feats["has_sprint"] = 1

        df = (results
              .merge(lap_agg, on="Abbreviation", how="left")
              .merge(sprint_feats, on="Abbreviation", how="left"))
        df["round"] = rn
        rows.append(df)

    data = pd.concat(rows, ignore_index=True)

    data["Position"] = pd.to_numeric(data["Position"], errors="coerce")
    data["GridPosition"] = pd.to_numeric(data["GridPosition"], errors="coerce")
    data["positions_gained"] = data["GridPosition"] - data["Position"]
    data["finished"] = (data["Status"] == "Finished").astype(int)
    data["sprint_pos"] = pd.to_numeric(data["sprint_pos"], errors="coerce").fillna(22)
    data["sprint_grid"] = pd.to_numeric(data["sprint_grid"], errors="coerce").fillna(22)
    data["sprint_points"] = pd.to_numeric(data["sprint_points"], errors="coerce").fillna(0)
    data["has_sprint"] = data["has_sprint"].fillna(0).astype(int)

    data["team_encoded"] = LabelEncoder().fit_transform(data["TeamName"].fillna("Unknown"))
    data["driver_encoded"] = LabelEncoder().fit_transform(data["FullName"].fillna("Unknown"))

    # Season-to-date form only; using the same race's own running position would leak the label.
    data = data.sort_values(["driver_encoded", "round"]).reset_index(drop=True)
    data["avg_race_position"] = (
        data.groupby("driver_encoded")["race_avg_position_raw"]
            .transform(lambda s: s.expanding().mean().shift(1))
    )

    data["target"] = (data["Position"] <= 3).astype(int)
    return data


def feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data[FEATURES].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, data["target"]
