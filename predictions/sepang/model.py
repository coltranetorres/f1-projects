import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import shap

DATA_DIR = Path(__file__).parent.parent / "data_pred" / "data"

# ── 1. LOAD & MERGE ──────────────────────────────────────────────────────────

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
SPRINT_ROUNDS = {2, 4, 5, 9}

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
    agg = agg.merge(stints, on="Abbreviation", how="left")
    return agg

rows = []
for rn, name in RACES:
    prefix     = f"R{rn:02d}_{name}"
    results    = pd.read_csv(DATA_DIR / f"{prefix}_results.csv")
    laps_path  = DATA_DIR / f"{prefix}_laps.csv"
    if laps_path.exists():
        laps    = load_laps(laps_path)
        lap_agg = aggregate_laps(laps)
    else:
        print(f"  R{rn:02d} {name} — no laps file, lap features will use medians")
        lap_agg = pd.DataFrame({"Abbreviation": results["Abbreviation"]})

    sprint_feats = pd.DataFrame({
        "Abbreviation":  results["Abbreviation"],
        "sprint_pos":    np.nan,
        "sprint_points": 0.0,
        "sprint_grid":   np.nan,
        "has_sprint":    0,
    })
    if rn in SPRINT_ROUNDS:
        sp = pd.read_csv(DATA_DIR / f"{prefix}_sprint_results.csv")
        sprint_feats = sp[["Abbreviation", "Position", "Points", "GridPosition"]].copy()
        sprint_feats.columns = ["Abbreviation", "sprint_pos", "sprint_points", "sprint_grid"]
        sprint_feats["has_sprint"] = 1

    df = (results
          .merge(lap_agg,      on="Abbreviation", how="left")
          .merge(sprint_feats, on="Abbreviation", how="left"))
    df["round"] = rn
    rows.append(df)

data = pd.concat(rows, ignore_index=True)

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

data["Position"]     = pd.to_numeric(data["Position"],    errors="coerce")
data["GridPosition"] = pd.to_numeric(data["GridPosition"], errors="coerce")
data["positions_gained"] = data["GridPosition"] - data["Position"]
data["finished"]     = (data["Status"] == "Finished").astype(int)
data["sprint_pos"]   = pd.to_numeric(data["sprint_pos"],   errors="coerce").fillna(22)
data["sprint_grid"]  = pd.to_numeric(data["sprint_grid"],  errors="coerce").fillna(22)
data["sprint_points"]= pd.to_numeric(data["sprint_points"],errors="coerce").fillna(0)
data["has_sprint"]   = data["has_sprint"].fillna(0).astype(int)

team_enc   = LabelEncoder()
driver_enc = LabelEncoder()
data["team_encoded"]   = team_enc.fit_transform(data["TeamName"].fillna("Unknown"))
data["driver_encoded"] = driver_enc.fit_transform(data["FullName"].fillna("Unknown"))

# avg_race_position must reflect SEASON-TO-DATE form, not the same race's own
# running position — using the current race's own laps here would leak the
# outcome (running position ≈ finishing position) straight into the label.
data = data.sort_values(["driver_encoded", "round"]).reset_index(drop=True)
data["avg_race_position"] = (
    data.groupby("driver_encoded")["race_avg_position_raw"]
        .transform(lambda s: s.expanding().mean().shift(1))
)

# ── TARGET: 0=non-podium, 1=podium (top 3) ──────────────────────────────────
data["target"] = (data["Position"] <= 3).astype(int)

n_neg = (data["target"] == 0).sum()
n_pos = (data["target"] == 1).sum()
scale_pos_weight = n_neg / n_pos

FEATURES = [
    "GridPosition", "positions_gained", "finished",
    "mean_lap_s", "best_lap_s", "lap_consistency",
    "mean_s1_s", "mean_s2_s", "mean_s3_s",
    "avg_speed_i1", "avg_speed_i2", "avg_speed_fl", "max_speed_st",
    "personal_best_count", "avg_race_position", "num_stints",
    "sprint_pos", "sprint_points", "sprint_grid", "has_sprint",
    "team_encoded", "driver_encoded",
]

X = data[FEATURES].copy().apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))
y = data["target"]

print(f"Dataset      : {X.shape[0]} rows x {X.shape[1]} features")
print(f"Class counts : {dict(y.value_counts().sort_index())}  (0=non-podium, 1=podium)")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── 3. TRAIN / TEST SPLIT (80-20, STRATIFIED) ────────────────────────────────

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")
print(f"Train class dist: {dict(y_train.value_counts().sort_index())}")
print(f"Test  class dist: {dict(y_test.value_counts().sort_index())}")

# ── 4. XGBOOST (binary) ──────────────────────────────────────────────────────

model = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)
print(f"\nBest iteration: {model.best_iteration}")

# ── 5. EVALUATION ────────────────────────────────────────────────────────────

train_proba = model.predict_proba(X_train)[:, 1]
test_proba  = model.predict_proba(X_test)[:, 1]

train_pred = (train_proba >= 0.5).astype(int)
test_pred  = (test_proba  >= 0.5).astype(int)

train_acc  = accuracy_score(y_train, train_pred)
test_acc   = accuracy_score(y_test,  test_pred)
train_auc  = roc_auc_score(y_train, train_proba)
test_auc   = roc_auc_score(y_test,  test_proba)

print("\n" + "="*55)
print("EVALUATION METRICS")
print("="*55)
print(f"{'Metric':<30} {'Train':>8} {'Test':>8}")
print("-"*48)
print(f"{'Accuracy':<30} {train_acc:>8.4f} {test_acc:>8.4f}")
print(f"{'ROC-AUC':<30} {train_auc:>8.4f} {test_auc:>8.4f}")

acc_gap = train_acc - test_acc
auc_gap = train_auc - test_auc
print(f"\nAccuracy gap (train-test): {acc_gap:.4f}", end="  ")
if acc_gap > 0.15:
    print("[!] OVERFIT")
elif acc_gap > 0.05:
    print("[!] Mild overfitting")
else:
    print("[OK] Generalising well")

print(f"ROC-AUC  gap (train-test): {auc_gap:.4f}", end="  ")
if auc_gap > 0.15:
    print("[!] OVERFIT")
elif auc_gap > 0.05:
    print("[!] Mild overfitting")
else:
    print("[OK] Generalising well")

print("\nClassification Report (test set):")
print(classification_report(
    y_test, test_pred,
    target_names=["Non-podium", "Podium"],
    zero_division=0,
))

test_data = data.loc[idx_test].copy()
test_data["proba_podium"] = test_proba

print("="*55)
print("PODIUM PREDICTION vs ACTUAL (TEST SET RACES)")
print("="*55)
for rn, grp in test_data.groupby("round"):
    actual_podium = grp[grp["target"] == 1]["FullName"].tolist()
    pred_podium   = grp.nlargest(3, "proba_podium")["FullName"].tolist()
    hit = len(set(pred_podium) & set(actual_podium))
    print(f"\n  Round {rn}  (hit {hit}/3):")
    print(f"    Predicted : {', '.join(pred_podium)}")
    print(f"    Actual    : {', '.join(actual_podium) if actual_podium else '?'}")

print("\n" + "="*55)
print("FEATURE IMPORTANCE (top 10)")
print("="*55)
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, score in imp.head(10).items():
    bar = "#" * int(score * 300)
    print(f"  {feat:<25} {score:.4f}  {bar}")

# ── 6. SHAP ANALYSIS ─────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SHAP FEATURE IMPORTANCE (trained model, full dataset)")
print("="*55)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
mean_abs_shap = mean_abs_shap.sort_values(ascending=False)

print(f"\n  {'Feature':<25} {'Mean |SHAP|':>12}  Impact")
print(f"  {'-'*60}")
max_val = mean_abs_shap.iloc[0]
for feat, val in mean_abs_shap.items():
    bar = "#" * int(val / max_val * 40)
    print(f"  {feat:<25} {val:>12.4f}  {bar}")

# ── 7. SEPANG (ROUND 16) PREDICTION — NO SPRINT ─────────────────────────────

print("\n" + "="*55)
print("SEPANG GP (ROUND 16) - RETRAIN ON ALL DATA & PREDICT")
print("="*55)

model_full = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    n_estimators=model.best_iteration + 1,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
model_full.fit(X, y)

def _qual_to_sec(t):
    try:
        m, s = str(t).split(":")
        return float(m) * 60 + float(s)
    except Exception:
        return np.nan

qual = pd.read_csv(DATA_DIR / "R16_Bahrain_Grand_Prix_Qualifying_results.csv")
qual["qual_time_s"] = qual["Qualifying Time"].apply(_qual_to_sec)
qual["car_num"] = pd.to_numeric(qual["Car Number"], errors="coerce")

car_meta = (data
    .assign(car_num=lambda d: pd.to_numeric(d["DriverNumber"], errors="coerce"))
    .drop_duplicates("car_num")
    [["car_num", "FullName", "TeamName", "driver_encoded"]])

qual = qual.merge(car_meta, on="car_num", how="left")

agg_feats = [f for f in FEATURES if f != "driver_encoded"]
feature_agg = data.groupby("driver_encoded")[agg_feats].mean().reset_index()

sepang_drivers = qual.merge(feature_agg, on="driver_encoded", how="left")
sepang_drivers = sepang_drivers.reset_index(drop=True)

med = X.median(numeric_only=True)
for col in agg_feats:
    if col in sepang_drivers.columns:
        sepang_drivers[col] = sepang_drivers[col].fillna(med[col])

# Sepang GP is not a sprint weekend
sepang_drivers["GridPosition"]     = sepang_drivers["Position"]
sepang_drivers["positions_gained"] = 0
sepang_drivers["finished"]         = 1
sepang_drivers["has_sprint"]       = 0
sepang_drivers["sprint_pos"]       = sepang_drivers["sprint_pos"].fillna(22)
sepang_drivers["sprint_points"]    = sepang_drivers["sprint_points"].fillna(0)
sepang_drivers["sprint_grid"]      = sepang_drivers["sprint_grid"].fillna(22)
sepang_drivers["driver_encoded"]   = (sepang_drivers["driver_encoded"]
                                     .fillna(int(med["driver_encoded"])).astype(int))

X_sepang = sepang_drivers[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(med)

proba_sepang = model_full.predict_proba(X_sepang)[:, 1]
sepang_drivers["proba_podium"] = proba_sepang

ranked = (sepang_drivers
          .sort_values("proba_podium", ascending=False)
          .reset_index(drop=False))

podium_labels = ["P1", "P2", "P3"]

print("\n  PREDICTED SEPANG GP PODIUM:")
print(f"  {'Pos':<6} {'Driver':<25} {'Team':<22} {'Grid':>5} {'Qual Time':>11} {'P(podium)':>10}")
print(f"  {'-'*83}")
for i, label in enumerate(podium_labels):
    row = ranked.iloc[i]
    name = row["FullName"] if pd.notna(row["FullName"]) else str(row["Driver"])
    team = row["TeamName"] if pd.notna(row["TeamName"]) else str(row["Team"])
    qt   = f"{row['qual_time_s']:.3f}s" if pd.notna(row["qual_time_s"]) else "N/A"
    print(f"  {label:<6} {name:<25} {team:<22} {int(row['GridPosition']):>5} {qt:>11} {row['proba_podium']:>10.4f}")

print(f"\n  Full driver ranking by podium probability:")
print(f"  {'Rank':<5} {'Driver':<25} {'Team':<22} {'Grid':>5} {'Qual Time':>11} {'P(podium)':>10}")
print(f"  {'-'*83}")
for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
    name = row["FullName"] if pd.notna(row["FullName"]) else str(row["Driver"])
    team = row["TeamName"] if pd.notna(row["TeamName"]) else str(row["Team"])
    qt   = f"{row['qual_time_s']:.3f}s" if pd.notna(row["qual_time_s"]) else "N/A"
    marker = " <-- podium" if rank <= 3 else ""
    print(f"  {rank:<5} {name:<25} {team:<22} {int(row['GridPosition']):>5} {qt:>11} {row['proba_podium']:>10.4f}{marker}")

# SHAP breakdown for top contenders
print("\n" + "="*55)
print("SHAP BREAKDOWN - TOP 4 SEPANG GP CONTENDERS")
print("="*55)
print("(positive SHAP = pushes toward podium, negative = away)\n")

explainer_full = shap.TreeExplainer(model_full)
shap_sepang = explainer_full.shap_values(X_sepang)

for _, row in ranked.head(4).iterrows():
    orig_pos = int(row["index"])
    name = row["FullName"] if pd.notna(row["FullName"]) else str(row["Driver"])
    team = row["TeamName"] if pd.notna(row["TeamName"]) else str(row["Team"])
    driver_shap = pd.Series(shap_sepang[orig_pos], index=FEATURES).sort_values(key=abs, ascending=False)
    print(f"  {name} ({team})  -  P(podium)={row['proba_podium']:.4f}")
    print(f"  {'Feature':<25} {'SHAP':>8}  Direction")
    print(f"  {'-'*50}")
    for feat, val in driver_shap.head(8).items():
        direction = "+" if val > 0 else "-"
        bar = "#" * int(abs(val) / (abs(driver_shap.iloc[0]) + 1e-9) * 20)
        print(f"  {feat:<25} {val:>8.4f}  {direction} {bar}")
    print()
