"""Podium model: training, prediction and explanation, extracted from sepang/model.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .data import DATA_DIR, FEATURES, QUALI_FILE, RACE_NAMES, build_training_data, feature_matrix

XGB_PARAMS = dict(
    objective="binary:logistic", max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)


def _qual_to_sec(t):
    try:
        m, s = str(t).split(":")
        return float(m) * 60 + float(s)
    except Exception:
        return np.nan


def _int_or_none(x):
    return None if x is None or pd.isna(x) else int(x)


class Predictor:
    """Trains once on construction; all methods are read-only afterwards."""

    def __init__(self, data_dir: Path = DATA_DIR, quali_path: Path | None = None):
        self.data_dir = data_dir
        self.quali_path = quali_path or (data_dir / QUALI_FILE)
        self.data = build_training_data(data_dir)
        self.X, self.y = feature_matrix(self.data)
        self.median = self.X.median(numeric_only=True)
        self.scale_pos_weight = (self.y == 0).sum() / (self.y == 1).sum()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.20, stratify=self.y, random_state=42)
        self.model = XGBClassifier(
            **XGB_PARAMS, scale_pos_weight=self.scale_pos_weight, n_estimators=300,
            eval_metric="logloss", early_stopping_rounds=30)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        test_proba = self.model.predict_proba(X_test)[:, 1]
        self._test_acc = float(accuracy_score(y_test, (test_proba >= 0.5).astype(int)))
        self._test_auc = float(roc_auc_score(y_test, test_proba))

        self.model_full = XGBClassifier(
            **XGB_PARAMS, scale_pos_weight=self.scale_pos_weight,
            n_estimators=self.model.best_iteration + 1)
        self.model_full.fit(self.X, self.y)

        self._target = self._build_target_frame()
        self._shap_cache = None
        self._importance = None

    # ── target-race frame (script lines 303-343) ─────────────────────────────
    def _build_target_frame(self) -> pd.DataFrame:
        data = self.data
        qual = pd.read_csv(self.quali_path)
        qual["qual_time_s"] = qual["Qualifying Time"].apply(_qual_to_sec)
        qual["car_num"] = pd.to_numeric(qual["Car Number"], errors="coerce")

        car_meta = (data
            .assign(car_num=lambda d: pd.to_numeric(d["DriverNumber"], errors="coerce"))
            .drop_duplicates("car_num")
            [["car_num", "FullName", "TeamName", "Abbreviation", "driver_encoded"]])
        qual = qual.merge(car_meta, on="car_num", how="left")

        agg_feats = [f for f in FEATURES if f != "driver_encoded"]
        feature_agg = data.groupby("driver_encoded")[agg_feats].mean().reset_index()
        t = qual.merge(feature_agg, on="driver_encoded", how="left").reset_index(drop=True)

        for col in agg_feats:
            if col in t.columns:
                t[col] = t[col].fillna(self.median[col])

        t["GridPosition"] = t["Position"]
        t["positions_gained"] = 0
        t["finished"] = 1
        t["has_sprint"] = 0
        t["sprint_pos"] = t["sprint_pos"].fillna(22)
        t["sprint_points"] = t["sprint_points"].fillna(0)
        t["sprint_grid"] = t["sprint_grid"].fillna(22)
        t["driver_encoded"] = t["driver_encoded"].fillna(int(self.median["driver_encoded"])).astype(int)

        # Entries never seen in training (new car number) fall back to the qualifying sheet.
        t["name"] = t["FullName"].where(t["FullName"].notna(), t["Driver"])
        t["team"] = t["TeamName"].where(t["TeamName"].notna(), t["Team"])
        fallback_code = t["Driver"].str.replace(r"[^A-Za-z]", "", regex=True).str[:3].str.upper()
        t["code"] = t["Abbreviation"].where(t["Abbreviation"].notna(), fallback_code)
        return t

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(self.median)

    def predict(self, grid_override: dict[str, int] | None = None) -> pd.DataFrame:
        """Rank the target-race field by podium probability. `grid_override` maps code -> grid slot."""
        df = self._target.copy()
        if grid_override:
            df["GridPosition"] = df["code"].map(grid_override).fillna(df["GridPosition"])
        df["proba_podium"] = self.model_full.predict_proba(self._features(df))[:, 1]
        df = df.sort_values("proba_podium", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    # ── lookups ──────────────────────────────────────────────────────────────
    def drivers(self) -> list[dict]:
        t = self._target.sort_values("GridPosition")
        return [dict(code=r["code"], name=r["name"], team=r["team"], grid=int(r["GridPosition"]))
                for _, r in t.iterrows()]

    def completed_rounds(self) -> list[int]:
        return sorted(int(r) for r in self.data["round"].unique())

    def driver_form(self, code: str, last_n: int = 5) -> list[dict]:
        d = self.data[self.data["Abbreviation"] == code].sort_values("round").tail(last_n)
        return [dict(round=int(r["round"]), race=RACE_NAMES[int(r["round"])],
                     grid=_int_or_none(r["GridPosition"]), finish=_int_or_none(r["Position"]),
                     positions_gained=_int_or_none(r["positions_gained"]), status=str(r["Status"]))
                for _, r in d.iterrows()]

    def race_result(self, round_number: int) -> list[dict]:
        d = self.data[self.data["round"] == round_number].sort_values("Position")
        return [dict(finish=_int_or_none(r["Position"]), code=r["Abbreviation"], name=r["FullName"],
                     team=r["TeamName"], grid=_int_or_none(r["GridPosition"]), status=str(r["Status"]),
                     points=float(r["Points"]) if pd.notna(r["Points"]) else 0.0)
                for _, r in d.iterrows()]

    def model_info(self) -> dict:
        return dict(n_rows=int(len(self.X)), n_features=int(self.X.shape[1]),
                    rounds_covered=self.completed_rounds(),
                    best_iteration=int(self.model.best_iteration),
                    test_accuracy=self._test_acc, test_auc=self._test_auc)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    def _target_shap(self):
        if self._shap_cache is None:
            df = self.predict()
            X_t = self._features(df)
            sv = shap.TreeExplainer(self.model_full).shap_values(X_t)
            self._shap_cache = (df, X_t, sv)
        return self._shap_cache

    def explain(self, code: str, top_n: int = 5) -> dict:
        df, X_t, sv = self._target_shap()
        pos = df.index[df["code"] == code][0]
        s = pd.Series(sv[pos], index=FEATURES)
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
        return dict(
            code=code, name=df.loc[pos, "name"], rank=int(df.loc[pos, "rank"]),
            p_podium=float(df.loc[pos, "proba_podium"]),
            factors=[dict(feature=f, value=float(X_t.loc[pos, f]), shap=float(v))
                     for f, v in top.items()])

    def global_importance(self) -> list[dict]:
        if self._importance is None:
            sv = shap.TreeExplainer(self.model).shap_values(self.X)
            self._importance = (pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)
                                .sort_values(ascending=False))
        return [dict(feature=f, mean_abs_shap=float(v)) for f, v in self._importance.items()]

    # ── what-if ──────────────────────────────────────────────────────────────
    def shifted_grid(self, code: str, grid_position: int) -> dict[str, int]:
        """Move `code` to `grid_position`; everyone else keeps relative order (a real grid stays 1..N)."""
        order = self._target.sort_values("GridPosition")["code"].tolist()
        order.remove(code)
        order.insert(grid_position - 1, code)
        return {c: i + 1 for i, c in enumerate(order)}

    def what_if(self, code: str, grid_position: int) -> dict:
        before = self.predict()
        after = self.predict(self.shifted_grid(code, grid_position))
        b = before[before["code"] == code].iloc[0]
        a = after[after["code"] == code].iloc[0]
        return dict(code=code, name=b["name"],
                    grid_before=int(b["GridPosition"]), grid_after=int(a["GridPosition"]),
                    p_before=float(b["proba_podium"]), p_after=float(a["proba_podium"]),
                    rank_before=int(b["rank"]), rank_after=int(a["rank"]))
