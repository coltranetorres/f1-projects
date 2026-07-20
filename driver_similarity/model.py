from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

FEATURE_COLS = [
    "qualifying_pace", "overtakes", "tire_preservation",
    "braking_style", "aggression", "wet_performance",
]

ARCHETYPE_RULES = {
    ("aggression", "high"): "Aggressive Overtaker",
    ("aggression", "low"): "Smooth Operator",
    ("tire_preservation", "high"): "Tire Manager",
    ("tire_preservation", "low"): "Flat-Out Racer",
    ("qualifying_pace", "low"): "Qualifying Specialist",
    ("wet_performance", "high"): "Wet-Weather Ace",
    ("braking_style", "high"): "Late Braker",
    ("braking_style", "low"): "Early Braker",
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std


def compute_style_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["braking_style"] = _zscore(out["brake_zone_count"]) + _zscore(out["mean_brake_duration"])
    out["aggression"] = (
        _zscore(out["overtakes"]) + _zscore(out["mean_brake_onset_speed"]) + _zscore(out["throttle_aggression"])
    )
    return out


def run_pca_and_cluster(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, tuple[float, float]]:
    out = df.copy()
    x = StandardScaler().fit_transform(out[feature_cols])

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(x)
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]

    best_k, best_score, best_labels = None, -1.0, None
    max_k = min(6, len(out) - 1)
    for k in range(3, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    out["cluster"] = best_labels if best_labels is not None else np.zeros(len(out), dtype=int)
    explained_variance = tuple(float(v) for v in pca.explained_variance_ratio_)
    return out, explained_variance


def label_archetypes(df: pd.DataFrame, feature_cols: list[str]) -> dict[int, str]:
    z = df[feature_cols].apply(_zscore)
    z["cluster"] = df["cluster"].values
    centroids = z.groupby("cluster")[feature_cols].mean()

    labels = {}
    for cluster_id, row in centroids.iterrows():
        strongest_feature = row.abs().idxmax()
        direction = "high" if row[strongest_feature] > 0 else "low"
        labels[cluster_id] = ARCHETYPE_RULES.get(
            (strongest_feature, direction), f"Style Group {cluster_id}"
        )
    return labels
