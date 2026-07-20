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

    for col in feature_cols:
        nan_mask = out[col].isna()
        if nan_mask.any():
            col_mean = out[col].mean(skipna=True)
            fill_value = col_mean if pd.notna(col_mean) else 0.0
            drivers = out.loc[nan_mask, "driver"].tolist() if "driver" in out.columns else nan_mask.sum()
            print(f"Imputed NaN in {col} for drivers: {drivers}")
            out[col] = out[col].fillna(fill_value)

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


def render_html(df: pd.DataFrame, archetype_labels: dict[int, str]) -> str:
    x_min, x_max = df["pc1"].min(), df["pc1"].max()
    y_min, y_max = df["pc2"].min(), df["pc2"].max()
    x_pad = (x_max - x_min) * 0.15 or 1.0
    y_pad = (y_max - y_min) * 0.15 or 1.0

    width, height = 900, 600

    def scale_x(v):
        return 60 + (v - (x_min - x_pad)) / ((x_max + x_pad) - (x_min - x_pad)) * (width - 120)

    def scale_y(v):
        return 60 + (1 - (v - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad))) * (height - 120)

    palette = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec", "#ffbe0b"]
    clusters = sorted(df["cluster"].unique())
    color_by_cluster = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}

    points_svg = []
    for _, row in df.iterrows():
        cx, cy = scale_x(row["pc1"]), scale_y(row["pc2"])
        color = color_by_cluster[row["cluster"]]
        points_svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{color}" stroke="#111" stroke-width="1"/>'
            f'<text x="{cx:.1f}" y="{cy - 12:.1f}" font-size="12" text-anchor="middle" fill="#111">{row["driver"]}</text>'
        )

    frame_left, frame_top = 60, 60
    frame_right, frame_bottom = width - 60, height - 60
    axis_svg = (
        f'<rect x="{frame_left}" y="{frame_top}" width="{frame_right - frame_left}" '
        f'height="{frame_bottom - frame_top}" fill="none" stroke="#999" stroke-width="1"/>'
        f'<line x1="{frame_left}" y1="{frame_bottom}" x2="{frame_right}" y2="{frame_bottom}" stroke="#999" stroke-width="1"/>'
        f'<line x1="{frame_left}" y1="{frame_top}" x2="{frame_left}" y2="{frame_bottom}" stroke="#999" stroke-width="1"/>'
        f'<text x="{(frame_left + frame_right) / 2:.1f}" y="{frame_bottom + 32:.1f}" font-size="12" '
        f'text-anchor="middle" fill="#666">Horizontal spread: strongest style difference (PC1)</text>'
        f'<text x="{frame_left - 40:.1f}" y="{(frame_top + frame_bottom) / 2:.1f}" font-size="12" '
        f'text-anchor="middle" fill="#666" transform="rotate(-90 {frame_left - 40:.1f} '
        f'{(frame_top + frame_bottom) / 2:.1f})">Vertical spread: second-strongest style difference (PC2)</text>'
    )

    captions = []
    for cluster_id in clusters:
        drivers_in_cluster = df.loc[df["cluster"] == cluster_id, "driver"].tolist()
        label = archetype_labels.get(cluster_id, f"Style Group {cluster_id}")
        color = color_by_cluster[cluster_id]
        captions.append(
            f'<div style="margin-bottom:12px;">'
            f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
            f'border-radius:50%;margin-right:8px;"></span>'
            f'<strong>{label}</strong> &mdash; {", ".join(drivers_in_cluster)}'
            f'</div>'
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>2026 Driver Similarity Map</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #111; background: #fff; }}
  h1 {{ font-size: 24px; }}
  p.subtitle {{ color: #555; max-width: 700px; }}
  .caption {{ font-size: 14px; }}
</style>
</head>
<body>
<h1>2026 Driver Similarity Map</h1>
<p class="subtitle">Drivers positioned by driving style across qualifying pace, overtaking, tire management, braking, aggression, and wet-weather performance. Drivers placed close together race in a similar way.</p>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  {axis_svg}
  {''.join(points_svg)}
</svg>
<p class="caption">Position reflects overall driving-style similarity: drivers placed close together race in a similar way. The horizontal and vertical axes each capture one of the two strongest style differences across the grid, but do not correspond to a single named trait &mdash; read distance between drivers, not direction, as the meaningful signal.</p>
<h2>Style groups</h2>
<div class="caption">
{''.join(captions)}
</div>
</body>
</html>"""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "driver_features.csv")
    df = compute_style_scores(df)
    df, explained_variance = run_pca_and_cluster(df, FEATURE_COLS)
    print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    labels = label_archetypes(df, FEATURE_COLS)
    for cluster_id, label in labels.items():
        drivers = df.loc[df["cluster"] == cluster_id, "driver"].tolist()
        print(f"Cluster {cluster_id} ({label}): {', '.join(drivers)}")

    html = render_html(df, labels)
    (OUTPUT_DIR / "driver_similarity.html").write_text(html, encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'driver_similarity.html'}")


if __name__ == "__main__":
    main()
