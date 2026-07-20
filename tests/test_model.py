import numpy as np
import pandas as pd
import pytest
from driver_similarity.model import compute_style_scores, run_pca_and_cluster, label_archetypes, render_html

FEATURE_COLS = [
    "qualifying_pace", "overtakes", "tire_preservation",
    "braking_style", "aggression", "wet_performance",
]


def _synthetic_drivers(n=12, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "driver": [f"D{i}" for i in range(n)],
        "qualifying_pace": rng.normal(1.0, 0.5, n),
        "overtakes": rng.normal(2.0, 1.0, n),
        "tire_preservation": rng.normal(0.1, 0.05, n),
        "brake_zone_count": rng.normal(10, 2, n),
        "mean_brake_duration": rng.normal(8, 1, n),
        "mean_brake_onset_speed": rng.normal(270, 10, n),
        "throttle_aggression": rng.normal(0.5, 0.1, n),
        "wet_performance": rng.normal(0.0, 1.0, n),
        "has_wet_data": [True] * n,
    })


def test_compute_style_scores_adds_composite_columns():
    df = _synthetic_drivers()
    result = compute_style_scores(df)
    assert "braking_style" in result.columns
    assert "aggression" in result.columns
    assert result["braking_style"].notna().all()
    assert result["aggression"].notna().all()


def test_run_pca_and_cluster_adds_expected_columns():
    df = compute_style_scores(_synthetic_drivers())
    result, explained_variance = run_pca_and_cluster(df, FEATURE_COLS)
    assert {"pc1", "pc2", "cluster"}.issubset(result.columns)
    assert result["cluster"].nunique() >= 2
    assert len(explained_variance) == 2
    assert all(0 <= v <= 1 for v in explained_variance)


def test_run_pca_and_cluster_imputes_nan_feature_for_sparse_driver(capsys):
    df = compute_style_scores(_synthetic_drivers())
    df.loc[df["driver"] == "D0", "tire_preservation"] = np.nan

    result, explained_variance = run_pca_and_cluster(df, FEATURE_COLS)

    assert result[["pc1", "pc2", "cluster"]].notna().all().all()
    assert len(result) == len(df)
    assert len(explained_variance) == 2

    captured = capsys.readouterr()
    assert "tire_preservation" in captured.out
    assert "D0" in captured.out


def test_label_archetypes_returns_one_label_per_cluster():
    df = compute_style_scores(_synthetic_drivers())
    clustered, _ = run_pca_and_cluster(df, FEATURE_COLS)
    labels = label_archetypes(clustered, FEATURE_COLS)
    assert set(labels.keys()) == set(clustered["cluster"].unique())
    assert all(isinstance(v, str) and len(v) > 0 for v in labels.values())


def test_render_html_includes_driver_names_and_cluster_captions():
    df = pd.DataFrame({
        "driver": ["VER", "HAM"],
        "pc1": [1.0, -1.0],
        "pc2": [0.5, -0.5],
        "cluster": [0, 1],
    })
    labels = {0: "Aggressive Overtaker", 1: "Smooth Operator"}

    html = render_html(df, labels)

    assert "VER" in html
    assert "HAM" in html
    assert "Aggressive Overtaker" in html
    assert "Smooth Operator" in html
    assert "<html" in html.lower()
    assert "http://" not in html and "https://" not in html  # self-contained, no external refs


def test_render_html_includes_qualitative_axis_framing():
    df = pd.DataFrame({
        "driver": ["VER", "HAM"],
        "pc1": [1.0, -1.0],
        "pc2": [0.5, -0.5],
        "cluster": [0, 1],
    })
    labels = {0: "Aggressive Overtaker", 1: "Smooth Operator"}

    html = render_html(df, labels)

    # A visible plot frame/border so the chart reads as axes, not a floating point cloud.
    assert "<rect" in html
    # A plain-language caption explaining what position/distance mean, without
    # fabricating specific per-axis semantic claims (PC1/PC2 meaning is data-dependent).
    assert "driving-style similarity" in html
    # No hardcoded literal claim about what a specific axis means (e.g. "PC1 = aggression").
    assert "PC1 = " not in html and "PC1=" not in html
