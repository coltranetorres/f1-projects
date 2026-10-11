import re
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent / "fixtures" / "sepang_model_output.txt"
RANK_RE = re.compile(r"^\s+(\d+)\s+(.+?)\s+(\d+)\s+(?:[\d.]+s|N/A)\s+(\d\.\d{4})")


def parse_baseline():
    text = BASELINE.read_text()
    section = text.split("Full driver ranking by podium probability:")[1].split("SHAP BREAKDOWN")[0]
    out = []
    for line in section.splitlines():
        m = RANK_RE.match(line)
        if m:
            out.append((int(m[1]), m[2], int(m[3]), float(m[4])))
    return out


def test_predict_matches_original_script(predictor):
    baseline = parse_baseline()
    ranked = predictor.predict()
    assert len(baseline) == len(ranked) == 22
    for (rank, label, grid, prob), (_, row) in zip(baseline, ranked.iterrows()):
        assert row["rank"] == rank
        assert row["name"] in label
        assert int(row["GridPosition"]) == grid
        assert row["proba_podium"] == pytest.approx(prob, abs=1e-4)


def test_predict_is_sorted_and_complete(predictor):
    ranked = predictor.predict()
    assert ranked["proba_podium"].is_monotonic_decreasing
    assert ranked["code"].is_unique
    assert ranked["proba_podium"].between(0, 1).all()
