import pandas as pd
import pytest

from f1_core import Predictor
from f1_core.data import DATA_DIR, QUALI_FILE


def test_drivers_sorted_by_grid(predictor):
    drivers = predictor.drivers()
    assert [d["grid"] for d in drivers] == list(range(1, 23))
    assert drivers[0]["code"] == "VER"  # pole in the Sepang qualifying CSV


def test_completed_rounds(predictor):
    assert predictor.completed_rounds() == list(range(1, 16))


def test_driver_form_ground_truth(predictor):
    rows = predictor.driver_form("VER", last_n=3)
    assert [r["round"] for r in rows] == [13, 14, 15]
    last = rows[-1]
    assert (last["grid"], last["finish"], last["status"]) == (8, 2, "Finished")


def test_race_result_ground_truth(predictor):
    rows = predictor.race_result(15)
    assert len(rows) == 22
    assert (rows[0]["code"], rows[0]["finish"], rows[0]["points"]) == ("RUS", 1, 25.0)


def test_explain_returns_sorted_factors(predictor):
    out = predictor.explain("VER")
    assert out["code"] == "VER" and 0 <= out["p_podium"] <= 1
    shaps = [abs(f["shap"]) for f in out["factors"]]
    assert len(shaps) == 5 and shaps == sorted(shaps, reverse=True)


def test_global_importance_sorted(predictor):
    rows = predictor.global_importance()
    assert len(rows) == 22
    vals = [r["mean_abs_shap"] for r in rows]
    assert vals == sorted(vals, reverse=True)


def test_shifted_grid_is_a_permutation(predictor):
    grid = predictor.shifted_grid("RUS", 1)
    assert sorted(grid.values()) == list(range(1, 23))
    assert grid["RUS"] == 1 and grid["VER"] == 2  # pole driver pushed back one slot


def test_what_if_back_of_grid_lowers_probability(predictor):
    out = predictor.what_if("VER", 22)
    assert out["grid_before"] == 1 and out["grid_after"] == 22
    assert out["p_after"] < out["p_before"]
    assert out["rank_after"] > out["rank_before"]


def test_what_if_current_slot_is_a_noop(predictor):  # Review Focus #3
    out = predictor.what_if("VER", 1)
    assert out["p_after"] == pytest.approx(out["p_before"])
    assert out["rank_after"] == out["rank_before"]


def test_model_info(predictor):
    info = predictor.model_info()
    assert info["n_rows"] == 330 and info["n_features"] == 22
    assert info["rounds_covered"] == list(range(1, 16))
    assert 0 < info["test_accuracy"] <= 1 and 0 < info["test_auc"] <= 1


def test_unknown_car_number_does_not_crash(tmp_path):  # Review Focus #2
    quali = pd.read_csv(DATA_DIR / QUALI_FILE)
    quali.loc[quali["Car Number"] == 3, "Car Number"] = 99  # a number never seen in training
    path = tmp_path / "quali.csv"
    quali.to_csv(path, index=False)
    ranked = Predictor(quali_path=path).predict()
    assert len(ranked) == 22
    assert "MVE" in set(ranked["code"])  # fallback code from "M. Verstappen"
    assert ranked["proba_podium"].notna().all()
