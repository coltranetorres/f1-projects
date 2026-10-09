import json

import pytest
from pydantic_ai import ModelRetry

from harness import tools


def test_get_prediction_top_n(predictor):
    res = tools.get_prediction(predictor, top_n=3)
    assert [r.rank for r in res.rows] == [1, 2, 3]
    assert res.target_round == 16 and "XGBoost" in res.source


@pytest.mark.parametrize("top_n", [0, 23])
def test_get_prediction_rejects_bad_top_n(predictor, top_n):
    with pytest.raises(ModelRetry, match="top_n"):
        tools.get_prediction(predictor, top_n=top_n)


def test_driver_code_is_normalised(predictor):  # Review Focus #1
    assert tools.explain_driver(predictor, " ver ").code == "VER"


def test_unknown_driver_lists_valid_codes(predictor):
    with pytest.raises(ModelRetry) as e:
        tools.explain_driver(predictor, "ZZZ")
    assert "Valid codes" in str(e.value) and "VER" in str(e.value)


def test_race_result_validates_round(predictor):
    assert tools.get_race_result(predictor, 15).rows[0].code == "RUS"
    with pytest.raises(ModelRetry, match="Completed rounds: 1-15"):
        tools.get_race_result(predictor, 30)


@pytest.mark.parametrize("pos", [0, 23, 40])
def test_what_if_rejects_bad_grid(predictor, pos):
    with pytest.raises(ModelRetry, match="1-22"):
        tools.what_if(predictor, "RUS", pos)


def test_what_if_reports_change(predictor):
    res = tools.what_if(predictor, "VER", 22)
    assert res.p_change == pytest.approx(res.p_after - res.p_before, abs=1e-3)
    assert res.p_change < 0


def test_driver_form_validates_last_n(predictor):
    assert len(tools.get_driver_form(predictor, "VER", last_n=3).rows) == 3
    with pytest.raises(ModelRetry, match="last_n"):
        tools.get_driver_form(predictor, "VER", last_n=0)


def test_all_tool_outputs_are_strict_json(predictor):  # Review Focus #4
    outputs = [
        tools.list_drivers(predictor), tools.get_prediction(predictor),
        tools.explain_driver(predictor, "NOR"), tools.global_importance(predictor),
        tools.get_driver_form(predictor, "ALO", last_n=15),
        tools.get_race_result(predictor, 1), tools.what_if(predictor, "HAM", 5),
        tools.model_info(predictor),
    ]
    for out in outputs:
        json.dumps(out.model_dump(), allow_nan=False)  # raises on NaN/inf
