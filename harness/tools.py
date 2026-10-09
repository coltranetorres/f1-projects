"""Read-only tools over f1_core. Pure functions: (predictor, validated args) -> schema."""
from pydantic_ai import ModelRetry

from f1_core import Predictor
from f1_core.data import QUALI_FILE, RACE_NAMES, TARGET_ROUND

from .schemas import (DriverRef, DriversResult, ExplainResult, FormResult, FormRow, ImportanceResult,
                      ImportanceRow, ModelInfoResult, PredictionResult, PredictionRow, RaceResult,
                      ResultRow, ShapFactor, WhatIfResult)

SRC_MODEL = "f1_core.Predictor (XGBoost retrained on all completed 2026 rounds)"
SRC_QUALI = f"data_pred/data/{QUALI_FILE}"


def _r(x: float) -> float:
    return round(float(x), 4)


def _code(p: Predictor, driver: str) -> str:
    code = driver.strip().upper()
    valid = [d["code"] for d in p.drivers()]
    if code not in valid:
        raise ModelRetry(f"Unknown driver code '{driver}'. Valid codes: {', '.join(valid)}.")
    return code


def list_drivers(p: Predictor) -> DriversResult:
    return DriversResult(drivers=[DriverRef(**d) for d in p.drivers()], source=SRC_QUALI)


def get_prediction(p: Predictor, top_n: int = 22) -> PredictionResult:
    n = len(p.drivers())
    if not 1 <= top_n <= n:
        raise ModelRetry(f"top_n must be between 1 and {n}, got {top_n}.")
    ranked = p.predict().head(top_n)
    rows = [PredictionRow(rank=int(r["rank"]), code=r["code"], name=r["name"], team=r["team"],
                          grid=int(r["GridPosition"]), p_podium=_r(r["proba_podium"]))
            for _, r in ranked.iterrows()]
    return PredictionResult(target_round=TARGET_ROUND, rows=rows, source=SRC_MODEL)


def explain_driver(p: Predictor, driver: str) -> ExplainResult:
    out = p.explain(_code(p, driver))
    return ExplainResult(
        code=out["code"], name=out["name"], rank=out["rank"], p_podium=_r(out["p_podium"]),
        factors=[ShapFactor(feature=f["feature"], value=_r(f["value"]), shap=_r(f["shap"]))
                 for f in out["factors"]],
        source=SRC_MODEL + " + SHAP TreeExplainer")


def global_importance(p: Predictor) -> ImportanceResult:
    rows = [ImportanceRow(feature=r["feature"], mean_abs_shap=_r(r["mean_abs_shap"]))
            for r in p.global_importance()]
    return ImportanceResult(rows=rows, source="SHAP TreeExplainer on the full training set")


def get_driver_form(p: Predictor, driver: str, last_n: int = 5) -> FormResult:
    rounds = p.completed_rounds()
    if not 1 <= last_n <= len(rounds):
        raise ModelRetry(f"last_n must be between 1 and {len(rounds)}, got {last_n}.")
    code = _code(p, driver)
    return FormResult(code=code, rows=[FormRow(**r) for r in p.driver_form(code, last_n)],
                      source="data_pred/data/R*_results.csv")


def get_race_result(p: Predictor, round_number: int) -> RaceResult:
    rounds = p.completed_rounds()
    if round_number not in rounds:
        raise ModelRetry(f"Round {round_number} has no result. Completed rounds: {rounds[0]}-{rounds[-1]}.")
    rows = [ResultRow(**{**r, "points": _r(r["points"])}) for r in p.race_result(round_number)]
    return RaceResult(round=round_number, race=RACE_NAMES[round_number], rows=rows,
                      source=f"data_pred/data/R{round_number:02d}_*_results.csv")


def what_if(p: Predictor, driver: str, grid_position: int) -> WhatIfResult:
    n = len(p.drivers())
    if not 1 <= grid_position <= n:
        raise ModelRetry(f"grid_position must be 1-{n}, got {grid_position}.")
    out = p.what_if(_code(p, driver), grid_position)
    return WhatIfResult(
        code=out["code"], name=out["name"], grid_before=out["grid_before"], grid_after=out["grid_after"],
        p_before=_r(out["p_before"]), p_after=_r(out["p_after"]),
        p_change=_r(out["p_after"] - out["p_before"]),
        rank_before=out["rank_before"], rank_after=out["rank_after"],
        source=SRC_MODEL + " (driver moved, others keep relative order)")


def model_info(p: Predictor) -> ModelInfoResult:
    info = p.model_info()
    return ModelInfoResult(**{**info, "test_accuracy": _r(info["test_accuracy"]),
                              "test_auc": _r(info["test_auc"])}, source=SRC_MODEL)
