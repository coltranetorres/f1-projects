from pydantic import BaseModel


class DriverRef(BaseModel):
    code: str
    name: str
    team: str
    grid: int


class DriversResult(BaseModel):
    drivers: list[DriverRef]
    source: str


class PredictionRow(BaseModel):
    rank: int
    code: str
    name: str
    team: str
    grid: int
    p_podium: float


class PredictionResult(BaseModel):
    target_round: int
    rows: list[PredictionRow]
    source: str


class ShapFactor(BaseModel):
    feature: str
    value: float
    shap: float


class ExplainResult(BaseModel):
    code: str
    name: str
    rank: int
    p_podium: float
    factors: list[ShapFactor]
    source: str


class ImportanceRow(BaseModel):
    feature: str
    mean_abs_shap: float


class ImportanceResult(BaseModel):
    rows: list[ImportanceRow]
    source: str


class FormRow(BaseModel):
    round: int
    race: str
    grid: int | None
    finish: int | None
    positions_gained: int | None
    status: str


class FormResult(BaseModel):
    code: str
    rows: list[FormRow]
    source: str


class ResultRow(BaseModel):
    finish: int | None
    code: str
    name: str
    team: str
    grid: int | None
    status: str
    points: float


class RaceResult(BaseModel):
    round: int
    race: str
    rows: list[ResultRow]
    source: str


class WhatIfResult(BaseModel):
    code: str
    name: str
    grid_before: int
    grid_after: int
    p_before: float
    p_after: float
    p_change: float
    rank_before: int
    rank_after: int
    source: str


class ModelInfoResult(BaseModel):
    n_rows: int
    n_features: int
    rounds_covered: list[int]
    best_iteration: int
    test_accuracy: float
    test_auc: float
    source: str
