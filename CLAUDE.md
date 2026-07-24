# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict the podium (top 3 finishers) for upcoming F1 Grand Prix races using an XGBoost binary classifier trained on all completed races in the 2026 season. Each prediction cycle follows the same workflow: fetch new race data → update training set → retrain model → predict next race using qualifying results.

## Commands

Dependencies are managed with `uv` (Python >= 3.13):

```bash
uv sync                            # install all dependencies
uv run python {gp}/fetch_data.py   # fetch race data from FastF1 API → {gp}/data/
uv run python {gp}/model.py        # train model + print podium prediction
```

Each target GP has its own directory (e.g. `monaco/`, and future `spain/`, etc.) containing `fetch_data.py`, `model.py`, `data/`, and `cache/`.

## Workflow for a New Race Prediction

### Step 1 — Set up the GP directory

Create a new directory for the target GP (e.g. `spain/`). Copy and adapt `fetch_data.py` and `model.py` from the previous GP's directory. Key things to update:

- `fetch_data.py`: update the round cutoff — fetch all completed races **before** the target GP
- `model.py`: update `RACES` list to include all completed rounds, update `SPRINT_ROUNDS`, update the qualifying CSV filename for the new GP

### Step 2 — Fetch completed race data

Run `fetch_data.py` to pull results and laps for all completed races via FastF1. The FastF1 cache in `{gp}/cache/` prevents redundant API calls across runs.

If FastF1 doesn't yet have the data for the most recent race (API lag), manually create the results CSV from official race results — see CSV Schema below.

### Step 3 — Add qualifying data

Before the target race, manually create `R{rn:02d}_{GP_Name}_Qualifying_results.csv` in `{gp}/data/` with columns:
```
Position, Driver, Nationality, Team, Car Number, Qualifying Time
```
`Qualifying Time` format: `M:SS.mmm` (e.g. `1:12.051`). This is used to override `GridPosition` in the prediction — qualifying grid position is the strongest single predictor at street circuits like Monaco.

### Step 4 — Run the model

`model.py` trains on all completed races, then retrains on the full dataset and predicts the target GP podium. It outputs:
- Evaluation metrics (accuracy, ROC-AUC, overfitting gap)
- Full driver ranking by P(podium)
- SHAP breakdown for top 4 contenders

## Architecture

### Data Pipeline (`fetch_data.py`)

Fetches all rounds before the target GP and saves per race:
- `R{rn:02d}_{Event_Name}_results.csv` — one row per driver
- `R{rn:02d}_{Event_Name}_laps.csv` — per-lap telemetry
- Sprint weekends additionally save `_sprint_results.csv` and `_sprint_laps.csv`

All `timedelta64` columns are cast to string before saving to avoid serialisation issues.

### Model Pipeline (`model.py`)

**Feature engineering** — 22 features per driver per race:

| Group | Features |
|---|---|
| Results | `GridPosition`, `positions_gained`, `finished`, `team_encoded`, `driver_encoded` |
| Lap aggregation | `mean_lap_s`, `best_lap_s`, `lap_consistency`, `mean_s1_s`, `mean_s2_s`, `mean_s3_s`, `avg_speed_i1`, `avg_speed_i2`, `avg_speed_fl`, `max_speed_st`, `personal_best_count`, `avg_race_position`, `num_stints` |
| Sprint | `sprint_pos`, `sprint_points`, `sprint_grid`, `has_sprint` |

Only accurate laps (`IsAccurate == True`) are used for lap aggregation. Non-sprint rounds get `sprint_pos=22`, `sprint_points=0`, `sprint_grid=22`, `has_sprint=0`.

`LabelEncoder` is fit on the full training dataset each run — not serialised, so encodings are only stable within a single execution.

**XGBoost config** — `binary:logistic`, `scale_pos_weight = n_non_podium / n_podium` (currently ~6.3 for 5 races; decreases as more races are added), `max_depth=3`, `learning_rate=0.05`, early stopping on logloss. The best iteration from the 80/20 split is reused when retraining `model_full` on the full dataset.

**Target GP prediction** — per-driver features are averaged across all training races to represent season form. Known values are then overridden: `GridPosition` = actual qualifying position, `positions_gained = 0`, `finished = 1`, sprint fields set based on whether target race is a sprint weekend.

**SHAP** — `shap.TreeExplainer` provides global feature importance and per-driver breakdowns. Currently `avg_race_position` dominates (~5× the impact of the next feature). Sprint features, `mean_lap_s`, `mean_s2_s`, `avg_speed_i2`, `num_stints`, and `team_encoded` currently contribute near-zero SHAP — likely to activate as dataset grows beyond ~100 rows.

### CSV Schema

Results files (FastF1 format):
```
DriverNumber, BroadcastName, Abbreviation, DriverId, TeamName, TeamColor, TeamId,
FirstName, LastName, FullName, HeadshotUrl, CountryCode,
Position, ClassifiedPosition, GridPosition, Q1, Q2, Q3, Time, Status, Points, Laps
```

- `Time`: `0 days HH:MM:SS.ffffff` — full race time for winner, gap to winner for finishers, `NaT` for DNF
- `ClassifiedPosition`: position number for finishers/lapped, `R` for retired, `W` for withdrew
- `Q1/Q2/Q3`: always `NaT` in race result files
- `Status`: `Finished`, `Lapped`, or `Retired`
- Sprint result files use the same schema

When creating a manual results CSV, use existing files (e.g. `monaco/data/R05_Canadian_Grand_Prix_results.csv`) as a reference for team colours, driver IDs, and headshot URL patterns.

## Current Season State (2026)

- Completed races: R01 Australia, R02 China (sprint), R03 Japan, R04 Miami (sprint), R05 Canada (sprint), R06 Monaco
- Sprint rounds: {2, 4, 5}
- Next target: Spain GP (Round 7)
- Training set grows by 22 rows with each completed race; model quality improves as dataset expands