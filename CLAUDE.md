# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict the podium (top 3 finishers) for upcoming F1 Grand Prix races using an XGBoost binary classifier trained on all completed races in the 2026 season. Each prediction cycle follows the same workflow: fetch new race data → update training set → retrain model → predict next race using qualifying results.

## Commands

Dependencies are managed with `uv` (Python >= 3.13):

```bash
uv sync                            # install all dependencies
uv run python {gp}/fetch_data.py   # fetch race data from FastF1 API → data_pred/data/
uv run python {gp}/model.py        # train model + print podium prediction
```

Race and qualifying CSVs for all rounds live in the shared `data_pred/data/`. Each target GP has its own directory (e.g. `monaco/`, `hungary/`, `monza/`, etc.) containing `fetch_data.py`, `model.py`, `cache/`, and (once a prediction has been made) `podium.html`.

## Workflow for a New Race Prediction

See the `new-gp-prediction` skill for the full 5-step workflow (set up the GP directory, fetch completed race data, add qualifying data, run the model, generate the podium graphic).

## Architecture

### Data Pipeline (`fetch_data.py`)

Fetches all rounds before the target GP and saves per race:
- `R{rn:02d}_{Event_Name}_results.csv` — one row per driver
- `R{rn:02d}_{Event_Name}_laps.csv` — per-lap telemetry
- Sprint weekends additionally save `_sprint_results.csv` and `_sprint_laps.csv`

All `timedelta64` columns are cast to string before saving to avoid serialisation issues.

### Model Pipeline (`model.py`)

**Feature engineering** — 22 features per driver per race across Results, Lap aggregation, and Sprint groups (see the `FEATURES` list in `model.py` for the exact set).

Only accurate laps (`IsAccurate == True`) are used for lap aggregation. Non-sprint rounds get `sprint_pos=22`, `sprint_points=0`, `sprint_grid=22`, `has_sprint=0`.

`LabelEncoder` is fit on the full training dataset each run — not serialised, so encodings are only stable within a single execution.

**XGBoost config** — `binary:logistic`, `scale_pos_weight = n_non_podium / n_podium` (~6.33 — stays essentially constant as races are added, since every race contributes 3 podium rows out of a ~22-driver field), `max_depth=3`, `learning_rate=0.05`, early stopping on logloss. The best iteration from the 80/20 split is reused when retraining `model_full` on the full dataset.

**Target GP prediction** — per-driver features are averaged across all training races to represent season form. Known values are then overridden: `GridPosition` = actual qualifying position, `positions_gained = 0`, `finished = 1`, sprint fields set based on whether target race is a sprint weekend.

**SHAP** — `shap.TreeExplainer` provides global feature importance and per-driver breakdowns. As of R12 (264 training rows), `GridPosition` dominates (~4× the impact of the next feature, `positions_gained`), with `finished` and `avg_race_position` also contributing meaningfully. As predicted, several features that were near-zero on the original 5-race dataset have activated now that training data has grown well past 100 rows: `sprint_pos`, `mean_lap_s`, and `team_encoded` now carry real SHAP weight. `num_stints`, `mean_s2_s`, and `avg_speed_i2` remain the weakest contributors.

### CSV Schema

Results files (FastF1 format) — see any `*_results.csv` header for the exact column list (e.g. `data_pred/data/R05_Canadian_Grand_Prix_results.csv`). Notes on the non-obvious columns:

- `Time`: `0 days HH:MM:SS.ffffff` — full race time for winner, gap to winner for finishers, `NaT` for DNF
- `ClassifiedPosition`: position number for finishers/lapped, `R` for retired, `W` for withdrew
- `Q1/Q2/Q3`: always `NaT` in race result files
- `Status`: `Finished`, `Lapped`, or `Retired`
- Sprint result files use the same schema

When creating a manual results CSV, use existing files (e.g. `data_pred/data/R05_Canadian_Grand_Prix_results.csv`) as a reference for team colours, driver IDs, and headshot URL patterns.

## Current Season State (2026)

- Completed races: R01 Australia, R02 China (sprint), R03 Japan, R04 Miami (sprint), R05 Canada (sprint), R06 Monaco, R07 Spain (Barcelona), R08 Austria, R09 Britain (sprint), R10 Belgium, R11 Hungary, R12 Netherlands (Dutch GP), R13 Italy (Monza), R14 Spain (Madrid/"madring"), R15 Azerbaijan
- Sprint rounds: {2, 4, 5, 9}
- Next target: Malaysian GP / Sepang (Round 16, FastF1 schedule name "Bahrain Grand Prix") — qualifying data already in `data_pred/data/`
- Training set grows by 22 rows with each completed race (330 rows as of R15); model quality improves as dataset expands

## Agent Harness (PoC)

Read-only Q&A assistant over the Sepang prediction. Spec: `docs/superpowers/specs/2026-10-09-f1-harness-poc-design.md` (local only, gitignored).

- `f1_core/` — importable pipeline extracted from `sepang/model.py` (`Predictor`); regression-tested against the script's output in `tests/fixtures/`.
- `harness/` — Pydantic AI agent, typed read-only tools, guardrails, Langfuse tracing, `evals/` (seed cases, evaluators, `taxonomy.yaml`).
- `api/` FastAPI (`uv run uvicorn api.main:app --port 8000`), `web/` React (`cd web && npm install` (first time), then `npm run dev`).
- Evals: `uv run python -m harness.evals.run --model <openrouter-id> [--suite S] [--split dev|heldout|all] [--limit N]`.
- Manual check: `uv run python -m harness.chat "question"`.
- Secrets live in `.env` (gitignored; see `.env.example`). XGBoost needs `brew install libomp` on macOS.
- Failure modes in `harness/evals/taxonomy.yaml` come only from inspected Langfuse traces.
