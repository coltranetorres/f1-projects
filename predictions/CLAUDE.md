# predictions/

Podium (top 3) prediction for upcoming 2026 F1 Grand Prix races using an XGBoost binary classifier trained on all completed races of the season. Each cycle: fetch new race data → update training set → retrain → predict the next race from qualifying results.

Each GP directory (`monaco/`, `hungary/`, `singapore/`, ...) is a point-in-time snapshot containing `fetch_data.py`, `model.py`, `cache/` (gitignored FastF1 cache) and, once predicted, `podium.html`. Copy-and-adapt the latest GP directory for a new race; there is no shared library here. (`ai_assistant/backend/f1_core/` is a separate extraction of the Sepang script, not something these scripts import.)

## Commands

Run from the repo root:

```bash
uv run python predictions/{gp}/fetch_data.py   # fetch race data from FastF1 API → predictions/data_pred/data/
uv run python predictions/{gp}/model.py        # train model + print podium prediction
```

Race and qualifying CSVs for all rounds live in the shared `data_pred/data/`. Scripts locate it with `Path(__file__).parent.parent / "data_pred" / "data"`, so a GP directory must sit directly under `predictions/`.

## Workflow for a New Race Prediction

See the `new-gp-prediction` skill for the full 5-step workflow (set up the GP directory, fetch completed race data, add qualifying data, run the model, generate the podium graphic).

## Architecture

### Data Pipeline (`fetch_data.py`)

All `timedelta64` columns are cast to string before saving to avoid serialisation issues.

### Model Pipeline (`model.py`)

Only accurate laps (`IsAccurate == True`) are used for lap aggregation. Non-sprint rounds get `sprint_pos=22`, `sprint_points=0`, `sprint_grid=22`, `has_sprint=0`.

`LabelEncoder` is fit on the full training dataset each run — not serialised, so encodings are only stable within a single execution.

**XGBoost config** — `scale_pos_weight = n_non_podium / n_podium` (~6.33 — stays essentially constant as races are added, since every race contributes 3 podium rows out of a ~22-driver field). The best iteration from the 80/20 split is reused when retraining `model_full` on the full dataset.

**Target GP prediction** — per-driver features are averaged across all training races to represent season form. Known values are then overridden: `GridPosition` = actual qualifying position, `positions_gained = 0`, `finished = 1`, sprint fields set based on whether target race is a sprint weekend.

### CSV Schema

Results files (FastF1 format) — see any `*_results.csv` header for the exact column list (e.g. `data_pred/data/R05_Canadian_Grand_Prix_results.csv`). Notes on the non-obvious columns:

- `Time`: `0 days HH:MM:SS.ffffff` — full race time for winner, gap to winner for finishers, `NaT` for DNF
- `ClassifiedPosition`: position number for finishers/lapped, `R` for retired, `W` for withdrew
- `Q1/Q2/Q3`: always `NaT` in race result files
- `Status`: `Finished`, `Lapped`, or `Retired`
- Sprint result files use the same schema

When creating a manual results CSV, use existing files (e.g. `data_pred/data/R05_Canadian_Grand_Prix_results.csv`) as a reference for team colours, driver IDs, and headshot URL patterns.

## Current Season State (2026)

- Completed races: R01 Australia, R02 China (sprint), R03 Japan, R04 Miami (sprint), R05 Canada (sprint), R06 Monaco, R07 Spain (Barcelona), R08 Austria, R09 Britain (sprint), R10 Belgium, R11 Hungary, R12 Netherlands (Dutch GP), R13 Italy (Monza), R14 Spain (Madrid/"madring"), R15 Azerbaijan, R16 Malaysia (Sepang)
- Sprint rounds: {2, 4, 5, 9} (R17 Singapore is also a sprint weekend; its sprint results are used as known features for the prediction)
- Next target: Singapore GP (Round 17) — qualifying and sprint data already in `data_pred/data/`; prediction in `singapore/`
- Training set grows by 22 rows with each completed race (352 rows as of R16); model quality improves as dataset expands
