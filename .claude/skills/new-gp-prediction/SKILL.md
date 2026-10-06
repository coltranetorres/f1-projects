---
name: new-gp-prediction
description: Use when setting up a new GP directory to predict the podium for an upcoming F1 Grand Prix in this project — covers fetching completed race data, adding qualifying data, running the model, and generating the podium graphic.
---

# Workflow for a New Race Prediction

### Step 1 — Set up the GP directory

Create a new directory for the target GP (e.g. `azerbaijan/` for the round after Monza). Copy and adapt `fetch_data.py` and `model.py` from the previous GP's directory. Key things to update:

- `fetch_data.py`: update the round cutoff — fetch all completed races **before** the target GP
- `model.py`: update `RACES` list to include all completed rounds, update `SPRINT_ROUNDS`, update the qualifying CSV filename for the new GP

### Step 2 — Fetch completed race data

Run `fetch_data.py` to pull results and laps for all completed races via FastF1. The FastF1 cache in `{gp}/cache/` prevents redundant API calls across runs.

If FastF1 doesn't yet have the data for the most recent race (API lag), manually create the results CSV from official race results — see CSV Schema in CLAUDE.md.

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

### Step 5 — Generate the podium graphic (optional)

Once `model.py` prints the predicted podium, build `{gp}/podium.html` — a static, self-contained IG Reel graphic (portrait 540×960 CSS px, i.e. 1080×1920 at full scale — a true 9:16 frame). Copy an existing one (e.g. `hungary/podium.html` or `monza/podium.html`) as the template and update: eyebrow round/circuit/year, header title, the three podium cards (driver, team, car number, grid position, qualifying time, P(podium)%, team-colored badge/card accent), and the footer. Team color tokens (`team-mercedes`, `team-ferrari`, `team-mclaren`, etc.) accumulate across GP directories — check a recent one before inventing new hex values for a team already styled elsewhere.
