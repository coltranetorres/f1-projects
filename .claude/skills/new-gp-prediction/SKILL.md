---
name: new-gp-prediction
description: Use when setting up a new GP directory to predict the podium for an upcoming F1 Grand Prix in this project — covers fetching completed race data, adding qualifying data, running the model, and generating the podium graphic.
---

# Workflow for a New Race Prediction

### Step 1 — Set up the GP directory

Create a new directory for the target GP (e.g. `azerbaijan/` for the round after Monza). Copy and adapt `fetch_data.py` and `model.py` from the previous GP's directory. Key things to update:

- `fetch_data.py`: update the round cutoff — fetch all completed races **before** the target GP
- Both scripts: set `DATA_DIR = Path(__file__).parent.parent / "data_pred" / "data"` — race and qualifying CSVs for all GPs live in the shared `data_pred/data/` folder, not per-GP `data/` dirs
- `model.py`: update `RACES` list to include all completed rounds, update `SPRINT_ROUNDS`, update the qualifying CSV filename for the new GP

### Step 2 — Fetch completed race data

Run `fetch_data.py` to pull results and laps for all completed races via FastF1. The FastF1 cache in `{gp}/cache/` prevents redundant API calls across runs.

After it runs, confirm both `R{rn:02d}_*_results.csv` and `R{rn:02d}_*_laps.csv` (the middle is FastF1's event name, e.g. `R07_Barcelona_Grand_Prix`, so use a wildcard) exist in `data_pred/data/` for every completed round (sprint rounds also need `_sprint_` pairs). If a file is missing, or the log shows `ERROR loading race` for the latest round (API lag), manually create the results CSV from official race results — see CSV Schema in CLAUDE.md. A missing laps file cannot be hand-made: wait for FastF1 to publish it.

### Step 3 — Add qualifying data

Before the target race, manually create `R{rn:02d}_{GP_Name}_Qualifying_results.csv` in `data_pred/data/`. `{GP_Name}` is the **FastF1 schedule event name**, not the circuit or common name (Malaysian GP / Sepang is `R16_Bahrain_Grand_Prix_...`). It must match the filename `model.py` reads — check the `pd.read_csv(... Qualifying_results.csv)` line. Columns:
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

Once `model.py` prints the predicted podium, produce `{gp}/podium.html` — a static, self-contained IG Reel graphic (portrait 540×960 CSS px, i.e. 1080×1920 at full scale — a true 9:16 frame).

- **If `{gp}/podium.html` already exists:** compare its three cards (driver, car number, grid, qualifying time, P(podium)%) and the P4 strip against the fresh model output. Leave it unchanged when all match; otherwise update only the fields that differ.
- **If it does not exist:** copy an existing one (e.g. `hungary/podium.html` or `monza/podium.html`) as the template and update: eyebrow round/circuit/year, header title, the three podium cards (driver, team, car number, grid position, qualifying time, P(podium)%, team-colored badge/card accent), and the footer.

Team color tokens (`.team-mercedes`, `.team-ferrari`, `.team-mclaren`, `.team-redbull`, `.team-astonmartin`, `.team-williams`, `.team-rb`, `.team-haas`, `.team-cadillac`, `.team-audi`, etc.) are CSS classes defined inside each `podium.html`, each with a card rule and a `.logo-inner` rule, plus a matching `.card-<name>::after` accent rule. For any podium team, find its definition with `grep -hE "\.(team|card)-<name>" */podium.html | sort -u` and copy all three rules; define new hex values only when the grep finds nothing.
