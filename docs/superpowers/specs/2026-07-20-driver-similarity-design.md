# Driver Similarity Embeddings — Design

## Goal

Instead of predicting race outcomes, learn a representation of each 2026-grid
driver's *driving style* — braking behavior, tire management, wet-weather
performance, overtaking, aggression, qualifying pace — and use PCA + clustering
to group drivers who race alike. Output is a public-friendly visual, not a
technical dashboard.

## Scope

- All drivers who appeared in any of rounds R01–R09 (Australia through
  British GP), including one-off substitutes.
- Race data only (sprint sessions excluded from pace/telemetry aggregation,
  consistent with how `model.py` in other GP directories treats sprints as a
  separate signal).
- Six features: qualifying pace, overtakes, tire preservation, braking style,
  aggression, wet performance.

## Directory layout

```
driver_similarity/
  fetch_data.py   # FastF1 fetch: results, laps, weather, sampled telemetry
                   #   for R01-R09 -> data/R{rn:02d}_{event}_driver_features.csv
  features.py      # Aggregates per-race feature CSVs -> data/driver_features.csv
                    #   (one row per driver, season-level averages)
  model.py          # Standardize -> PCA(2) -> KMeans (auto-k via silhouette)
                     #   -> archetype labeling -> output/driver_similarity.html
  cache/             # FastF1 cache (existing per-project convention)
  data/              # Feature CSVs
  output/            # Generated HTML artifact
```

## Data flow & feature engineering

For each of R01–R09, for each driver:

1. Load `session.laps`, `session.weather_data` via FastF1 (`telemetry=False`
   for the bulk load — telemetry fetched separately per lap, see below).
2. Select the driver's 5 fastest race laps (by `LapTime`, `IsAccurate == True`,
   excluding pit in/out laps) and fetch telemetry (`Brake`, `Throttle`,
   `Speed`, `nGear`) for just those laps via `lap.get_car_data()`. Sampling
   fastest laps (not all ~65 laps/driver/race) keeps the telemetry fetch
   tractable — full-lap telemetry across 9 races × ~20 drivers would be a
   multi-hour fetch for marginal signal gain over clean, representative laps.
3. Compute per-race, per-driver features:

| Feature | Computation |
|---|---|
| `qualifying_pace` | % delta to session pole time, from `Qualifying_results.csv` where present, else Q-session lap times from `session.laps` |
| `overtakes` | Sum of lap-to-lap position gains during the race, excluding lap 1 and pit in/out laps |
| `tire_preservation` | Linear regression slope of `LapTime` vs `TyreLife` within each stint (flatter/lower slope = better preservation), averaged across the race's stints |
| `braking_style` | From sampled telemetry: brake-zone count per lap, mean brake-zone duration, mean speed at brake onset (late-braking proxy) |
| `aggression` | Composite z-score of: overtake rate + late-braking tendency + post-apex throttle-reapplication speed |
| `wet_performance` | Pace/position delta vs. the driver's own dry-race baseline, computed **only** for races where `weather_data.Rainfall` is majority-true across the session. Null for drivers with no wet-race laps in the dataset. |

4. `features.py` averages each feature across all races a driver appears in,
   producing one row per driver in `data/driver_features.csv`. `wet_performance`
   is mean-imputed across the grid for drivers with no wet-race data (not
   fabricated per-driver) so PCA has a complete matrix; a `has_wet_data`
   boolean column is retained for transparency but not used as a PCA input.

## Model (`model.py`)

1. Z-score all 6 features across the grid.
2. PCA to 2 components. Print explained variance ratio.
3. KMeans swept over k = 3..6; select k by best silhouette score.
4. Label each cluster from its centroid's most extreme z-scored features
   (e.g., high `aggression` + high `overtakes` + steep `tire_preservation`
   slope → "Aggressive Overtaker"). Labels are generated from a small
   lookup of feature-combination → archetype phrase, not hardcoded per
   driver.

## Deliverable

`output/driver_similarity.html` — self-contained HTML artifact:

- 2D scatter of drivers positioned by PCA embedding, colored by cluster,
  dots in team color, driver name always labeled (not hover-only).
- Axes are framed qualitatively based on what each component actually
  correlates with post-hoc (e.g. "Pace ↔ Consistency"), not "PC1"/"PC2".
- Short plain-language caption per cluster describing the shared trait.
- No raw feature tables or technical jargon in the primary view — this is
  meant to be skimmable by a non-technical viewer.

## Edge cases

- Substitute drivers with only 1–2 races: included per user decision, no
  special low-sample-size flag in v1.
- No wet races in R01–R09: `wet_performance` becomes a uniform
  (imputed) column — code stays generic for future seasons where wet races
  exist, but current run's clustering will be driven by the other 5
  features in practice.
- Driver present in `laps.csv` but missing from `Qualifying_results.csv`
  for a given round (e.g. DNQ, or round predates qualifying CSV creation):
  fall back to `GridPosition` from `results.csv` as a proxy for that race's
  qualifying pace, or skip that race's qualifying feature for that driver
  (favor: skip and average over remaining races, to avoid conflating grid
  penalties with pace).

## Out of scope (v1)

- Interactive drill-down / live capability artifact (static snapshot only).
- Per-driver low-sample-size confidence indicators.
- Hyperparameter weighting of features (all 6 weighted equally via z-score).
