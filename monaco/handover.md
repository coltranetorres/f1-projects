# Monaco GP 2026 — Podium Prediction Handover

## Approach

### 1. Data Collection
- Source: FastF1 library (`fastf1` Python package)
- Fetched all 5 races from the 2026 F1 season prior to Monaco GP (Round 6, June 7 2026):
  - R01: Australian Grand Prix
  - R02: Chinese Grand Prix (sprint weekend)
  - R03: Japanese Grand Prix
  - R04: Miami Grand Prix (sprint weekend)
  - R05: Canadian Grand Prix (sprint weekend)
- For each race: race results + lap data saved as CSV in `monaco/data/`
- For sprint weekends (R02, R04, R05): sprint results also saved (sprint qualifying excluded)
- Weather data excluded as not relevant to the model

### 2. Feature Engineering
22 features derived from race results and lap data:

| Feature | Source | Description |
|---|---|---|
| GridPosition | Results | Starting grid position |
| positions_gained | Results | GridPosition minus finishing Position |
| finished | Results | 1 if Status == "Finished", else 0 |
| mean_lap_s | Laps | Mean accurate lap time in seconds |
| best_lap_s | Laps | Fastest accurate lap in seconds |
| lap_consistency | Laps | Std dev of accurate lap times |
| mean_s1_s | Laps | Mean Sector 1 time |
| mean_s2_s | Laps | Mean Sector 2 time |
| mean_s3_s | Laps | Mean Sector 3 time |
| avg_speed_i1 | Laps | Mean speed at intermediate 1 |
| avg_speed_i2 | Laps | Mean speed at intermediate 2 |
| avg_speed_fl | Laps | Mean speed at finish line |
| max_speed_st | Laps | Max speed trap |
| personal_best_count | Laps | Count of personal best laps set |
| avg_race_position | Laps | Mean in-race position across all laps |
| num_stints | Laps | Number of distinct stints |
| sprint_pos | Sprint results | Sprint finishing position (22 if no sprint) |
| sprint_points | Sprint results | Sprint points scored (0 if no sprint) |
| sprint_grid | Sprint results | Sprint grid position (22 if no sprint) |
| has_sprint | Sprint results | 1 if sprint weekend, else 0 |
| team_encoded | Results | Label-encoded team name |
| driver_encoded | Results | Label-encoded driver name |

Only accurate laps (IsAccurate == True) used for lap aggregation.

### 3. Target Variable
Binary classification:
- `1` = podium finisher (Position <= 3)
- `0` = non-podium (Position > 3)

110 total rows (22 drivers x 5 races): 15 podium, 95 non-podium.

### 4. Monaco Prediction Input
Rather than using only the most recent race (which would bias toward Canadian GP results), features were **averaged across all 5 races per driver** to capture season-long form. Team and driver identity taken from most recent round.

### 5. Train / Test Split
- 80-20 stratified split (random_state=42)
- Train: 88 rows (76 non-podium, 12 podium)
- Test: 22 rows (19 non-podium, 3 podium)

### 6. Model
XGBoost binary classifier:
- `objective="binary:logistic"`
- `scale_pos_weight=6.33` (95/15) to handle class imbalance
- `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- Early stopping: 30 rounds on logloss
- Best iteration: 80

### 7. Evaluation
| Metric | Train | Test | Gap | Verdict |
|---|---|---|---|---|
| Accuracy | 0.9773 | 0.9545 | 0.023 | Generalising well |
| ROC-AUC | 0.9989 | 0.8772 | 0.122 | Mild overfitting |

Classification report (test set):
- Non-podium: precision 0.95, recall 1.00, f1 0.97
- Podium: precision 1.00, recall 0.67, f1 0.80

The mild overfitting on ROC-AUC is expected given the small dataset (110 rows). The model correctly identifies 2/3 podium drivers in test races with zero false positives on the podium class.

Top features by importance:
1. avg_race_position (0.4578)
2. GridPosition (0.2755)
3. mean_s1_s (0.0460)
4. max_speed_st (0.0348)
5. best_lap_s (0.0321)

### 8. Monaco Prediction (model retrained on all 110 rows)

#### Predicted Podium
| Pos | Driver | Team | P(podium) |
|---|---|---|---|
| P1 | Kimi Antonelli | Mercedes | 0.9481 |
| P2 | George Russell | Mercedes | 0.8732 |
| P3 | Charles Leclerc | Ferrari | 0.6769 |

#### Full Driver Ranking
| Rank | Driver | Team | P(podium) |
|---|---|---|---|
| 1 | Kimi Antonelli | Mercedes | 0.9481 |
| 2 | George Russell | Mercedes | 0.8732 |
| 3 | Charles Leclerc | Ferrari | 0.6769 |
| 4 | Lewis Hamilton | Ferrari | 0.6424 |
| 5 | Oscar Piastri | McLaren | 0.0502 |
| 6 | Lando Norris | McLaren | 0.0435 |
| 7 | Alexander Albon | Williams | 0.0284 |
| 8 | Arvid Lindblad | Racing Bulls | 0.0274 |
| 9 | Esteban Ocon | Haas F1 Team | 0.0274 |
| 10 | Carlos Sainz | Williams | 0.0263 |
| 11 | Isack Hadjar | Red Bull Racing | 0.0262 |
| 12 | Franco Colapinto | Alpine | 0.0262 |
| 13 | Oliver Bearman | Haas F1 Team | 0.0262 |
| 14 | Max Verstappen | Red Bull Racing | 0.0262 |
| 15 | Liam Lawson | Racing Bulls | 0.0254 |
| 16 | Gabriel Bortoleto | Audi | 0.0253 |
| 17 | Nico Hulkenberg | Audi | 0.0251 |
| 18 | Valtteri Bottas | Cadillac | 0.0243 |
| 19 | Sergio Perez | Cadillac | 0.0243 |
| 20 | Lance Stroll | Aston Martin | 0.0232 |
| 21 | Pierre Gasly | Alpine | 0.0232 |
| 22 | Fernando Alonso | Aston Martin | 0.0223 |

Note: P3 vs P4 (Leclerc 0.6769 vs Hamilton 0.6424) is very close — effectively a coin flip between them.

---

## SHAP Analysis

### Global Feature Importance (Mean |SHAP| across all 110 rows)

| Feature | Mean |SHAP| | Impact |
|---|---|---|
| avg_race_position | 2.6079 | Dominant — nearly 5x more impactful than anything else |
| GridPosition | 0.4851 | Strong secondary driver |
| personal_best_count | 0.1829 | Marginal |
| mean_s3_s | 0.0872 | Marginal |
| avg_speed_i1 | 0.0525 | Marginal |
| best_lap_s | 0.0339 | Marginal |
| mean_s1_s | 0.0310 | Marginal |
| max_speed_st | 0.0265 | Marginal |
| positions_gained | 0.0167 | Marginal |
| lap_consistency | 0.0147 | Marginal |
| sprint_grid | 0.0126 | Marginal |
| avg_speed_fl | 0.0066 | Marginal |
| driver_encoded | 0.0052 | Marginal |
| mean_lap_s | 0.0000 | No contribution |
| finished | 0.0000 | No contribution |
| mean_s2_s | 0.0000 | No contribution |
| avg_speed_i2 | 0.0000 | No contribution |
| num_stints | 0.0000 | No contribution |
| sprint_points | 0.0000 | No contribution |
| sprint_pos | 0.0000 | No contribution |
| has_sprint | 0.0000 | No contribution |
| team_encoded | 0.0000 | No contribution |

9 features have zero SHAP contribution (all sprint features, `finished`, `mean_lap_s`, `mean_s2_s`, `avg_speed_i2`, `num_stints`, `team_encoded`) — dead weight in this model with the current dataset size. They would likely activate with more training data.

### Per-Driver SHAP Breakdown — Top 4 Monaco Contenders

Positive SHAP = pushes toward podium. Negative = pushes away.

**Kimi Antonelli (Mercedes) — P(podium)=0.9481**
| Feature | SHAP | Direction |
|---|---|---|
| avg_race_position | +2.5305 | Strong push toward podium |
| GridPosition | +0.2192 | Toward podium |
| finished | +0.1454 | Toward podium |
| avg_speed_fl | +0.1055 | Toward podium |
| driver_encoded | +0.0875 | Toward podium |
| mean_s3_s | -0.1072 | Away from podium |
| sprint_grid | -0.0981 | Away from podium |
| positions_gained | -0.0738 | Away from podium |

**George Russell (Mercedes) — P(podium)=0.8732**
| Feature | SHAP | Direction |
|---|---|---|
| avg_race_position | +2.4613 | Strong push toward podium |
| GridPosition | +0.1977 | Toward podium |
| finished | -0.3781 | Away from podium (reliability concern) |
| driver_encoded | -0.1775 | Away from podium |
| mean_s3_s | -0.1072 | Away from podium |
| personal_best_count | -0.0822 | Away from podium |
| sprint_grid | -0.0800 | Away from podium |
| positions_gained | -0.0738 | Away from podium |

**Charles Leclerc (Ferrari) — P(podium)=0.6769**
| Feature | SHAP | Direction |
|---|---|---|
| avg_race_position | +0.8042 | Toward podium |
| GridPosition | +0.2113 | Toward podium |
| finished | +0.1370 | Toward podium |
| driver_encoded | -0.1775 | Away from podium |
| mean_s3_s | -0.1243 | Away from podium |
| personal_best_count | -0.1095 | Away from podium |
| sprint_grid | -0.0981 | Away from podium |
| positions_gained | -0.0738 | Away from podium |

**Lewis Hamilton (Ferrari) — P(podium)=0.6424**
| Feature | SHAP | Direction |
|---|---|---|
| avg_race_position | +0.2987 | Toward podium |
| finished | +0.1378 | Toward podium |
| GridPosition | +0.1357 | Toward podium |
| driver_encoded | +0.0875 | Toward podium |
| avg_speed_fl | +0.0661 | Toward podium |
| positions_gained | +0.0573 | Toward podium |
| mean_s3_s | -0.1243 | Away from podium |
| sprint_grid | -0.0981 | Away from podium |

### Key SHAP Takeaways

- **Antonelli vs Russell separation**: Both driven almost entirely by `avg_race_position` (2.53 vs 2.46 SHAP) — nearly identical. Russell's `finished` is negative (-0.38) suggesting reliability concerns pulling his probability down.
- **Ferrari pair separation**: Leclerc vs Hamilton is very tight (0.6769 vs 0.6424). Leclerc's `avg_race_position` SHAP (0.80) edges Hamilton's (0.30) — consistent season form across all 5 races is the difference.
- **The cliff after P4**: Antonelli/Russell have `avg_race_position` SHAP of 2.4-2.5; Leclerc/Hamilton have 0.3-0.8; everyone else drops below 0.05 P(podium). The model clearly separates two tiers of contenders from the rest.
- **Sprint features are noise here**: All sprint SHAP values are negative or zero, suggesting sprint performance does not positively predict race podium with this dataset.

---

## Next Step: Re-run Prediction with Monaco Qualifying Data

Once Monaco qualifying results are available (Saturday before the race), re-run the prediction incorporating qualifying data as additional features. This will significantly improve accuracy because:

- **Grid position at Monaco is exceptionally predictive** — overtaking is near-impossible on this street circuit, so qualifying position is a stronger predictor of race outcome here than at any other track
- Qualifying lap times (Q1, Q2, Q3) give precise single-lap pace specific to the Monaco circuit, which has unique characteristics (slow, narrow, high downforce, no real straights) that season-average speed trap data does not capture

### How to incorporate qualifying data
1. Fetch Monaco qualifying session via FastF1:
   ```python
   quali = fastf1.get_session(2026, "Monaco", "Q")
   quali.load()
   ```
2. Extract per-driver: `QualifyingPosition`, `Q1_time_s`, `Q2_time_s`, `Q3_time_s`
3. Merge into the `monaco_drivers` feature DataFrame before running `model_full.predict_proba()`
4. Add the new columns to `FEATURES` and retrain `model_full` on the augmented dataset
5. Re-rank all 22 drivers by P(podium) — expect qualifying position to become the dominant feature
