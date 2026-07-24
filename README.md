# f1-projects

A collection of Python (and one VS Code extension) projects built around the
[FastF1](https://github.com/theOehrly/Fast-F1) API, covering the 2026 F1
season: podium prediction, driver style analysis, telemetry visualization,
and highlight-reel rendering.

## Podium prediction (main project)

Each of `austria/`, `belgium/`, `british/`, `monaco/`, `spain/` is a
self-contained snapshot of the same pipeline, one per Grand Prix, predicting
the top-3 finishers using an XGBoost binary classifier trained on all
completed races in the 2026 season up to that point.

```bash
uv sync                            # install dependencies (Python >= 3.13, managed by uv)
uv run python {gp}/fetch_data.py   # fetch race data from FastF1 -> {gp}/data/
uv run python {gp}/model.py        # train model + print podium prediction
```

Workflow for predicting a new race, feature engineering details (22 features:
grid position, lap aggregates, sprint results, etc.), XGBoost config, and CSV
schema are documented in [CLAUDE.md](CLAUDE.md).

## Other projects

| Directory | What it is |
|---|---|
| `driver_similarity/` | PCA + clustering over six driving-style features (qualifying pace, overtakes, tire preservation, braking, aggression, wet performance) to group 2026-grid drivers by how they race, rendered as an HTML map. See `docs/superpowers/specs/2026-07-20-driver-similarity-design.md`. |
| `gforce_mapping/` | Reconstructs lateral G-force around a lap (Norris, 2025 British GP) from FastF1 telemetry and renders an animated track map / highlight reel. See `gforce_mapping/HANDOVER.md` for the G-force calculation approach. |
| `leclerc_british_gp_highlights/` | Telemetry-driven highlight video generation for a specific race battle (2025 British GP). |
| `lewis_barcelona_p1/` | Standalone HTML/video build for a single lap/session visualization. |
| `custom_flag_alarm/` | A local VS Code extension — a green/yellow/red flag in the sidebar that tracks how long you've been coding and nudges you to take a break. Not F1 data related; see `custom_flag_alarm/README.md`. |

## Repo layout notes

- Each GP directory is a point-in-time snapshot (own `fetch_data.py`,
  `model.py`, `data/`), not a shared library — copy-and-adapt is the
  intended workflow, per `CLAUDE.md`.
- `tests/` covers `driver_similarity`'s feature engineering and model code.
- `docs/superpowers/` holds design specs and plans written before
  implementing `driver_recognition` (superseded) and `driver_similarity`.
- `cache/` directories (FastF1's local API cache) and `.venv/` are
  gitignored — regenerate the cache by running each project's
  `fetch_data.py`.
