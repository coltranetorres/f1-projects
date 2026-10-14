# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict the podium (top 3 finishers) for upcoming F1 Grand Prix races using an XGBoost binary classifier trained on all completed races in the 2026 season, plus a read-only Q&A assistant over one of those predictions and a few unrelated side projects.

## Layout

- `predictions/` — one directory per Grand Prix plus the shared `data_pred/data/` CSVs. Pipeline, model details, CSV schema and current season state: see `predictions/CLAUDE.md`. New race: use the `new-gp-prediction` skill.
- `ai_assistant/` — `backend/` (`f1_core/`, `harness/`, `api/`, `tests/`), `frontend/`, `docs/`. See `ai_assistant/CLAUDE.md`.
- `miscellaneous/` — standalone projects (`driver_similarity/`, `gforce_mapping/`, `leclerc_british_gp_highlights/`, `lewis_barcelona_p1/`, `custom_flag_alarm/`, `race_dashboard/` = FastAPI + React replay dashboard of the 2026 Singapore GP; see its CLAUDE.md). Each has its own README/handover; they don't depend on the other folders.

Root holds shared tooling only: `.venv`, `.env`, `pyproject.toml`, `uv.lock`, `.claude/`.

## Commands

Dependencies are managed with `uv` (Python >= 3.13). Run everything from the repo root:

```bash
uv sync        # install all dependencies
uv run pytest  # all tests (ai_assistant/backend/tests + miscellaneous/driver_similarity/tests + miscellaneous/race_dashboard/backend/tests)
```

Folder-specific commands live in the nested CLAUDE.md files.
