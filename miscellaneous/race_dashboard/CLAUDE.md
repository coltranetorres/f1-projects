# race_dashboard

Replay dashboard for the real 2026 Singapore Grand Prix: a FastAPI backend serves a precomputed replay (positions, gaps, laps) and a React/Vite frontend plays it back with a track map, leaderboard, race-status card and controls.

## Layout

- `backend/replay.py` — pure numpy replay computation and `.npz` save/load. Must stay free of FastF1 imports.
- `backend/loader.py` — FastF1 session loading, builds and disk-caches the replay (`get_replay`).
- `backend/api.py` — FastAPI app: `/race` metadata, `/frames?t0=&t1=` windows and `/insights` (cached per-lap insights, 404 until built).
- `backend/insights/` — per-lap insights: `model.py` (LapRecord/LapTable), `facts.py` (candidate generators), `select.py` (top 3, one per type, cooldown), `ranker.py` (Jev `/api/alpha/decisions` scoring), `phrase.py` + `templates.py` (Gemini phrasing, number validation, template fallback), `build.py` (cache + orchestration), `__main__.py` (CLI). Must stay free of FastF1 imports; only `loader.build_lap_table` touches FastF1.
- `backend/tests/` — pytest tests for the backend.
- `frontend/src/` — `api.js`, `playback.js` (FrameBuffer, window prefetch, formatters; tested in `playback.test.js`), `usePlayback.js` (rAF playback hook), `App.jsx`, `components/` (TrackMap, Leaderboard, RaceStatus, Controls, Insights = the 3 insight cards under the map), `styles.css`.
- `cache/` — cached replays and insights (git-ignored).

## Commands

Run from the repo root:

```bash
PYTHONPATH=miscellaneous uv run python -m race_dashboard.backend.loader          # build/refresh the cached replay and print the final order
PYTHONPATH=miscellaneous uv run python -m race_dashboard.backend.insights --max-lap 3   # smoke test: print laps 1-3, no cache
PYTHONPATH=miscellaneous uv run python -m race_dashboard.backend.insights [--force]     # build cache/2026_Singapore_insights_v{INSIGHTS_VERSION}.json
uv run uvicorn race_dashboard.backend.api:app --app-dir miscellaneous --port 8000  # API
cd miscellaneous/race_dashboard/frontend && npm install && npm run dev            # UI on :5173 (proxies /api to :8000)
uv run pytest miscellaneous/race_dashboard/backend/tests                          # backend tests
cd miscellaneous/race_dashboard/frontend && npm test                               # frontend tests
```

## Notes

- The replay cache is `cache/2026_Singapore_v{REPLAY_VERSION}.npz`. Bumping `REPLAY_VERSION` in `loader.py` invalidates old caches and forces a rebuild; delete the file to rebuild manually.
- Per-car `status` in frames: 0 RUNNING, 1 FINISHED, 2 DNF, 3 PIT_START (pit-lane starter that has not moved yet; DNF wins over it).
- Singapore (2026) is pinned in the defaults of `loader.get_replay`.
- `replay.py` must stay free of FastF1 imports so it can be tested without network or FastF1.
- Running order is by race progress (laps + distance along the track outline, projected onto outline *segments*), with classified drivers' progress capped at their own lap count and official classification as a tie-break so the final order equals the official result; DNF cars rank last. Pit-lane starters (first sample more than `PIT_START_DIST` = 50 units from the outline) rank last and show status 3 until their first movement at t >= 0. If you change ranking logic, bump `REPLAY_VERSION` and re-check the real race (final order == official, no rank jumps of 3+ between frames).
- `/frames` only accepts finite `0 <= t0 < t1 <= 1e7` (422 otherwise; window length capped at `MAX_WINDOW_S`). The frontend must never request a negative start; `playback.js` clamps the playhead (`advancePlayhead`) and the fetch list.
- Test coverage: the backend and the pure frontend logic (`playback.js`: buffer, interpolation, retry tracker, playhead clock, formatters) are unit-tested. The React hook (`usePlayback.js`) and the CSS/layout have no automated tests and no browser end-to-end test; check the UI by hand (play, seek, end of race, API down/up) after changing them.
- Insights need `OPENROUTER_API_KEY`, `F1_DEFAULT_MODEL` (Gemini phrasing) and `F1_RANKER_MODEL` (Jev ranking) in the repo-root `.env`. Bump `INSIGHTS_VERSION` in `insights/build.py` when facts, ranking or prompts change. The set for lap N uses laps <= N and is shown once the leader has completed lap N. A lap whose Jev call fails is marked `ranked_by: "heuristic"`; items Gemini could not phrase with valid numbers are marked `source: "template"`.
- Insight numbers are computed in code; Gemini only phrases them and `phrase.numbers_ok` rejects any number not in the facts. Do not mention DRS (2026 cars have none).
