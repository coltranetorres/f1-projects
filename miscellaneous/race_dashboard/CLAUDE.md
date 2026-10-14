# race_dashboard

Replay dashboard for the real 2026 Singapore Grand Prix: a FastAPI backend serves a precomputed replay (positions, gaps, laps) and a React/Vite frontend plays it back with a track map, leaderboard, race-status card and controls.

## Layout

- `backend/replay.py` — pure numpy replay computation and `.npz` save/load. Must stay free of FastF1 imports.
- `backend/loader.py` — FastF1 session loading, builds and disk-caches the replay (`get_replay`).
- `backend/api.py` — FastAPI app: `/race` metadata and `/frames?t0=&t1=` windows.
- `backend/tests/` — pytest tests for the backend.
- `frontend/src/` — `api.js`, `playback.js` (FrameBuffer, window prefetch, formatters; tested in `playback.test.js`), `usePlayback.js` (rAF playback hook), `App.jsx`, `components/` (TrackMap, Leaderboard, RaceStatus, Controls), `styles.css`.
- `cache/` — cached replays (git-ignored).

## Commands

Run from the repo root:

```bash
PYTHONPATH=miscellaneous uv run python -m race_dashboard.backend.loader          # build/refresh the cached replay and print the final order
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
