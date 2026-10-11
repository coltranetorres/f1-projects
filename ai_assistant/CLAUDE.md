# ai_assistant/

Read-only Q&A assistant over the Sepang podium prediction (PoC). Spec: `docs/superpowers/specs/2026-10-09-f1-harness-poc-design.md` and plan in `docs/superpowers/plans/` (local only, gitignored).

## Layout

- `backend/` — Python code, imported as `ai_assistant.backend.*`:
  - `f1_core/` — importable pipeline extracted from `predictions/sepang/model.py` (`Predictor`); regression-tested against the script's output in `backend/tests/fixtures/`. Reads race data from `predictions/data_pred/data/` (path set in `f1_core/data.py`).
  - `harness/` — Pydantic AI agent, typed read-only tools, guardrails, Langfuse tracing, `evals/` (seed cases, evaluators, `taxonomy.yaml`).
  - `api/` — FastAPI app.
  - `tests/` — pytest suite for the above.
- `frontend/` — React app (Vite; proxies `/api` to `localhost:8000`).
- `docs/` — `architecture.png` (diagram of the stack and the harness, source `architecture.svg`), plus local-only specs and plans.

The assistant is pinned to Sepang (`TARGET_ROUND = 16`, `QUALI_FILE` and the 15-race training list in `backend/f1_core/data.py`). It does not follow the prediction scripts to newer rounds; update those constants (and the regression fixture) deliberately if the target changes.

## Commands

Run from the repo root (the package is imported as `ai_assistant.*`):

```bash
uv run pytest                                              # full suite (assistant + driver_similarity)
uv run pytest ai_assistant/backend/tests                          # assistant tests only
uv run uvicorn ai_assistant.backend.api.main:app --port 8000       # API
cd ai_assistant/frontend && npm install && npm run dev          # frontend (npm install first time only)
uv run python -m ai_assistant.backend.harness.chat "question"      # manual one-shot check
uv run python -m ai_assistant.backend.harness.evals.run --model <openrouter-id> [--suite S] [--split dev|heldout|all] [--limit N]
```

## Notes

- Secrets live in the root `.env` (gitignored; see `.env.example`). `load_dotenv()` finds it by walking up from `backend/harness/`.
- XGBoost needs `brew install libomp` on macOS.
- Tests set `F1_DISABLE_TRACING=1` in `backend/tests/conftest.py` so they never send traces to Langfuse.
- Failure modes in `backend/harness/evals/taxonomy.yaml` come only from inspected Langfuse traces — don't add entries from guesses.
