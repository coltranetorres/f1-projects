from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from race_dashboard.backend.replay import MAX_WINDOW_S, Replay


def create_app(replay: Replay | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if replay is not None:
            app.state.replay = replay
        else:
            from race_dashboard.backend.loader import get_replay  # lazy: pulls in fastf1
            app.state.replay = get_replay()
        yield

    app = FastAPI(title="F1 race dashboard", lifespan=lifespan)

    @app.get("/race")
    def race():
        return app.state.replay.meta

    @app.get("/frames")
    def frames(t0: float = Query(ge=0, le=1e7, allow_inf_nan=False),
               t1: float = Query(gt=0, le=1e7, allow_inf_nan=False)):
        if t1 <= t0:
            raise HTTPException(status_code=422, detail="t1 must be greater than t0")
        return app.state.replay.window(t0, min(t1, t0 + MAX_WINDOW_S))

    return app


app = create_app()
