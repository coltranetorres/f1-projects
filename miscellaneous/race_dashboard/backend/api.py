from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from race_dashboard.backend.insights.build import cache_path, load_insights
from race_dashboard.backend.replay import MAX_WINDOW_S, Replay


def create_app(replay: Replay | None = None, insights: list | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if replay is not None:
            app.state.replay = replay
            app.state.insights = insights
        else:
            from race_dashboard.backend.loader import get_replay  # lazy: pulls in fastf1
            app.state.replay = get_replay()
            app.state.insights = insights if insights is not None else load_insights(cache_path())
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

    @app.get("/insights")
    def get_insights():
        if app.state.insights is None:
            raise HTTPException(status_code=404,
                                detail="insights not built; run python -m race_dashboard.backend.insights")
        return app.state.insights

    return app


app = create_app()
