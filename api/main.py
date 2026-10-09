import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from harness.agent import build_agent, build_model
from harness.config import available_models, default_model, get_predictor
from harness.feedback import submit_feedback
from harness.runner import stream_chat
from harness.tracing import flush, init_tracing


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    model: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    value: Literal[0, 1]
    reason: str = Field(default="", max_length=2000)


def _default_agent_factory(model_id: str):
    return build_agent(build_model(model_id))


def create_app(predictor=None, agent_factory=None) -> FastAPI:
    factory = agent_factory or _default_agent_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_tracing()
        app.state.predictor = predictor or get_predictor()  # trains at startup, not on first request
        yield
        flush()

    app = FastAPI(title="F1 harness", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/models")
    def models():
        return {"models": available_models(), "default": default_model()}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        if req.model is not None and req.model not in available_models():
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}; available: {available_models()}")
        model_id = req.model or default_model()
        try:
            agent = factory(model_id)
        except RuntimeError as e:  # e.g. missing OpenRouter key
            raise HTTPException(status_code=400, detail=str(e))

        async def events():
            async for ev in stream_chat(agent, app.state.predictor, req.message, req.session_id, model_id):
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        try:
            submit_feedback(req.trace_id, req.value, req.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True}

    return app


app = create_app()
