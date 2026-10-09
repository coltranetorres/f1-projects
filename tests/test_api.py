import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from harness.agent import build_agent
from tests.helpers import scripted_model


def make_client(predictor, steps):
    app = create_app(predictor=predictor, agent_factory=lambda model_id: build_agent(scripted_model(steps)))
    return TestClient(app)


def sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_health_and_models(predictor):
    with make_client(predictor, [("text", "hi")]) as c:
        assert c.get("/health").json() == {"ok": True}
        body = c.get("/models").json()
        assert body["models"] and isinstance(body["default"], str)


def test_chat_streams_events_in_order(predictor):
    steps = [("tool", "get_prediction", {"top_n": 3}), ("text", "Verstappen leads.")]
    with make_client(predictor, steps) as c:
        r = c.post("/chat", json={"message": "podium?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    types = [e["type"] for e in sse_events(r)]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")


@pytest.mark.parametrize("message", ["", "x" * 2001])
def test_chat_rejects_bad_message(predictor, message):
    with make_client(predictor, [("text", "hi")]) as c:
        assert c.post("/chat", json={"message": message}).status_code == 422


def test_chat_provider_failure_is_an_error_event_not_a_500(predictor):
    with make_client(predictor, [("error", "boom")]) as c:
        r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert sse_events(r)[-1]["type"] == "error"


def test_feedback_calls_langfuse(predictor, monkeypatch):
    calls = []
    monkeypatch.setattr("api.main.submit_feedback", lambda trace_id, value, reason="": calls.append((trace_id, value, reason)))
    with make_client(predictor, [("text", "hi")]) as c:
        r = c.post("/feedback", json={"trace_id": "abc", "value": 0, "reason": "wrong driver"})
    assert r.json() == {"ok": True} and calls == [("abc", 0, "wrong driver")]


def test_feedback_validation(predictor):
    with make_client(predictor, [("text", "hi")]) as c:
        assert c.post("/feedback", json={"trace_id": "abc", "value": 5}).status_code == 422
        assert c.post("/feedback", json={"trace_id": "", "value": 1}).status_code == 400


def test_chat_rejects_unknown_model(predictor):
    with make_client(predictor, [("text", "hi")]) as c:
        r = c.post("/chat", json={"message": "hi", "model": "not/a-model"})
    assert r.status_code == 400 and "not/a-model" in r.json()["detail"]
