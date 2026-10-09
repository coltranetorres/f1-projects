import asyncio

from harness.agent import build_agent
from harness.runner import run_chat_sync, stream_chat
from tests.helpers import scripted_model


def collect(agent, predictor, message="q"):
    async def go():
        return [e async for e in stream_chat(agent, predictor, message, None, "test-model")]
    return asyncio.run(go())


def test_run_chat_sync_collects_tool_calls(predictor):
    agent = build_agent(scripted_model([("tool", "get_prediction", {"top_n": 3}),
                                        ("text", "Verstappen leads the field.")]))
    res = run_chat_sync(agent, predictor, "podium?")
    assert res.answer == "Verstappen leads the field."
    assert [c["name"] for c in res.tool_calls] == ["get_prediction"]
    assert not res.nudge and res.trace_id is None  # tracing off in tests
    assert res.to_output()["tool_calls"][0]["args"] == {"top_n": 3}


def test_run_chat_sync_reports_cap(predictor):
    agent = build_agent(scripted_model([("tool", "list_drivers", {})]))
    res = run_chat_sync(agent, predictor, "loop forever")
    assert res.capped and res.nudge
    assert "tool-call limit" in res.answer


def test_stream_emits_tool_events_then_done(predictor):
    agent = build_agent(scripted_model([("tool", "get_prediction", {"top_n": 3}),
                                        ("text", "Verstappen leads.")]))
    events = collect(agent, predictor)
    types = [e["type"] for e in events]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    assert types[-1] == "done"
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["name"] == "get_prediction" and call["args"] == {"top_n": 3}
    assert events[-1]["answer"] == "Verstappen leads." and events[-1]["unverified"] == []


def test_stream_surfaces_rejected_tool_call(predictor):
    agent = build_agent(scripted_model([("tool", "explain_driver", {"driver": "ZZZ"}),
                                        ("tool", "explain_driver", {"driver": "VER"}),
                                        ("text", "Done.")]))
    events = collect(agent, predictor)
    retry = next(e for e in events if e["type"] == "tool_retry")
    assert retry["name"] == "explain_driver" and "Valid codes" in retry["message"]
    assert events[-1]["type"] == "done" and events[-1]["nudge"] is True


def test_exception_becomes_error_event(predictor):  # Review Focus #5
    agent = build_agent(scripted_model([("error", "provider exploded")]))
    events = collect(agent, predictor)
    assert events[-1]["type"] == "error" and "provider exploded" in events[-1]["message"]
    assert not any(e["type"] == "done" for e in events)


def test_closing_stream_early_cancels_llm_call(predictor):  # SSE client disconnect -> no paid call
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    cancelled, in_flight = [], asyncio.Event()

    async def stream(messages, info: AgentInfo):
        yield "partial answer"  # one event reaches the UI, then the provider call hangs
        in_flight.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    def function(messages, info):  # unused; streaming path only
        raise AssertionError("non-streaming path should not run")

    agent = build_agent(FunctionModel(function, stream_function=stream))

    async def go():
        gen = stream_chat(agent, predictor, "q", None, "test-model")
        first = await gen.__anext__()
        assert first["type"] == "text_delta"
        await asyncio.wait_for(in_flight.wait(), 5)  # provider call is now hanging
        await gen.aclose()  # client went away
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return others

    others = asyncio.run(go())
    assert cancelled == [True]
    assert others == []
