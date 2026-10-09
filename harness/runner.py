import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import AsyncIterator

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent,
                                  PartStartEvent, RetryPromptPart, TextPart, TextPartDelta,
                                  ToolReturnPart)
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python

from f1_core import Predictor

from .agent import HarnessDeps
from .tracing import chat_trace

LIMITS = UsageLimits(tool_calls_limit=8)
REQUEST_TIMEOUT_S = 90.0
CAP_MESSAGE = "I hit my tool-call limit before finishing, so I can't give a reliable answer to that."
TIMEOUT_MESSAGE = "The request timed out before I could finish."


@dataclass
class ChatResult:
    answer: str
    trace_id: str | None
    tool_calls: list[dict] = field(default_factory=list)
    tool_errors: list[dict] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    capped: bool = False
    timed_out: bool = False
    grounding_retries: int = 0

    @property
    def nudge(self) -> bool:
        """Highlight the feedback prompt: something suspicious happened in this run."""
        return bool(self.unverified or self.tool_errors or self.grounding_retries
                    or self.capped or self.timed_out)

    def to_output(self) -> dict:
        return to_jsonable_python({
            "answer": self.answer, "tool_calls": self.tool_calls, "tool_errors": self.tool_errors,
            "unverified": self.unverified, "capped": self.capped, "trace_id": self.trace_id})

    def guardrail(self) -> dict:
        return {"capped": self.capped, "timed_out": self.timed_out,
                "unverified": self.unverified, "tool_errors": len(self.tool_errors),
                "grounding_retries": self.grounding_retries}


def _result(deps: HarnessDeps, answer: str, trace_id, capped=False, timed_out=False) -> ChatResult:
    return ChatResult(answer=answer, trace_id=trace_id, tool_calls=deps.tool_calls,
                      tool_errors=deps.tool_errors, unverified=deps.unverified, capped=capped,
                      timed_out=timed_out, grounding_retries=deps.grounding_retries)


async def run_chat(agent, predictor: Predictor, message: str, session_id: str | None = None,
                   model_id: str = "") -> ChatResult:
    deps = HarnessDeps(predictor)
    with chat_trace(message, session_id, model_id) as span:
        capped = False
        try:
            run = await agent.run(message, deps=deps, usage_limits=LIMITS,
                                  model_settings={"timeout": REQUEST_TIMEOUT_S})
            answer = run.output
        except UsageLimitExceeded:
            answer, capped = CAP_MESSAGE, True
        out = _result(deps, answer, span.trace_id, capped=capped)
        span.update(output=answer, metadata=out.guardrail())
    return out


def run_chat_sync(agent, predictor: Predictor, message: str, session_id: str | None = None,
                  model_id: str = "") -> ChatResult:
    return asyncio.run(run_chat(agent, predictor, message, session_id, model_id))


def _translate(ev) -> list[dict]:
    if isinstance(ev, PartStartEvent) and isinstance(ev.part, TextPart) and ev.part.content:
        return [{"type": "text_delta", "text": ev.part.content}]
    if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
        return [{"type": "text_delta", "text": ev.delta.content_delta}]
    if isinstance(ev, FunctionToolCallEvent):
        return [{"type": "tool_call", "name": ev.part.tool_name, "args": ev.part.args_as_dict()}]
    if isinstance(ev, FunctionToolResultEvent):
        part = ev.part  # pydantic-ai 2.54: FunctionToolResultEvent.part (was .result)
        if isinstance(part, ToolReturnPart):
            return [{"type": "tool_result", "name": part.tool_name,
                     "result": to_jsonable_python(part.content)}]
        if isinstance(part, RetryPromptPart):
            return [{"type": "tool_retry", "name": part.tool_name or "",
                     "message": part.model_response()}]
    return []


async def stream_chat(agent, predictor: Predictor, message: str, session_id: str | None = None,
                      model_id: str = "") -> AsyncIterator[dict]:
    """Yield UI events. The Langfuse span lives entirely inside one producer task."""
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        deps = HarnessDeps(predictor)
        try:
            with chat_trace(message, session_id, model_id) as span:
                answer, capped, timed_out = "", False, False
                try:
                    async with asyncio.timeout(REQUEST_TIMEOUT_S):
                        # pydantic-ai 2.54: run_stream_events() returns an async context manager
                        async with agent.run_stream_events(message, deps=deps,
                                                           usage_limits=LIMITS) as events:
                            async for ev in events:
                                for out in _translate(ev):
                                    await queue.put(out)
                                if isinstance(ev, AgentRunResultEvent):
                                    answer = ev.result.output
                except UsageLimitExceeded:
                    answer, capped = CAP_MESSAGE, True
                except TimeoutError:
                    answer, timed_out = TIMEOUT_MESSAGE, True
                res = _result(deps, answer, span.trace_id, capped=capped, timed_out=timed_out)
                span.update(output=answer, metadata=res.guardrail())
                await queue.put({"type": "done", "answer": answer, "trace_id": res.trace_id,
                                 "unverified": res.unverified, "nudge": res.nudge})
        except Exception as e:  # provider/auth/bad-model errors must reach the UI as an event
            await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put(None)

    task = asyncio.create_task(producer())
    try:
        while (item := await queue.get()) is not None:
            yield item
        await task
    finally:
        # Client disconnected (aclose) or consumer stopped early: don't leave a paid LLM call running.
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
