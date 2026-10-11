import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel


def scripted_model(steps):
    """Deterministic fake LLM. steps[n] drives the n-th model response; the last step repeats."""
    def _step(messages):
        n = sum(isinstance(m, ModelResponse) for m in messages)
        return steps[min(n, len(steps) - 1)]

    def function(messages, info: AgentInfo) -> ModelResponse:
        kind, *rest = _step(messages)
        if kind == "tool":
            return ModelResponse(parts=[ToolCallPart(rest[0], rest[1])])
        if kind == "error":
            raise RuntimeError(rest[0])
        return ModelResponse(parts=[TextPart(rest[0])])

    async def stream(messages, info: AgentInfo):
        kind, *rest = _step(messages)
        if kind == "tool":
            yield {0: DeltaToolCall(name=rest[0], json_args=json.dumps(rest[1]))}
        elif kind == "error":
            raise RuntimeError(rest[0])
        else:
            yield rest[0]

    return FunctionModel(function, stream_function=stream)
