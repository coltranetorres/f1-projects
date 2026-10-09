import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from harness.agent import HarnessDeps, build_agent
from tests.helpers import scripted_model


def run(predictor, steps, limits=None):
    agent = build_agent(scripted_model(steps))
    deps = HarnessDeps(predictor)
    result = agent.run_sync("test question", deps=deps, usage_limits=limits)
    return result, deps


def test_bad_driver_is_retried_then_recovers(predictor):
    result, deps = run(predictor, [
        ("tool", "explain_driver", {"driver": "ZZZ"}),
        ("tool", "explain_driver", {"driver": "VER"}),
        ("text", "Verstappen's grid position matters most."),
    ])
    assert len(deps.tool_errors) == 1 and deps.tool_errors[0]["args"] == {"driver": "ZZZ"}
    assert [c["name"] for c in deps.tool_calls] == ["explain_driver"]
    assert result.output.startswith("Verstappen")


def test_tool_call_cap_stops_a_loop(predictor):
    with pytest.raises(UsageLimitExceeded):
        run(predictor, [("tool", "list_drivers", {})], limits=UsageLimits(tool_calls_limit=3))


def test_grounded_answer_passes_without_flag(predictor):
    p = predictor.predict().iloc[0]["proba_podium"]
    result, deps = run(predictor, [
        ("tool", "get_prediction", {"top_n": 1}),
        ("text", f"The top pick has a {p * 100:.1f}% podium probability."),
    ])
    assert deps.unverified == [] and deps.grounding_retries == 0


def test_invented_number_retries_once_then_flags(predictor):
    result, deps = run(predictor, [
        ("tool", "get_prediction", {"top_n": 1}),
        ("text", "He has an 87.31% chance."),
        ("text", "He has an 87.31% chance."),
    ])
    assert deps.grounding_retries == 1
    assert deps.unverified == ["87.31%"]
    assert "87.31%" in result.output


def test_invented_number_corrected_on_retry_is_not_flagged(predictor):
    p = predictor.predict().iloc[0]["proba_podium"]
    result, deps = run(predictor, [
        ("tool", "get_prediction", {"top_n": 1}),
        ("text", "He has an 87.31% chance."),
        ("text", f"Correction: {p * 100:.1f}%."),
    ])
    assert deps.grounding_retries == 1 and deps.unverified == []
