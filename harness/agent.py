from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from f1_core import Predictor

from . import tools
from .config import openrouter_key
from .grounding import unmatched_numbers
from .schemas import (DriversResult, ExplainResult, FormResult, ImportanceResult, ModelInfoResult,
                      PredictionResult, RaceResult, WhatIfResult)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an analyst assistant for a 2026 Formula 1 podium predictor (an XGBoost model).
The model predicts the Sepang Grand Prix (round 16) using the qualifying grid and season form.

Rules:
1. Every number, ranking, probability or result you state must come from a tool result in this
   conversation. Never estimate, recall from memory, or compute new statistics yourself.
2. If a driver code, round or grid position is invalid, say so and use the corrected value from the
   tool error. Do not guess.
3. Probabilities are model outputs, not certainties. Never call an outcome guaranteed.
4. If the user's premise conflicts with tool results, say so plainly.
5. Only discuss the 2026 predictions and data. Decline anything else in one short sentence.
Keep answers concise. Refer to drivers by name."""


@dataclass
class HarnessDeps:
    predictor: Predictor
    tool_calls: list[dict] = field(default_factory=list)
    tool_errors: list[dict] = field(default_factory=list)
    grounding_retries: int = 0
    unverified: list[str] = field(default_factory=list)

    def results(self) -> list[dict]:
        return [c["result"] for c in self.tool_calls]


def build_model(model_id: str) -> OpenRouterModel:
    return OpenRouterModel(model_id, provider=OpenRouterProvider(api_key=openrouter_key()))


def build_agent(model) -> Agent[HarnessDeps, str]:
    # retries=2 sets both the tool and output budgets (installed pydantic-ai has no output_retries kwarg)
    agent = Agent(model, deps_type=HarnessDeps, output_type=str, instructions=SYSTEM_PROMPT,
                  retries=2)

    def run_tool(ctx: RunContext[HarnessDeps], name: str, args: dict, fn):
        try:
            result = fn(ctx.deps.predictor)
        except ModelRetry as e:
            ctx.deps.tool_errors.append({"name": name, "args": args, "error": str(e)})
            raise
        ctx.deps.tool_calls.append({"name": name, "args": args, "result": result.model_dump()})
        return result

    @agent.tool
    def list_drivers(ctx: RunContext[HarnessDeps]) -> DriversResult:
        """List every valid driver code on the Sepang grid with name, team and qualifying grid slot."""
        return run_tool(ctx, "list_drivers", {}, tools.list_drivers)

    @agent.tool
    def get_prediction(ctx: RunContext[HarnessDeps], top_n: int = 22) -> PredictionResult:
        """Predicted Sepang GP drivers ranked by podium probability (top 3 = predicted podium).
        top_n: how many drivers to return, 1-22."""
        return run_tool(ctx, "get_prediction", {"top_n": top_n},
                        lambda p: tools.get_prediction(p, top_n))

    @agent.tool
    def explain_driver(ctx: RunContext[HarnessDeps], driver: str) -> ExplainResult:
        """Why the model rates a driver as it does: top 5 SHAP features (positive shap pushes toward
        the podium, negative away). driver: three-letter code such as VER."""
        return run_tool(ctx, "explain_driver", {"driver": driver},
                        lambda p: tools.explain_driver(p, driver))

    @agent.tool
    def global_importance(ctx: RunContext[HarnessDeps]) -> ImportanceResult:
        """Which features matter most across the whole model (mean absolute SHAP)."""
        return run_tool(ctx, "global_importance", {}, tools.global_importance)

    @agent.tool
    def get_driver_form(ctx: RunContext[HarnessDeps], driver: str, last_n: int = 5) -> FormResult:
        """A driver's grid, finish, positions gained and status in their last_n completed races."""
        return run_tool(ctx, "get_driver_form", {"driver": driver, "last_n": last_n},
                        lambda p: tools.get_driver_form(p, driver, last_n))

    @agent.tool
    def get_race_result(ctx: RunContext[HarnessDeps], round_number: int) -> RaceResult:
        """Full classified result of a completed 2026 round (1-15)."""
        return run_tool(ctx, "get_race_result", {"round_number": round_number},
                        lambda p: tools.get_race_result(p, round_number))

    @agent.tool
    def what_if(ctx: RunContext[HarnessDeps], driver: str, grid_position: int) -> WhatIfResult:
        """Re-run the prediction as if the driver had qualified at grid_position (1-22). Returns
        podium probability and rank before and after, plus the change."""
        return run_tool(ctx, "what_if", {"driver": driver, "grid_position": grid_position},
                        lambda p: tools.what_if(p, driver, grid_position))

    @agent.tool
    def model_info(ctx: RunContext[HarnessDeps]) -> ModelInfoResult:
        """Training set size, rounds covered, best iteration and held-out accuracy / ROC-AUC."""
        return run_tool(ctx, "model_info", {}, tools.model_info)

    @agent.output_validator
    def grounding(ctx: RunContext[HarnessDeps], output: str) -> str:
        bad = unmatched_numbers(output, ctx.deps.results())
        if bad and ctx.deps.grounding_retries < 1:
            ctx.deps.grounding_retries += 1
            raise ModelRetry(f"These numbers appear in no tool result: {bad}. "
                             "Re-check with tools or remove them.")
        ctx.deps.unverified = bad
        return output

    return agent
