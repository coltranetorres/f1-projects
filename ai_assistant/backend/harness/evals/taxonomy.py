# harness/evals/taxonomy.py
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_PATH = Path(__file__).parent / "taxonomy.yaml"

Suite = Literal["tool_use", "correctness", "robustness", "scope", "agent_loop"]
Layer = Literal["tool_selection", "tool_arguments", "tool_implementation", "grounding",
                "interpretation", "scope", "loop_control", "prompt"]


class FailureMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    suite: Suite
    secondary_suite: Suite | None = None
    layer: Layer
    tool: str | None = None
    evidence: list[str] = []
    detector: str | None = None
    resolution: str | None = None
    status: Literal["observed", "fixed", "regressed"] = "observed"


class Taxonomy(BaseModel):
    failure_modes: list[FailureMode] = []


def load_taxonomy(path: Path = DEFAULT_PATH) -> Taxonomy:
    raw = yaml.safe_load(path.read_text()) or {}
    tax = Taxonomy(failure_modes=raw.get("failure_modes") or [])
    ids = [f.id for f in tax.failure_modes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate failure mode ids: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    return tax
