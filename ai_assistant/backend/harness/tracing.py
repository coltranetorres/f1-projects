import os
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .agent import PROMPT_VERSION

_ENABLED = False
_REPO = Path(__file__).resolve().parent.parent


class _NullSpan:
    trace_id = None

    def update(self, **kwargs):
        pass


def tracing_enabled() -> bool:
    return _ENABLED


def init_tracing() -> bool:
    """Enable Langfuse + Pydantic AI OpenTelemetry instrumentation when env vars are present."""
    global _ENABLED
    load_dotenv()
    if os.environ.get("F1_DISABLE_TRACING"):
        return False
    needed = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    if not all(os.environ.get(k) for k in needed):
        return False
    from langfuse import get_client
    from pydantic_ai import Agent
    get_client()  # initialises the OpenTelemetry provider that Langfuse listens on
    Agent.instrument_all()
    _ENABLED = True
    return True


def flush() -> None:
    if _ENABLED:
        from langfuse import get_client
        get_client().flush()


@lru_cache(maxsize=1)
def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@contextmanager
def chat_trace(message: str, session_id: str | None, model_id: str):
    if not _ENABLED:
        yield _NullSpan()
        return
    from langfuse import get_client, propagate_attributes
    lf = get_client()
    with lf.start_as_current_observation(as_type="span", name="chat", input=message) as span:
        with propagate_attributes(
            session_id=session_id or "default",
            metadata={"model": model_id, "git_sha": _git_sha(),
                      "prompt_version": PROMPT_VERSION, "target_gp": "sepang"},
        ):
            yield span
