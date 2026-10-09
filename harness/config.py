import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = ["anthropic/claude-sonnet-4.6", "openai/gpt-4.1-mini", "google/gemini-2.5-flash"]


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_APIKEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENROUTER_APIKEY) in .env")
    return key


def default_model() -> str:
    return os.environ.get("F1_DEFAULT_MODEL", DEFAULT_MODELS[0])


def available_models() -> list[str]:
    raw = os.environ.get("F1_MODELS")
    return [m.strip() for m in raw.split(",") if m.strip()] if raw else list(DEFAULT_MODELS)


@lru_cache(maxsize=1)
def get_predictor():
    from f1_core import Predictor
    return Predictor()
