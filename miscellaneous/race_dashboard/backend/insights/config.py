import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(REPO_ROOT / ".env")


def _require(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(f"Set {names[0]} in .env")


def openrouter_key() -> str:
    return _require("OPENROUTER_API_KEY", "OPENROUTER_APIKEY")


def ranker_model() -> str:
    return _require("F1_RANKER_MODEL")


def phrase_model() -> str:
    return _require("F1_DEFAULT_MODEL")
