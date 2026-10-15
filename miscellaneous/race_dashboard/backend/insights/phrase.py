import json
import re

from .model import Candidate
from .templates import render

_NUM = re.compile(r"\d+(?:\.\d+)?")

SYSTEM = (
    "You write short, factual Formula 1 race insights for a live replay dashboard. You get 3 facts as JSON. "
    "Return JSON: {\"insights\": [{\"headline\": ..., \"detail\": ...}, ...]} with exactly 3 items in the same order. "
    "Headline: at most 8 words. Detail: one sentence of at most 25 words. Use only numbers that appear in the "
    "facts; never invent or recompute numbers. Refer to drivers by their three-letter codes. Projections must be "
    "worded as 'at the current rate'. Never mention DRS (2026 cars have none)."
)


def flatten(numbers: dict) -> list[float]:
    return [float(v) for v in numbers.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


def numbers_ok(text: str, numbers: dict) -> bool:
    allowed = flatten(numbers)
    return all(any(abs(float(m) - v) <= max(0.051, 0.01 * abs(v)) for v in allowed) for m in _NUM.findall(text))


def _item(c: Candidate, headline: str, detail: str, source: str) -> dict:
    return {"type": c.type, "drivers": list(c.drivers), "headline": headline, "detail": detail, "source": source}


def _templates(cands: list[Candidate]) -> list[dict]:
    return [_item(c, *render(c), "template") for c in cands]


def phrase_lap(cands: list[Candidate], chat) -> list[dict]:
    try:
        reply = chat(SYSTEM, json.dumps([{"type": c.type, **c.numbers} for c in cands]))
        items = reply["insights"]
        if not isinstance(items, list) or len(items) != len(cands):
            return _templates(cands)
    except Exception:
        return _templates(cands)
    out = []
    for c, item in zip(cands, items):
        head, detail = (item.get("headline"), item.get("detail")) if isinstance(item, dict) else (None, None)
        if c.numbers.get("kind") == "undercut":      # the LLM reads these as stops that happened; keep the fixed wording
            out.append(_item(c, *render(c), "template"))
        elif (isinstance(head, str) and isinstance(detail, str) and head.strip() and detail.strip()
                and numbers_ok(f"{head} {detail}", c.numbers)):
            out.append(_item(c, head.strip(), detail.strip(), "llm"))
        else:
            out.append(_item(c, *render(c), "template"))
    return out
