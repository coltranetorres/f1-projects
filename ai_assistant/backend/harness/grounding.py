# harness/grounding.py
"""Heuristic: flag percentages/decimals in an answer that no tool result supports.

Catches invented numbers, not misinterpretation (evals cover that). Derived numbers the
model computes itself (e.g. a difference) will be flagged too — tools should return them.
"""
import re

_NUM = re.compile(r"(?<![\w.])(-?\d+\.\d+|\d+(?=%))(%?)")
PCT_TOL = 0.5
DEC_TOL = 0.005


def numbers_in(answer: str) -> list[tuple[float, bool, str]]:
    return [(float(m.group(1)), m.group(2) == "%", m.group(0)) for m in _NUM.finditer(answer)]


def _flatten(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)


def unmatched_numbers(answer: str, tool_results: list[dict]) -> list[str]:
    allowed = [abs(x) for r in tool_results for x in _flatten(r)]
    bad = []
    for value, is_pct, text in numbers_in(answer):
        v = abs(value)
        if is_pct:
            ok = any(abs(v - a * 100) <= PCT_TOL or abs(v - a) <= PCT_TOL for a in allowed)
        else:
            ok = any(abs(v - a) <= DEC_TOL for a in allowed)
        if not ok:
            bad.append(text)
    return bad
