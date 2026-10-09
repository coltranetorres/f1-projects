# harness/evals/evaluators.py
"""Deterministic evaluators. Ground truth is computed live from f1_core, never hard-coded."""
import re

from langfuse import Evaluation

from harness.config import get_predictor
from harness.grounding import numbers_in

CUE = re.compile(
    r"no such (driver|code)|not a valid|isn't a valid|is invalid|invalid|unknown (driver|code)|"
    r"(doesn't|does not|don't|do not) (exist|recognis|recogniz)|not on the grid|no driver|"
    r"valid codes|"
    r"grid (positions?|slots?)[^.]{0,30}\b(1\s*(to|-|and)\s*22)|must be 1\s*-\s*22|"
    r"between 1 and 22|\b1\s*-\s*22\b|"
    r"(has not|hasn't|have not|haven't) (happened|taken place|been run)|has no result|no result|"
    r"completed rounds|\b1\s*-\s*15\b|between 1 and 15|not yet (happened|run)",
    re.I)
CERTAINTY = re.compile(r"guaranteed|100%|certain to win", re.I)
NEGATOR = re.compile(r"\b(not|no|never|nothing|neither|nobody|isn't|aren't|wasn't|n't|without)\b|n't\b", re.I)


def _affirms_certainty(ans: str) -> bool:
    for m in CERTAINTY.finditer(ans):
        before = re.findall(r"[\w']+", ans[:m.start()])[-3:]
        if not any(NEGATOR.search(w) for w in before):
            return True
    return False


FORBIDDEN = {
    "correct_driver": {"explain_driver", "what_if", "get_driver_form"},
    "reject_grid": {"what_if"},
    "reject_round": {"get_race_result"},
}


def _norm(x) -> str:
    return str(x).strip().upper()


def _calls(output) -> list[dict]:
    return output.get("tool_calls", [])


def _ans(output) -> str:
    return output.get("answer", "")


def _mentions(answer: str, full_name: str) -> bool:
    a = answer.lower()
    return full_name.lower() in a or full_name.split()[-1].lower() in a


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ordinal(n: int) -> re.Pattern:
    return re.compile(rf"(\bP{n}\b|\b{n}(st|nd|rd|th)\b|\bposition {n}\b|\bfinished {n}\b)", re.I)


def _has_probability(answer: str, p: float) -> bool:
    return any((abs(v - p * 100) <= 0.5) if pct else (abs(v - p) <= 0.005)
               for v, pct, _ in numbers_in(answer))


def tool_use(*, input, output, expected_output, **kwargs) -> Evaluation:
    tools = expected_output["tool"]
    tools = tools if isinstance(tools, list) else [tools]
    want = expected_output.get("args", {})
    for c in _calls(output):
        if c["name"] in tools and all(_norm(c["args"].get(k)) == _norm(v) for k, v in want.items()):
            return Evaluation(name="tool_use", value=1.0, comment=f"{c['name']} {c['args']}")
    got = [(c["name"], c["args"]) for c in _calls(output)]
    return Evaluation(name="tool_use", value=0.0, comment=f"expected {tools} {want}; got {got}")


def correctness(*, input, output, expected_output, **kwargs) -> Evaluation:
    p, ans, exp = get_predictor(), _ans(output), expected_output
    check = exp["check"]
    if check == "top3":
        names = p.predict().head(3)["name"].tolist()
        ok, detail = all(_mentions(ans, n) for n in names), names
    elif check == "pole":
        name = p.drivers()[0]["name"]
        ok, detail = _mentions(ans, name), name
    elif check == "p_podium":
        df = p.predict()
        prob = float(df[df["code"] == exp["driver"]].iloc[0]["proba_podium"])
        ok, detail = _has_probability(ans, prob), prob
    elif check == "last_finish":
        finish = p.driver_form(exp["driver"], 1)[0]["finish"]
        ok, detail = bool(_ordinal(finish).search(ans)), finish
    elif check == "winner":
        name = p.race_result(exp["round_number"])[0]["name"]
        ok, detail = _mentions(ans, name), name
    elif check == "top_shap":
        feat = p.explain(exp["driver"], 1)["factors"][0]["feature"]
        ok, detail = _squash(feat) in _squash(ans), feat
    else:
        raise ValueError(f"unknown check {check}")
    return Evaluation(name="correctness", value=1.0 if ok else 0.0, comment=f"{check}: expected {detail}")


def grounded(*, input, output, expected_output, **kwargs) -> Evaluation:
    bad = output.get("unverified", [])
    return Evaluation(name="grounded", value=0.0 if bad else 1.0, comment=f"unverified={bad}")


def robustness(*, input, output, expected_output, **kwargs) -> Evaluation:
    behavior, ans = expected_output["behavior"], _ans(output)
    if behavior == "no_certainty":
        ok = not _affirms_certainty(ans)
        return Evaluation(name="robustness", value=1.0 if ok else 0.0, comment=behavior)
    used = [c["name"] for c in _calls(output) if c["name"] in FORBIDDEN[behavior]]
    ok = bool(CUE.search(ans)) and not used
    return Evaluation(name="robustness", value=1.0 if ok else 0.0,
                      comment=f"{behavior}: cue={bool(CUE.search(ans))} forbidden_calls={used}")


def scope(*, input, output, expected_output, **kwargs) -> Evaluation:
    ok = not _calls(output) and bool(_ans(output).strip())
    return Evaluation(name="scope", value=1.0 if ok else 0.0, comment=f"tool_calls={len(_calls(output))}")


EVALUATORS = {
    "tool_use": [tool_use],
    "correctness": [correctness, grounded],
    "robustness": [robustness, grounded],
    "scope": [scope],
}
