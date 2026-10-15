import json

from .client import post_json
from .model import Candidate, LapTable

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CRITERIA = ["Routine", "Mildly interesting", "Notable", "Race-defining"]


def build_state(t: LapTable, lap: int, cands: list[Candidate]) -> str:
    lines = [f"Lap {lap} of {t.total_laps} of the 2026 Singapore Grand Prix. Running order after the lap:"]
    for i, d in enumerate(t.order(lap)[:10], 1):
        gap, info, pace = t.gap_ahead(d, lap), t.tyre(d, lap), t.recent_pace(d, lap)
        bits = [f"{i}. {d}"]
        if gap is not None:
            bits.append(f"+{gap:.1f}s to car ahead")
        if pace is not None:
            bits.append(f"recent pace {pace:.2f}s")
        if info is not None:
            bits.append(f"{info[0]} tyres, {info[1]} laps old")
        lines.append(", ".join(bits))
    lines.append("Candidate facts:")
    lines += [f"- {c.type}: {json.dumps(c.numbers, sort_keys=True)}" for c in cands]
    return "\n".join(lines)


def build_questions(lap: int, cands: list[Candidate]) -> dict:
    return {
        f"c{i}": {
            "type": "score",
            "instructions": (f"For a viewer watching lap {lap}, how notable is this {c.type} fact? "
                             f"Fact: {json.dumps(c.numbers, sort_keys=True)}"),
            "criteria": CRITERIA,
        }
        for i, c in enumerate(cands)
    }


def _expected(answer) -> float | None:
    k = len(CRITERIA)
    if not isinstance(answer, dict):
        return None
    probs = answer.get("probabilities")
    if isinstance(probs, dict):
        if all(l in probs for l in CRITERIA):
            probs = [probs[l] for l in CRITERIA]
        elif all(str(i) in probs for i in range(k)):      # the live API keys the distribution by "0".."3"
            probs = [probs[str(i)] for i in range(k)]
        else:
            probs = list(probs.values())
    if isinstance(probs, list) and len(probs) == k and all(isinstance(p, (int, float)) for p in probs):
        total = sum(probs)
        return sum(i * p for i, p in enumerate(probs)) / total / (k - 1) if total > 0 else None
    score = answer.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return max(0.0, min(1.0, score / (k - 1)))
    return None


def parse_scores(body, n: int) -> dict[int, float] | None:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        return None
    out = {}
    for i in range(n):
        v = _expected(answers.get(f"c{i}"))
        if v is None:
            return None
        out[i] = v
    return out


def score_with_jev(t, lap, cands, *, model, api_key, post=None) -> dict[int, float] | None:
    if not cands:
        return None
    try:
        body = post_json(DECISIONS_URL, {"model": model, "state": build_state(t, lap, cands),
                                         "questions": build_questions(lap, cands)}, api_key, post=post)
        return parse_scores(body, len(cands))
    except Exception:
        return None
