from .model import Candidate

REPEAT_PENALTY = 0.5
IGNORED = {"lap", "window_laps", "stint_laps"}


def changed(prev: dict, cur: dict) -> bool:
    """True when any field differs meaningfully: numbers by max(0.3, 15% of the old value), others by value."""
    for k, v in cur.items():
        if k in IGNORED or k not in prev:
            continue
        old = prev[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(old, (int, float)):
            if abs(v - old) >= max(0.3, 0.15 * abs(old)):
                return True
        elif v != old:
            return True
    return False


def select_top3(cands: list[Candidate], jev: dict[int, float] | None, prev: dict[str, dict]) -> list[Candidate]:
    def effective(i: int) -> float:
        c = cands[i]
        base = jev[i] if jev is not None else c.score
        if c.key in prev and not changed(prev[c.key], c.numbers):
            base *= REPEAT_PENALTY
        return base

    order = sorted(range(len(cands)), key=lambda i: (-effective(i), -cands[i].score))
    chosen, types = [], set()
    for i in order:
        if cands[i].type not in types:
            chosen.append(i)
            types.add(cands[i].type)
        if len(chosen) == 3:
            break
    for i in order:
        if len(chosen) == 3:
            break
        if i not in chosen:
            chosen.append(i)
    return [cands[i] for i in chosen]
