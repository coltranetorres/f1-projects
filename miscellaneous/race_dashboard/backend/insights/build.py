import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .facts import candidates_for_lap
from .model import LapTable
from .select import select_top3

INSIGHTS_VERSION = 2  # bump when the insight content or ranking changes
COOLDOWN_LAPS = 4  # a fact shown in the last N laps is penalised unless its numbers changed
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def cache_path(year: int = 2026, event: str = "Singapore") -> Path:
    return CACHE_DIR / f"{year}_{event}_insights_v{INSIGHTS_VERSION}.json"


def _recent(history: list[dict]) -> dict:
    """Key -> numbers of its most recent appearance within the last COOLDOWN_LAPS laps."""
    memory = {}
    for shown in history[-COOLDOWN_LAPS:]:
        memory.update(shown)
    return memory


def _safe_scores(score_fn, t, lap, cands):
    try:
        return score_fn(t, lap, cands)
    except Exception:
        return None


def build_all(t: LapTable, score_fn, phrase_fn, *, laps=None, workers: int = 5) -> list[dict]:
    laps = list(laps) if laps is not None else list(range(1, t.max_lap + 1))
    cands = {lap: candidates_for_lap(t, lap) for lap in laps}
    with ThreadPoolExecutor(max_workers=workers) as pool:       # Jev calls are independent per lap
        scores = dict(zip(laps, pool.map(lambda l: _safe_scores(score_fn, t, l, cands[l]), laps)))
    chosen, history = {}, []
    for lap in laps:                                             # selection is sequential: the cooldown needs earlier laps
        chosen[lap] = select_top3(cands[lap], scores[lap], _recent(history))
        history.append({c.key: c.numbers for c in chosen[lap]})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        phrased = dict(zip(laps, pool.map(lambda l: phrase_fn(chosen[l]), laps)))
    return [{"lap": lap, "ranked_by": "jev" if scores[lap] is not None else "heuristic", "insights": phrased[lap]}
            for lap in laps]


def save_insights(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def load_insights(path: Path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, list) else None
