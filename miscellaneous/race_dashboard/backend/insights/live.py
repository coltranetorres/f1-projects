from . import config
from .client import chat_json
from .phrase import phrase_lap
from .ranker import score_with_jev


def jev_score_fn():
    key, model = config.openrouter_key(), config.ranker_model()
    return lambda t, lap, cands: score_with_jev(t, lap, cands, model=model, api_key=key)


def llm_phrase_fn():
    key, model = config.openrouter_key(), config.phrase_model()
    return lambda cands: phrase_lap(cands, lambda system, user: chat_json(model, system, user, key))
