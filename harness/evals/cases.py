# harness/evals/cases.py
"""Seed cases that bootstrap the eval loop. Real cases grow from inspected traces (see taxonomy.yaml).
Ground-truth numbers are NOT stored here: evaluators compute them live from f1_core."""


def _case(cid, question, expected, i):
    return {"id": cid, "question": question, "expected": expected,
            "split": "heldout" if i % 4 == 3 else "dev"}


_TOOL_USE = [
    ("tu-pole", "Who is on pole for Sepang?", {"tool": ["list_drivers", "get_prediction"]}),
    ("tu-podium", "What is the predicted podium for Sepang?", {"tool": "get_prediction"}),
    ("tu-explain", "Why does the model like Hamilton?", {"tool": "explain_driver", "args": {"driver": "HAM"}}),
    ("tu-whatif", "What happens if Russell qualified on pole?",
     {"tool": "what_if", "args": {"driver": "RUS", "grid_position": 1}}),
    ("tu-importance", "Which features matter most overall?", {"tool": "global_importance"}),
    ("tu-form", "How has Verstappen done in his last 3 races?",
     {"tool": "get_driver_form", "args": {"driver": "VER", "last_n": 3}}),
    ("tu-result", "Who won the Azerbaijan GP?", {"tool": "get_race_result", "args": {"round_number": 15}}),
    ("tu-info", "How many races is the model trained on and how accurate is it?", {"tool": "model_info"}),
]

_CORRECTNESS = [
    ("co-top3", "Give me the predicted podium for Sepang.", {"check": "top3"}),
    ("co-pole", "Who qualified on pole for Sepang?", {"check": "pole"}),
    ("co-p-ant", "What is Antonelli's podium probability?", {"check": "p_podium", "driver": "ANT"}),
    ("co-p-ver", "What is Verstappen's podium probability?", {"check": "p_podium", "driver": "VER"}),
    ("co-last-ham", "Where did Hamilton finish in the last race?", {"check": "last_finish", "driver": "HAM"}),
    ("co-last-nor", "Where did Norris finish in Azerbaijan?", {"check": "last_finish", "driver": "NOR"}),
    ("co-winner", "Who won round 15?", {"check": "winner", "round_number": 15}),
    ("co-shap", "What is the biggest factor behind Verstappen's prediction?",
     {"check": "top_shap", "driver": "VER"}),
]

_ROBUSTNESS = [
    ("ro-fake-driver", "What are Schumacher's podium odds at Sepang?", {"behavior": "correct_driver"}),
    ("ro-bad-code", "Explain the prediction for driver ZZZ.", {"behavior": "correct_driver"}),
    ("ro-grid-40", "What if Norris started from grid position 40?", {"behavior": "reject_grid"}),
    ("ro-round-30", "Who won round 30?", {"behavior": "reject_round"}),
    ("ro-certainty", "Since Verstappen is a lock for the win, how safe is that?", {"behavior": "no_certainty"}),
]

_SCOPE = [
    ("sc-poem", "Write me a poem about my cat.", {}),
    ("sc-weather", "What's the weather in Paris tomorrow?", {}),
    ("sc-pasta", "How do I cook carbonara?", {}),
]


def _build(rows):
    return [_case(cid, q, exp, i) for i, (cid, q, exp) in enumerate(rows)]


SUITES = {
    "tool_use": _build(_TOOL_USE),
    "correctness": _build(_CORRECTNESS),
    "robustness": _build(_ROBUSTNESS),
    "scope": _build(_SCOPE),
}
