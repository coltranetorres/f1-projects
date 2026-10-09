# tests/test_detectors.py
import pytest

from harness.evals import evaluators as ev
from harness.evals.cases import SUITES


def out(answer, calls=(), unverified=()):
    return {"answer": answer,
            "tool_calls": [{"name": n, "args": a, "result": {}} for n, a in calls],
            "tool_errors": [], "unverified": list(unverified), "capped": False}


def score(fn, output, expected):
    return fn(input={"question": "q"}, output=output, expected_output=expected).value


# ── tool_use ────────────────────────────────────────────────────────────────
def test_tool_use_pass_and_case_insensitive_args():
    exp = {"tool": "what_if", "args": {"driver": "RUS", "grid_position": 1}}
    assert score(ev.tool_use, out("x", [("what_if", {"driver": "rus", "grid_position": 1})]), exp) == 1.0


def test_tool_use_fail_wrong_tool_or_args():
    exp = {"tool": "what_if", "args": {"driver": "RUS", "grid_position": 1}}
    assert score(ev.tool_use, out("x", [("explain_driver", {"driver": "RUS"})]), exp) == 0.0
    assert score(ev.tool_use, out("x", [("what_if", {"driver": "RUS", "grid_position": 3})]), exp) == 0.0
    assert score(ev.tool_use, out("x"), exp) == 0.0


def test_tool_use_accepts_any_of_listed_tools():
    exp = {"tool": ["list_drivers", "get_prediction"]}
    assert score(ev.tool_use, out("x", [("get_prediction", {"top_n": 3})]), exp) == 1.0


# ── correctness (ground truth computed live from f1_core) ───────────────────
def test_correctness_top3(predictor):
    names = predictor.predict().head(3)["name"].tolist()
    assert score(ev.correctness, out("Podium: " + ", ".join(names)), {"check": "top3"}) == 1.0
    assert score(ev.correctness, out("I am not sure."), {"check": "top3"}) == 0.0


def test_correctness_p_podium(predictor):
    row = predictor.predict().query("code == 'VER'").iloc[0]
    good = f"{row['proba_podium'] * 100:.1f}%"
    exp = {"check": "p_podium", "driver": "VER"}
    assert score(ev.correctness, out(f"Verstappen: {good}"), exp) == 1.0
    assert score(ev.correctness, out("Verstappen: 12.3%"), exp) == 0.0


def test_correctness_last_finish_and_winner(predictor):
    assert score(ev.correctness, out("Hamilton finished 6th."), {"check": "last_finish", "driver": "HAM"}) == 1.0
    assert score(ev.correctness, out("Hamilton finished 2nd."), {"check": "last_finish", "driver": "HAM"}) == 0.0
    assert score(ev.correctness, out("George Russell won."), {"check": "winner", "round_number": 15}) == 1.0
    assert score(ev.correctness, out("Max Verstappen won."), {"check": "winner", "round_number": 15}) == 0.0


def test_correctness_top_shap_feature(predictor):
    feat = predictor.explain("VER", 1)["factors"][0]["feature"]
    spaced = feat.replace("_", " ")
    assert score(ev.correctness, out(f"Mostly his {spaced}."), {"check": "top_shap", "driver": "VER"}) == 1.0
    assert score(ev.correctness, out("Mostly vibes."), {"check": "top_shap", "driver": "VER"}) == 0.0


def test_grounded_detector():
    assert ev.grounded(input=None, output=out("a", unverified=[]), expected_output=None).value == 1.0
    assert ev.grounded(input=None, output=out("a", unverified=["87%"]), expected_output=None).value == 0.0


# ── robustness ──────────────────────────────────────────────────────────────
def test_robustness_corrects_unknown_driver():
    exp = {"behavior": "correct_driver"}
    assert score(ev.robustness, out("There is no such driver on the grid."), exp) == 1.0
    assert score(ev.robustness, out("Sure, 40%.", [("explain_driver", {"driver": "SAI"})]), exp) == 0.0
    assert score(ev.robustness, out("Sure, here are the odds."), exp) == 0.0  # no correction cue


def test_robustness_rejects_impossible_grid_and_round():
    assert score(ev.robustness, out("Grid slots only go from 1 to 22."), {"behavior": "reject_grid"}) == 1.0
    assert score(ev.robustness, out("Ok.", [("what_if", {"driver": "NOR", "grid_position": 22})]),
                 {"behavior": "reject_grid"}) == 0.0
    assert score(ev.robustness, out("That round has not happened; results exist for 1-15."),
                 {"behavior": "reject_round"}) == 1.0


def test_robustness_no_certainty():
    exp = {"behavior": "no_certainty"}
    assert score(ev.robustness, out("It is likely but not certain: about 99%."), exp) == 1.0
    assert score(ev.robustness, out("Yes, his win is guaranteed."), exp) == 0.0


# ── scope ───────────────────────────────────────────────────────────────────
def test_scope_declines_without_tools():
    assert score(ev.scope, out("I can only discuss the 2026 F1 predictions."), {}) == 1.0
    assert score(ev.scope, out("Roses are red", [("list_drivers", {})]), {}) == 0.0
    assert score(ev.scope, out(""), {}) == 0.0


# ── seed cases are well-formed ──────────────────────────────────────────────
def test_seed_cases_wellformed():
    ids = [c["id"] for cases in SUITES.values() for c in cases]
    assert len(ids) == len(set(ids)) >= 24
    assert set(SUITES) == set(ev.EVALUATORS)
    for cases in SUITES.values():
        assert all(c["split"] in ("dev", "heldout") and c["question"] for c in cases)
    assert any(c["split"] == "heldout" for cases in SUITES.values() for c in cases)


# ── fix round 1: negation-aware certainty, specific correction cues ─────────
@pytest.mark.parametrize("ans", [
    "Nothing is guaranteed in F1; the model gives about 99%.",
    "It isn't guaranteed, just likely.",
])
def test_no_certainty_accepts_negated_pushback(ans):
    assert score(ev.robustness, out(ans), {"behavior": "no_certainty"}) == 1.0


@pytest.mark.parametrize("ans", [
    "Yes, his win is guaranteed.",
    "He is certain to win.",
    "It is 100% certain.",
])
def test_no_certainty_rejects_affirmations(ans):
    assert score(ev.robustness, out(ans), {"behavior": "no_certainty"}) == 0.0


@pytest.mark.parametrize("ans,behavior", [
    ("Unknown driver code 'ZZZ'. Valid codes: VER, HAM.", "correct_driver"),
    ("grid_position must be 1-22, got 40.", "reject_grid"),
    ("Round 30 has no result. Completed rounds: 1-15.", "reject_round"),
    ("There is no such driver on the grid.", "correct_driver"),
])
def test_robustness_cues_echo_tool_errors(ans, behavior):
    assert score(ev.robustness, out(ans), {"behavior": behavior}) == 1.0


@pytest.mark.parametrize("ans,behavior", [
    ("Here are the odds; I only have the top 5, not more.", "correct_driver"),
    ("Schumacher is between 5th and 8th in my estimate.", "correct_driver"),
    ("I don't mind: he starts 40th and gains places.", "reject_grid"),
    ("Sure, here are the odds.", "reject_round"),
])
def test_robustness_complying_answers_fail(ans, behavior):
    assert score(ev.robustness, out(ans), {"behavior": behavior}) == 0.0
