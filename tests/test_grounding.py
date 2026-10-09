# tests/test_grounding.py
from harness.grounding import numbers_in, unmatched_numbers

RESULTS = [{"rows": [{"code": "VER", "p_podium": 0.9957, "grid": 1}], "shap": -0.4123}]


def test_percent_matches_probability():
    assert unmatched_numbers("Verstappen has a 99.6% chance.", RESULTS) == []


def test_invented_percent_is_flagged():
    assert unmatched_numbers("He has an 87.3% chance.", RESULTS) == ["87.3%"]


def test_decimal_probability_and_rounding():
    assert unmatched_numbers("p = 0.9957 or about 1.00", RESULTS) == []
    assert unmatched_numbers("p = 0.8123", RESULTS) == ["0.8123"]


def test_negative_shap_matches_by_magnitude():
    assert unmatched_numbers("SHAP of -0.41 pushes him down", RESULTS) == []


def test_plain_integers_are_ignored():
    assert unmatched_numbers("He starts P1 of 22 in round 16 with 3 stints.", RESULTS) == []


def test_numbers_in_extracts_pct_and_decimals():
    got = numbers_in("About 12% or 0.25, from P3")
    assert [(v, pct) for v, pct, _ in got] == [(12.0, True), (0.25, False)]


def test_no_tool_results_flags_everything():
    assert unmatched_numbers("Chance is 50%.", []) == ["50%"]
