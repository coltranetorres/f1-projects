import pytest

from race_dashboard.backend.insights.ranker import (CRITERIA, build_questions, build_state, parse_scores,
                                                    score_with_jev)
from race_dashboard.backend.tests.insights_fixtures import cand, series, table


def test_questions_are_one_score_question_per_candidate():
    cs = [cand("pace", "p", 0.5, {"driver": "A"}), cand("battle", "b", 0.5, {"gap_s": 1.1})]
    q = build_questions(7, cs)
    assert list(q) == ["c0", "c1"]
    assert q["c0"]["type"] == "score" and q["c0"]["criteria"] == CRITERIA
    assert '"driver": "A"' in q["c0"]["instructions"] and "lap 7" in q["c0"]["instructions"]


def test_state_summarises_the_order_and_the_candidates():
    t = table(series("A", 3, 90.0, pos=1), series("B", 3, 90.5, pos=2, offset=2.0))
    s = build_state(t, 3, [cand("pace", "p", 0.5)])
    assert "Lap 3 of 10" in s and "1. A" in s and "2. B" in s and "DRS" not in s


def test_parse_probability_distribution():
    body = {"answers": {"c0": {"score": 2, "probabilities": [0, 0, 1, 0]},
                        "c1": {"score": 0, "probabilities": {"Routine": 0.5, "Mildly interesting": 0.5,
                                                              "Notable": 0, "Race-defining": 0}}}}
    got = parse_scores(body, 2)
    assert got[0] == pytest.approx(2 / 3) and got[1] == pytest.approx(0.5 / 3)


def test_parse_real_jev_shape_with_index_keyed_probabilities_in_any_order():
    body = {"answers": {"c0": {"type": "score", "score": 0.88, "confidence": 0.62,
                               "legend": {"0": "Routine", "1": "Mildly interesting", "2": "Notable", "3": "Race-defining"},
                               "probabilities": {"3": 0, "1": 0.62, "0": 0.25, "2": 0.13}}}}
    assert parse_scores(body, 1)[0] == pytest.approx(0.88 / 3)


def test_parse_plain_score_is_the_criteria_index():
    assert parse_scores({"answers": {"c0": {"score": 3}}}, 1) == {0: pytest.approx(1.0)}


@pytest.mark.parametrize("body", [{}, {"answers": {}}, {"answers": {"c0": {"score": 1}}}, {"answers": {"c0": {}, "c1": {}}},
                                  {"answers": {"c0": {"score": "high"}, "c1": {"score": 1}}}, None])
def test_parse_rejects_missing_or_malformed_answers(body):
    assert parse_scores(body, 2) is None


class Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_score_with_jev_happy_path_and_payload():
    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json)
        return Resp({"answers": {"c0": {"score": 3}}})

    t = table(series("A", 3, 90.0), series("B", 3, 90.0, pos=2, offset=1.0))
    got = score_with_jev(t, 3, [cand("pace", "p", 0.5)], model="typesafe/x", api_key="K", post=post)
    assert got == {0: pytest.approx(1.0)}
    assert seen["url"].endswith("/api/alpha/decisions") and seen["json"]["model"] == "typesafe/x"
    assert "state" in seen["json"] and "c0" in seen["json"]["questions"]


def test_score_with_jev_never_raises():
    def boom(**kw):
        raise OSError("down")

    t = table(series("A", 3, 90.0))
    assert score_with_jev(t, 3, [cand("pace", "p", 0.5)], model="m", api_key="K", post=boom) is None
    assert score_with_jev(t, 3, [], model="m", api_key="K", post=boom) is None
