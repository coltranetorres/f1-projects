from race_dashboard.backend.insights.facts import candidates_for_lap
from race_dashboard.backend.insights.model import Candidate
from race_dashboard.backend.insights.phrase import flatten, numbers_ok, phrase_lap
from race_dashboard.backend.insights.templates import render
from race_dashboard.backend.tests.insights_fixtures import series, table


def C(type_, **numbers):
    return Candidate(type_, "k", ("AAA",), numbers, 0.5)


THREE = [C("battle", kind="battle", ahead="AAA", behind="BBB", gap_s=0.8, closing_s_per_lap=0.3, laps_to_close=3, lap=6),
         C("pace", kind="fastest", driver="AAA", pace_s=89.5, next="BBB", margin_s=0.5, lap=6),
         C("tyres", kind="age", driver="BBB", compound="SOFT", tyre_age=15, median_age=6.0, lap=6)]

DETAILS = ["BBB is 0.8s behind AAA, 3 laps to close.", "AAA averages 89.5s, 0.5s clear of BBB.",
           "BBB runs 15 lap old SOFT tyres against a median of 6."]


def good(n=3):
    return {"insights": [{"headline": "Headline", "detail": DETAILS[i]} for i in range(n)]}


def test_flatten_and_numbers_ok():
    nums = {"gap_s": 0.8, "lap": 6, "driver": "AAA", "flag": True}
    assert sorted(flatten(nums)) == [0.8, 6]
    assert numbers_ok("Gap is 0.8s on lap 6", nums)
    assert numbers_ok("Gap is 0.79s", nums)          # rounding tolerance
    assert not numbers_ok("Gap is 1.4s", nums)       # invented number
    assert numbers_ok("No digits here", nums)


def test_phrase_lap_uses_the_llm_when_valid():
    out = phrase_lap(THREE, lambda system, user: good())
    assert [o["source"] for o in out] == ["llm"] * 3
    assert out[0]["type"] == "battle" and out[0]["drivers"] == ["AAA"] and out[0]["headline"] == "Headline"


def test_invalid_numbers_fall_back_to_template_for_that_item_only():
    bad = good()
    bad["insights"][1]["detail"] = "AAA is 9.9s clear."
    out = phrase_lap(THREE, lambda s, u: bad)
    assert [o["source"] for o in out] == ["llm", "template", "llm"]
    assert "89.5" in out[1]["detail"]


def test_wrong_count_non_json_or_exception_fall_back_to_templates():
    for chat in (lambda s, u: good(2), lambda s, u: {"nope": 1}, lambda s, u: (_ for _ in ()).throw(RuntimeError("x"))):
        assert [o["source"] for o in phrase_lap(THREE, chat)] == ["template"] * 3


def test_prompt_forbids_drs_and_invented_numbers():
    seen = {}
    phrase_lap(THREE, lambda system, user: seen.update(system=system, user=user) or good())
    assert "DRS" in seen["system"] and "only" in seen["system"].lower() and '"gap_s": 0.8' in seen["user"]


def test_stop_template_survives_missing_optional_fields():
    head, detail = render(C("pit", kind="stop", driver="AAA", lap=4))
    assert "AAA" in head and "lap 4" in detail


def test_every_generated_candidate_has_a_template():
    t = table(series("A", 6, 90.0, step=0.1, pos=1), series("B", 6, 89.5, pos=2, offset=3.8, life0=12),
              series("C", 6, 90.0, pos=3, offset=4.5), series("D", 6, 90.0, pos=4, offset=30.0, life0=14))
    for c in candidates_for_lap(t, 6):
        head, detail = render(c)
        assert head and detail and "DRS" not in head + detail


def test_undercut_is_always_template_text_so_no_stop_is_invented():
    u = C("pit", kind="undercut", ahead="AAA", behind="BBB", gap_s=1.5, behind_tyre_age=19, ahead_tyre_age=6,
          pit_loss_s=25.0, lap=27)
    chat = lambda s, user: {"insights": [{"headline": "BBB pits on lap 27", "detail": "BBB pits on lap 27 for fresh tyres."}]}
    (o,) = phrase_lap([u], chat)
    assert o["source"] == "template" and "pits on lap" not in o["detail"]
