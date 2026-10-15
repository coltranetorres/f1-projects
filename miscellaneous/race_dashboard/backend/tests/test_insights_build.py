import json

from race_dashboard.backend.insights.build import (COOLDOWN_LAPS, INSIGHTS_VERSION, _recent, build_all, cache_path, load_insights,
                                                   save_insights)
from race_dashboard.backend.insights.templates import render
from race_dashboard.backend.tests.insights_fixtures import series, table


def small_table():
    # grid differs from the running order so lap 1 has mover candidates, as the real start does
    return table(series("A", 6, 90.0, step=0.1, pos=1), series("B", 6, 89.5, pos=2, offset=3.8),
                 series("C", 6, 90.0, pos=3, offset=5.0), series("D", 6, 90.0, pos=4, offset=40.0, life0=12),
                 grid={"A": 3, "B": 4, "C": 1, "D": 2})


def phrase(cands):
    return [{"type": c.type, "drivers": list(c.drivers), "headline": render(c)[0], "detail": render(c)[1],
             "source": "template"} for c in cands]


def test_every_lap_gets_three_insights():
    out = build_all(small_table(), lambda t, lap, cands: None, phrase, workers=2)
    assert [e["lap"] for e in out] == [1, 2, 3, 4, 5, 6]
    assert all(len(e["insights"]) == 3 and e["ranked_by"] == "heuristic" for e in out)


def test_jev_scores_are_used_and_marked():
    def score(t, lap, cands):
        return {i: (1.0 if c.type == "tyres" else 0.1) for i, c in enumerate(cands)}

    out = build_all(small_table(), score, phrase, laps=[6])
    assert out[0]["ranked_by"] == "jev"
    assert out[0]["insights"][0]["type"] == "tyres"


def test_a_failing_scorer_degrades_to_heuristic_without_aborting():
    def score(t, lap, cands):
        raise RuntimeError("jev down")

    out = build_all(small_table(), score, phrase, laps=[3, 4])
    assert [e["ranked_by"] for e in out] == ["heuristic", "heuristic"]


def test_repeat_cooldown_changes_the_set_across_laps():
    out = build_all(small_table(), lambda t, lap, cands: None, phrase)
    sets = [tuple(tuple(i["drivers"]) for i in e["insights"]) for e in out]
    assert len(set(sets)) > 1


def test_cache_round_trip_and_bad_files(tmp_path):
    p = tmp_path / "x.json"
    assert load_insights(p) is None
    save_insights(p, [{"lap": 1, "insights": [], "ranked_by": "jev"}])
    assert load_insights(p) == [{"lap": 1, "insights": [], "ranked_by": "jev"}]
    p.write_text("{not json")
    assert load_insights(p) is None
    p.write_text(json.dumps({"lap": 1}))
    assert load_insights(p) is None


def test_cache_path_is_versioned():
    assert cache_path().name == f"2026_Singapore_insights_v{INSIGHTS_VERSION}.json"


def test_cooldown_remembers_the_last_few_laps_not_just_one():
    history = [{"old": {"gap_s": 1.0}}] + [{f"k{i}": {"gap_s": 1.0}} for i in range(COOLDOWN_LAPS)]
    memory = _recent(history)
    assert "old" not in memory                                   # fell out of the window
    assert all(f"k{i}" in memory for i in range(COOLDOWN_LAPS))  # includes laps further back than the previous one
    assert _recent([{"a": {"gap_s": 1.0}}, {"a": {"gap_s": 2.0}}])["a"] == {"gap_s": 2.0}  # most recent wins
