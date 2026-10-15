from race_dashboard.backend.insights.select import changed, select_top3
from race_dashboard.backend.tests.insights_fixtures import cand


def keys(cs):
    return [c.key for c in cs]


def test_one_per_type_then_best_of_the_rest():
    cs = [cand("pace", "p1", 0.9), cand("pace", "p2", 0.8), cand("battle", "b1", 0.5), cand("tyres", "t1", 0.4)]
    assert keys(select_top3(cs, None, {})) == ["p1", "b1", "t1"]


def test_fills_to_three_when_types_run_out():
    cs = [cand("pace", f"p{i}", 0.9 - i / 10) for i in range(4)]
    assert keys(select_top3(cs, None, {})) == ["p0", "p1", "p2"]


def test_fewer_than_three_candidates_returns_what_exists():
    assert keys(select_top3([cand("pace", "p", 0.5)], None, {})) == ["p"]


def test_jev_scores_override_heuristic_scores():
    cs = [cand("pace", "p1", 0.9), cand("battle", "b1", 0.1), cand("tyres", "t1", 0.2), cand("pit", "x", 0.3)]
    assert keys(select_top3(cs, {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.2}, {})) == ["b1", "t1", "x"]


def test_heuristic_score_breaks_jev_ties():
    cs = [cand("pace", "p1", 0.2), cand("battle", "b1", 0.9)]
    assert keys(select_top3(cs, {0: 0.5, 1: 0.5}, {})) == ["b1", "p1"]


def test_unchanged_repeat_is_penalised_but_a_change_is_not():
    cs = [cand("pace", "p1", 0.9, {"gap_s": 1.0}), cand("battle", "b1", 0.5, {"gap_s": 1.0})]
    assert keys(select_top3(cs, None, {"p1": {"kind": "pace", "gap_s": 1.0}})) == ["b1", "p1"]
    assert keys(select_top3(cs, None, {"p1": {"kind": "pace", "gap_s": 2.0}})) == ["p1", "b1"]


def test_changed_needs_a_meaningful_difference():
    assert not changed({"gap_s": 1.0, "lap": 3}, {"gap_s": 1.2, "lap": 4})
    assert changed({"gap_s": 1.0}, {"gap_s": 1.4})
    assert changed({"gap_s": 10.0}, {"gap_s": 11.6})
    assert changed({"driver": "A"}, {"driver": "B"})


def test_window_and_stint_counters_alone_are_not_a_change():
    assert not changed({"window_laps": 1, "stint_laps": 4, "gap_s": 1.0}, {"window_laps": 2, "stint_laps": 5, "gap_s": 1.0})
