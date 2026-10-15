import math
from statistics import median

from .model import Candidate, LapTable


MAX_PROJECTION_LAPS = 10   # never project a catch further out than this
MIN_LAPS_FOR_UNDERCUT = 5


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _top(cands: list[Candidate], n: int = 3) -> list[Candidate]:
    return sorted(cands, key=lambda c: -c.score)[:n]


def _clean(numbers: dict) -> dict:
    return {k: v for k, v in numbers.items() if v is not None}


def pace_candidates(t: LapTable, lap: int) -> list[Candidate]:
    out = []
    order = t.order(lap)
    if len(order) >= 2:
        gap = t.gap_ahead(order[1], lap)
        if gap is not None:
            out.append(Candidate("pace", f"leader:{order[0]}", (order[0], order[1]),
                                 {"kind": "leader", "leader": order[0], "second": order[1],
                                  "gap_s": round(gap, 2), "lap": lap}, 0.05))
    paces = {d: p for d in t.drivers if (p := t.recent_pace(d, lap)) is not None}
    if len(paces) < 2:
        return out
    ranked = sorted(paces, key=paces.get)
    best, second = ranked[0], ranked[1]
    margin = paces[second] - paces[best]
    nums = {"kind": "fastest", "driver": best, "pace_s": round(paces[best], 3), "next": second,
            "margin_s": round(margin, 3), "lap": lap}
    drivers, score = (best,), _clip(margin / 0.8)
    if best in order and order.index(best) > 0:
        ahead = order[order.index(best) - 1]
        gap, closing = t.gap_ahead(best, lap), None
        if ahead in paces:
            closing = paces[ahead] - paces[best]
        drivers = (best, ahead)
        nums["ahead"] = ahead
        if gap is not None and closing is not None and closing > 0.02:
            laps = math.ceil(gap / closing)
            if laps <= MAX_PROJECTION_LAPS and lap + laps <= t.total_laps:
                nums.update(gap_s=round(gap, 2), closing_s_per_lap=round(closing, 3),
                            laps_to_catch=laps, catch_lap=lap + laps)
                score = _clip(score + 0.2)
    elif best in order and len(order) > 1:
        lead = t.gap_ahead(order[1], lap)
        nums["leader"] = True
        if lead is not None:
            nums["lead_s"] = round(lead, 2)
            delta = paces[order[1]] - paces[best] if order[1] in paces else None
            if delta is not None and delta > 0.02 and lap + 5 <= t.total_laps:
                nums["lead_in_5_laps_s"] = round(lead + 5 * delta, 2)
    out.append(Candidate("pace", f"fastest:{best}", drivers, nums, score))
    return out


def _slope(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def tyres_candidates(t: LapTable, lap: int) -> list[Candidate]:
    top = t.order(lap)[:10]
    ages = {d: info[1] for d in top if (info := t.tyre(d, lap)) is not None}
    if len(ages) < 3:
        return []
    med_age = median(ages.values())
    out = []
    for d in top:
        info = t.tyre(d, lap)
        if info is None:
            continue
        compound, age = info
        laps = t.stint_laps(d, lap)
        if len(laps) >= 4:
            slope = _slope([r.tyre_life for r in laps], [r.time_s for r in laps])
            if slope >= 0.05:
                out.append(Candidate("tyres", f"deg:{d}", (d,),
                                     {"kind": "degradation", "driver": d, "compound": compound, "tyre_age": age,
                                      "deg_s_per_lap": round(slope, 3), "stint_laps": len(laps), "lap": lap},
                                     _clip(slope / 0.3)))
        if age - med_age >= 4:
            out.append(Candidate("tyres", f"age:{d}", (d,),
                                 {"kind": "age", "driver": d, "compound": compound, "tyre_age": age,
                                  "median_age": round(med_age, 1), "lap": lap},
                                 _clip((age - med_age) / 12) * 0.8))
    return _top(out)


MAX_BATTLE_GAP = 2.5


def battle_candidates(t: LapTable, lap: int) -> list[Candidate]:
    order = t.order(lap)[:15]
    out = []
    for ahead, behind in zip(order, order[1:]):
        gap = t.gap_ahead(behind, lap)
        if gap is None or gap > MAX_BATTLE_GAP:
            continue
        pa, pb = t.recent_pace(ahead, lap), t.recent_pace(behind, lap)
        closing = (pa - pb) if pa is not None and pb is not None else None
        ta, tb = t.tyre(ahead, lap), t.tyre(behind, lap)
        nums = {"kind": "battle", "ahead": ahead, "behind": behind, "gap_s": round(gap, 2), "lap": lap,
                "ahead_tyre_age": ta[1] if ta else None, "behind_tyre_age": tb[1] if tb else None}
        score = 0.6 * (1 - gap / MAX_BATTLE_GAP)
        if closing is not None:
            nums["closing_s_per_lap"] = round(closing, 3)
            score += 0.4 * _clip(max(closing, 0.0) / 0.5)
            if closing > 0.03:
                laps = math.ceil(gap / closing)
                if laps <= MAX_PROJECTION_LAPS and lap + laps <= t.total_laps:
                    nums["laps_to_close"] = laps
        out.append(Candidate("battle", f"battle:{ahead}-{behind}", (ahead, behind), _clean(nums), score))
    return _top(out)


def pit_candidates(t: LapTable, lap: int) -> list[Candidate]:
    out = []
    for d in t.drivers:
        r = t.rec(d, lap)
        if r is None or not r.pit_in:
            continue
        before, after = t.position(d, lap - 1), t.position(d, lap)
        score = 0.4 + (0.4 if before is not None and before <= 10 else 0.0)
        if before is not None and after is not None:
            score += 0.2 * _clip(abs(after - before) / 4)
        out.append(Candidate("pit", f"stop:{d}:{lap}", (d,),
                             _clean({"kind": "stop", "driver": d, "lap": lap, "old_compound": r.compound,
                                     "tyre_age": r.tyre_life, "pos_before": before, "pos_after": after}), score))
    loss = t.pit_loss(lap)
    if loss is not None and t.total_laps - lap >= MIN_LAPS_FOR_UNDERCUT:
        order = t.order(lap)[:10]
        for ahead, behind in zip(order, order[1:]):
            gap, ta, tb = t.gap_ahead(behind, lap), t.tyre(ahead, lap), t.tyre(behind, lap)
            if gap is None or gap > 3.0 or ta is None or tb is None:
                continue
            if tb[1] < 12 or tb[1] - ta[1] < 3:
                continue
            out.append(Candidate("pit", f"undercut:{ahead}-{behind}", (ahead, behind),
                                 {"kind": "undercut", "ahead": ahead, "behind": behind, "gap_s": round(gap, 2),
                                  "behind_tyre_age": tb[1], "ahead_tyre_age": ta[1],
                                  "pit_loss_s": round(loss, 1), "lap": lap},
                                 0.5 * (1 - gap / 3.0) + 0.5 * _clip(tb[1] / 25)))
    return _top(out)


MOVER_WINDOW = 5


def movers_candidates(t: LapTable, lap: int) -> list[Candidate]:
    w = min(MOVER_WINDOW, lap)
    start = lap - w
    out = []
    for d in t.order(lap):
        p0, p1 = t.position(d, start), t.position(d, lap)
        if p0 is None or p1 is None or abs(p0 - p1) < 2:
            continue
        pitted = any((r := t.rec(d, l)) is not None and r.pit_in for l in range(start + 1, lap + 1))
        out.append(Candidate("movers", f"move:{d}", (d,),
                             {"kind": "gain" if p0 > p1 else "loss", "driver": d, "positions": abs(p0 - p1),
                              "from_pos": p0, "to_pos": p1, "window_laps": w,
                              "cause": "pit_cycle" if pitted else "on_track", "from_grid": start == 0, "lap": lap},
                             _clip(abs(p0 - p1) / 6)))
    return _top(out)


GENERATORS = (pace_candidates, tyres_candidates, battle_candidates, pit_candidates, movers_candidates)


def candidates_for_lap(t: LapTable, lap: int) -> list[Candidate]:
    return [c for gen in GENERATORS for c in gen(t, lap)]
