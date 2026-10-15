from dataclasses import replace

from race_dashboard.backend.insights.model import Candidate, LapRecord, LapTable


def series(driver, n, base, step=0.0, pos=1, offset=0.0, compound="MEDIUM", life0=1, stint=1, positions=None):
    """n consecutive clean laps; lap i takes base + step*i seconds; end_s accumulates from `offset`."""
    out, end = [], offset
    for i in range(n):
        t = base + step * i
        end += t
        out.append(LapRecord(driver, i + 1, t, end, positions[i] if positions else pos,
                             compound, life0 + i, stint, False, False, True))
    return out


def patch(records, lap, **kw):
    return [replace(r, **kw) if r.lap == lap else r for r in records]


def table(*groups, total_laps=10, grid=None):
    return LapTable([r for g in groups for r in g], total_laps, grid or {})


def cand(type_, key, score, numbers=None):
    return Candidate(type_, key, (), {"kind": type_, **(numbers or {})}, score)
