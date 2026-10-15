from .model import Candidate


def render(c: Candidate) -> tuple[str, str]:
    n, kind = c.numbers, c.numbers["kind"]
    if kind == "fastest":
        detail = f"{n['driver']} averages {n['pace_s']}s over the last 3 laps, {n['margin_s']}s clear of {n['next']}."
        if "laps_to_catch" in n:
            detail += f" At this rate {n['driver']} reaches {n['ahead']} in {n['laps_to_catch']} laps."
        elif "lead_in_5_laps_s" in n:
            detail += f" The lead of {n['lead_s']}s would grow to {n['lead_in_5_laps_s']}s in 5 laps."
        return f"{n['driver']} sets the pace", detail
    if kind == "leader":
        return f"{n['leader']} leads", f"{n['leader']} is {n['gap_s']}s ahead of {n['second']} after lap {n['lap']}."
    if kind == "degradation":
        return (f"{n['driver']}'s tyres are fading",
                f"{n['driver']} is losing {n['deg_s_per_lap']}s per lap on {n['compound']} tyres, {n['tyre_age']} laps old.")
    if kind == "age":
        return (f"{n['driver']} on old tyres",
                f"{n['driver']}'s {n['compound']} tyres are {n['tyre_age']} laps old, against a top-10 median of {n['median_age']}.")
    if kind == "battle":
        detail = f"{n['behind']} is {n['gap_s']}s behind {n['ahead']}."
        if "laps_to_close" in n:
            detail += f" At the current rates the gap closes in {n['laps_to_close']} laps."
        elif "closing_s_per_lap" in n and n["closing_s_per_lap"] < 0:
            detail += f" {n['ahead']} is pulling away."
        return f"{n['behind']} chases {n['ahead']}", detail
    if kind == "stop":
        detail = f"{n['driver']} pits on lap {n['lap']}"
        if "old_compound" in n:
            detail += f" from {n['old_compound']} tyres"
        if "tyre_age" in n:
            detail += f", {n['tyre_age']} laps old"
        detail += "."
        if "pos_before" in n and "pos_after" in n:
            detail += f" P{n['pos_before']} to P{n['pos_after']}."
        return f"{n['driver']} pits", detail
    if kind == "undercut":
        return (f"Undercut window for {n['behind']}",
                f"{n['behind']} is {n['gap_s']}s behind {n['ahead']} on {n['behind_tyre_age']}-lap-old tyres "
                f"against {n['ahead_tyre_age']}; a stop costs about {n['pit_loss_s']}s.")
    if kind in ("gain", "loss"):
        verb = "gains" if kind == "gain" else "loses"
        since = "since the start" if n["from_grid"] else f"in {n['window_laps']} laps"
        cause = "the pit cycle" if n["cause"] == "pit_cycle" else "on-track moves"
        return (f"{n['driver']} {verb} {n['positions']} places",
                f"{n['driver']} goes from P{n['from_pos']} to P{n['to_pos']} {since}, mostly through {cause}.")
    raise ValueError(f"no template for kind {kind!r}")
