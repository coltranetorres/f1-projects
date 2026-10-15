import argparse
import json

from .build import build_all, cache_path, save_insights
from .live import jev_score_fn, llm_phrase_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the per-lap insights cache for the 2026 Singapore GP.")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache file exists")
    ap.add_argument("--max-lap", type=int, help="smoke test: build laps 1..N and print them, writing no cache")
    args = ap.parse_args(argv)
    path = cache_path()
    if path.exists() and not args.force and args.max_lap is None:
        print(f"{path} exists; use --force to rebuild")
        return 0
    from race_dashboard.backend.loader import build_lap_table, load_session  # lazy: pulls in fastf1
    table = build_lap_table(load_session())
    laps = range(1, args.max_lap + 1) if args.max_lap else None
    out = build_all(table, jev_score_fn(), llm_phrase_fn(), laps=laps)
    if args.max_lap:
        print(json.dumps(out, indent=2))
        return 0
    save_insights(path, out)
    heuristic = sum(e["ranked_by"] == "heuristic" for e in out)
    templated = sum(i["source"] == "template" for e in out for i in e["insights"])
    print(f"wrote {path}: {len(out)} laps, {heuristic} ranked by heuristic fallback, {templated} template-phrased")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
