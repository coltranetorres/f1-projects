"""One-shot manual smoke test: uv run python -m harness.chat "question" [--model ID]"""
import argparse

from .agent import build_agent, build_model
from .config import default_model, get_predictor
from .runner import run_chat_sync
from .tracing import flush, init_tracing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=default_model())
    args = ap.parse_args()
    print("tracing:", "on" if init_tracing() else "OFF (Langfuse env missing)")
    res = run_chat_sync(build_agent(build_model(args.model)), get_predictor(), args.question,
                        model_id=args.model)
    for c in res.tool_calls:
        print(f"[tool] {c['name']}({c['args']})")
    for e in res.tool_errors:
        print(f"[rejected] {e['name']}({e['args']}): {e['error']}")
    print("\n" + res.answer)
    print(f"\nunverified={res.unverified} capped={res.capped} trace_id={res.trace_id}")
    flush()


if __name__ == "__main__":
    main()
