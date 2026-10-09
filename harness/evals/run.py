# harness/evals/run.py
import argparse
import logging
from datetime import datetime, timezone

from langfuse import get_client

from harness.agent import build_agent, build_model
from harness.config import default_model, get_predictor
from harness.runner import run_chat
from harness.tracing import flush, init_tracing

from .cases import SUITES
from .evaluators import EVALUATORS

log = logging.getLogger(__name__)


def seed_datasets(lf) -> None:
    for suite, cases in SUITES.items():
        name = f"f1-harness/{suite}"
        try:
            lf.create_dataset(name=name, description=f"F1 harness seed cases: {suite}")
        except Exception as e:  # dataset may already exist; item upsert below fails loudly if it truly is missing
            log.warning("create_dataset(%s) failed (may already exist): %r", name, e)
        for c in cases:
            lf.create_dataset_item(dataset_name=name, id=c["id"], input={"question": c["question"]},
                                   expected_output=c["expected"], metadata={"suite": suite, "split": c["split"]})


def make_task(agent, predictor, model_id, session_id=None):
    async def task(*, item, **kwargs):  # langfuse awaits coroutines inside its running loop
        try:
            res = await run_chat(agent, predictor, item.input["question"], session_id=session_id,
                                 model_id=model_id)
        except Exception as e:  # score provider/model errors as failures instead of dropping the item
            return {"answer": "", "tool_calls": [], "tool_errors": [], "unverified": [],
                    "capped": False, "trace_id": None, "error": repr(e)}
        return res.to_output()
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--suite", choices=list(SUITES) + ["all"], default="all")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not init_tracing():
        raise SystemExit("Langfuse env vars missing; evals need tracing on.")
    lf = get_client()
    seed_datasets(lf)

    predictor = get_predictor()
    agent = build_agent(build_model(args.model))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suites = list(SUITES) if args.suite == "all" else [args.suite]

    run_name = f"{args.model}-{stamp}"
    task = make_task(agent, predictor, args.model, session_id=f"eval-{run_name}")

    for suite in suites:
        items = lf.get_dataset(f"f1-harness/{suite}").items
        if args.split != "all":
            items = [i for i in items if (i.metadata or {}).get("split") == args.split]
        if args.limit is not None:
            items = items[: args.limit]
        if not items:
            print(f"[{suite}] no items for split={args.split}")
            continue
        result = lf.run_experiment(
            name=suite, run_name=run_name, data=items, task=task,
            evaluators=EVALUATORS[suite], max_concurrency=3, metadata={"model": args.model, "suite": suite})
        print(result.format())
    flush()


if __name__ == "__main__":
    main()
