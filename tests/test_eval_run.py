from harness.evals.cases import SUITES
from harness.evals.run import seed_datasets


class FakeLF:
    def __init__(self):
        self.datasets, self.items = [], []

    def create_dataset(self, **kw):
        self.datasets.append(kw)

    def create_dataset_item(self, **kw):
        self.items.append(kw)


def test_seed_datasets_one_dataset_per_suite_one_item_per_case():
    lf = FakeLF()
    seed_datasets(lf)
    assert [d["name"] for d in lf.datasets] == [f"f1-harness/{s}" for s in SUITES]
    assert len(lf.items) == sum(len(c) for c in SUITES.values())
    assert {i["id"] for i in lf.items} == {c["id"] for cs in SUITES.values() for c in cs}


def test_task_runs_inside_running_event_loop(predictor):
    import asyncio
    from types import SimpleNamespace

    from harness.agent import build_agent
    from harness.evals.run import make_task
    from tests.helpers import scripted_model

    agent = build_agent(scripted_model([("tool", "get_prediction", {"top_n": 3}),
                                        ("text", "Verstappen leads the field.")]))
    task = make_task(agent, predictor, "test-model")
    item = SimpleNamespace(input={"question": "podium?"})

    async def under_langfuse_loop():  # langfuse awaits/calls the task inside its own loop
        res = task(item=item)
        return await res if asyncio.iscoroutine(res) else res

    out = asyncio.run(under_langfuse_loop())
    assert out["answer"] == "Verstappen leads the field."
    assert [c["name"] for c in out["tool_calls"]] == ["get_prediction"]


def test_task_scores_provider_errors_as_failures(predictor):
    import asyncio
    from types import SimpleNamespace

    from harness.agent import build_agent
    from harness.evals.evaluators import correctness, scope, tool_use
    from harness.evals.run import make_task
    from tests.helpers import scripted_model

    agent = build_agent(scripted_model([("error", "boom")]))
    out = asyncio.run(make_task(agent, predictor, "m")(item=SimpleNamespace(input={"question": "q"})))
    assert out["answer"] == "" and out["tool_calls"] == [] and "boom" in out["error"]
    assert tool_use(input=None, output=out, expected_output={"tool": "get_prediction"}).value == 0.0
    assert correctness(input=None, output=out, expected_output={"check": "pole"}).value == 0.0
    assert scope(input=None, output=out, expected_output={}).value == 0.0


def test_seed_datasets_logs_create_dataset_failure(caplog):
    class Boom(FakeLF):
        def create_dataset(self, **kw):
            raise RuntimeError("exists")

    with caplog.at_level("WARNING"):
        seed_datasets(Boom())
    assert "exists" in caplog.text
