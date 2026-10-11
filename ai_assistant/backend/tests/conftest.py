import os

os.environ["F1_DISABLE_TRACING"] = "1"  # tests never send traces to Langfuse (set before .env loads)

import pytest  # noqa: E402

from ai_assistant.backend.f1_core import Predictor  # noqa: E402


@pytest.fixture(scope="session")
def predictor():
    return Predictor()
