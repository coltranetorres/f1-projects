import pytest

from harness.feedback import submit_feedback


class FakeClient:
    def __init__(self):
        self.calls = []

    def create_score(self, **kw):
        self.calls.append(kw)


def test_thumbs_down_with_reason():
    c = FakeClient()
    submit_feedback("abc123", 0, "wrong driver mentioned", client=c)
    assert c.calls == [dict(trace_id="abc123", name="user_feedback", value=0,
                            comment="wrong driver mentioned")]


def test_thumbs_up_without_reason_has_no_comment():
    c = FakeClient()
    submit_feedback("abc123", 1, "  ", client=c)
    assert c.calls[0]["comment"] is None and c.calls[0]["value"] == 1


@pytest.mark.parametrize("value", [2, -1, "up"])
def test_invalid_value_rejected(value):
    with pytest.raises(ValueError):
        submit_feedback("abc123", value, client=FakeClient())


def test_missing_trace_id_rejected():
    with pytest.raises(ValueError):
        submit_feedback("", 1, client=FakeClient())
