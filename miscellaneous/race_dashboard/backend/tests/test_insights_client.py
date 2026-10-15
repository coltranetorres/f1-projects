import httpx
import pytest

from race_dashboard.backend.insights import config
from race_dashboard.backend.insights.client import chat_json, post_json


class Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_post_json_returns_body():
    calls = []

    def post(url, json, headers, timeout):
        calls.append((url, json, headers["Authorization"]))
        return Resp({"ok": 1})

    assert post_json("http://x", {"a": 1}, "KEY", post=post) == {"ok": 1}
    assert calls == [("http://x", {"a": 1}, "Bearer KEY")]


def test_post_json_retries_then_succeeds():
    seq = [Resp({}, 500), Resp({"ok": 2})]
    assert post_json("http://x", {}, "K", backoff_s=0, post=lambda **kw: seq.pop(0)) == {"ok": 2}


def test_post_json_raises_after_retries():
    with pytest.raises(RuntimeError):
        post_json("http://x", {}, "K", retries=1, backoff_s=0, post=lambda **kw: Resp({}, 500))


def test_post_json_bad_json_is_an_error():
    with pytest.raises(RuntimeError):
        post_json("http://x", {}, "K", retries=0, backoff_s=0, post=lambda **kw: Resp(ValueError("no")))


def test_chat_json_parses_message_content():
    body = {"choices": [{"message": {"content": '```json\n{"insights": []}\n```'}}]}
    assert chat_json("m", "sys", "usr", "K", post=lambda **kw: Resp(body)) == {"insights": []}


def test_chat_json_rejects_non_json_content():
    body = {"choices": [{"message": {"content": "sorry"}}]}
    with pytest.raises(RuntimeError):
        chat_json("m", "sys", "usr", "K", post=lambda **kw: Resp(body))


def test_config_requires_values(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OPENROUTER_APIKEY", "F1_RANKER_MODEL", "F1_DEFAULT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for fn in (config.openrouter_key, config.ranker_model, config.phrase_model):
        with pytest.raises(RuntimeError):
            fn()
    monkeypatch.setenv("F1_RANKER_MODEL", "typesafe/jev-1.14")
    assert config.ranker_model() == "typesafe/jev-1.14"
