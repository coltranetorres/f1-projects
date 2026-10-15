import json
import re
import time

import httpx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def post_json(url, payload, api_key, *, timeout=60.0, retries=2, backoff_s=0.5, post=None) -> dict:
    post = post or httpx.post
    last = None
    for attempt in range(retries + 1):
        try:
            r = post(url=url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            last = e
            if attempt < retries:
                time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"POST {url} failed: {last}") from last


def chat_json(model, system, user, api_key, *, post=None) -> dict:
    body = post_json(CHAT_URL, {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }, api_key, post=post)
    try:
        text = body["choices"][0]["message"]["content"]
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        return json.loads(text)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError(f"unparseable chat response: {e}") from e
