#!/usr/bin/env python3
"""Exercise a running shadow backend from the Nemor host (safe, synthetic messages)."""

import argparse
import json
from pathlib import Path
import time
import requests

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", default="http://127.0.0.1:8097")
p.add_argument("--output", required=True)
p.add_argument("--repeat", type=int, default=36)
p.add_argument("--max-turns", type=int, default=18)
p.add_argument("--allow-recall-miss", action="store_true", help="Record factual recall separately from mechanical/resource checks")
p.add_argument("--timeout", type=int, default=600)
a = p.parse_args()
result = {"url": a.url, "turns": []}
path = Path(a.output)
path.parent.mkdir(parents=True, exist_ok=True)


def save():
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def chat(history, session="deployment-probe", limit=16):
    start = time.monotonic()
    r = requests.post(
        a.url + "/v1/chat/completions",
        json={
            "model": "vcortex-shadow",
            "messages": history,
            "session_id": session,
            "max_tokens": limit,
            "temperature": 0,
        },
        timeout=a.timeout,
    )
    r.raise_for_status()
    value = r.json()
    value["wall_s"] = time.monotonic() - start
    return value


try:
    h = [
        {
            "role": "system",
            "content": "Запоминай факты. На заметки отвечай только OK. На вопрос о коде ответь только кодом.",
        }
    ]
    waited = False
    for i in range(a.max_turns):
        note = "Код проекта Vega: VEGA-913. " if i == 0 else ""
        h.append(
            {
                "role": "user",
                "content": note
                + "Фоновая заметка: погода ясная, ветер слабый, облака редкие. " * a.repeat,
            }
        )
        response = chat(h)
        h.append(response["choices"][0]["message"])
        result["turns"].append(response)
        save()
        print(i, response["context_state"], flush=True)
        if response["shadow"]["shadow_phase"] not in ("empty", "ready") and not waited:
            deadline = time.monotonic() + a.timeout
            while time.monotonic() < deadline:
                time.sleep(0.5)
                state = requests.get(a.url + "/v1/shadow/sessions", timeout=30).json()[
                    "sessions"
                ]["deployment-probe"]
                if state["shadow_phase"] == "ready":
                    waited = True
                    break
                if state["shadow_phase"] == "failed":
                    raise RuntimeError(state)
        if response["context_state"]["revision"] >= 2:
            break
    h.append(
        {"role": "user", "content": "Какой код проекта Vega? Ответь только кодом."}
    )
    result["recall"] = chat(h, limit=32)
    result["recall_correct"] = "VEGA-913" in result["recall"]["choices"][0]["message"]["content"]
    if not a.allow_recall_miss:
        assert result["recall_correct"]
    assert result["recall"]["context_state"]["revision"] >= 2
    assert result["recall"]["shadow"]["ready_commits"] >= 1
    speed_history = [
        {
            "role": "user",
            "content": "Перечисляй числа от 1 до 1000, каждое на новой строке.",
        }
    ]
    result["speed"] = [
        chat(speed_history, session="speed-probe", limit=128) for _ in range(3)
    ]
    result["completed"] = True
    save()
except BaseException as exc:
    result["failure"] = str(exc)
    save()
    raise
