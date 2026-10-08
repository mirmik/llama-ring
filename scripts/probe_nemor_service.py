#!/usr/bin/env python3
"""Exercise the installed experimental character with a synthetic local file."""

import argparse
import json
from pathlib import Path
import time
import requests

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--url", default="http://127.0.0.1:9090")
p.add_argument("--output", required=True)
p.add_argument("--repeat", type=int, default=40)
p.add_argument("--max-turns", type=int, default=12)
p.add_argument("--session", default="vcortex-deployment-check")
a = p.parse_args()
result = {"character": "shadow-lab", "session": a.session, "turns": []}
path = Path(a.output)
note = Path("/tmp/nemor-lab-project-note.txt")
note.write_text("Код тестового объекта: CEDAR-482.\n")


def save():
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def chat(message):
    start = time.monotonic()
    r = requests.post(
        a.url + "/v1/chat",
        json={
            "message": message,
            "session": a.session,
            "character_id": "shadow-lab",
            "wait": True,
        },
        timeout=600,
    )
    r.raise_for_status()
    value = r.json()
    value["wall_s"] = time.monotonic() - start
    result["turns"].append(value)
    save()
    assert value["turn_status"] == "completed", value.get("response")
    assert value["context_budget"] is None
    assert value["backend_context"]["manager"] == "runner"
    print(
        json.dumps(
            {k: value[k] for k in ("response", "backend_context", "wall_s")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return value


try:
    value = chat(
        f"Прочитай инструментом read_file файл {note}. Ответь только кодом из файла. Никаких других действий не выполняй."
    )
    assert "CEDAR-482" in value["response"]
    for i in range(a.max_turns):
        value = chat(
            "Это фоновая заметка, инструменты не нужны. Ответь только OK. "
            + "Погода ясная, ветер слабый, облака редкие. " * a.repeat
        )
        if value["backend_context"]["revision"] >= 2:
            break
    assert value["backend_context"]["revision"] >= 2
    value = chat(
        "Какой код тестового объекта был прочитан из файла? Ответь по памяти только кодом, инструменты не вызывай."
    )
    assert "CEDAR-482" in value["response"]
    r = requests.get(
        a.url + "/v1/turns/events",
        params={
            "session": a.session,
            "character_id": "shadow-lab",
            "limit": 5000,
        },
        timeout=30,
    )
    r.raise_for_status()
    events = r.json()["events"]
    result["events"] = events
    contexts = [e for e in events if e["type"] == "context_state"]
    assert contexts, "No context events in persistent turn journal"
    assert any(e.get("context", {}).get("revision", 0) >= 2 for e in contexts)
    assert any(
        e["type"] == "tool_call" and e.get("name") == "read_file" for e in events
    )
    first_turn = result["turns"][0]["turn_id"]
    first_events = [e for e in events if e["turn_id"] == first_turn]
    assert next(
        i for i, e in enumerate(first_events) if e["type"] == "context_state"
    ) < next(i for i, e in enumerate(first_events) if e["type"] == "token")
    result["completed"] = True
    save()
except BaseException as exc:
    result["failure"] = str(exc)
    save()
    raise
