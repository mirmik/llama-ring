#!/usr/bin/env python3
"""Reproducible GPU + real Nemor integration smoke test; starts an isolated backend."""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ram-rotation", type=int, choices=(0,1), default=0)
    parser.add_argument("--cache-type", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--nemor", required=True, help="Path to the Nemor checkout")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18097)
    parser.add_argument("--active", type=int, default=2048)
    parser.add_argument("--prefix", type=int, default=8192)
    args = parser.parse_args()
    sys.path.insert(0, args.nemor)
    from nemor.agent.client import chat_completion
    from nemor.app.llm_client import get_llm_client
    from nemor.agent.runtime import Agent
    from nemor.core.tool_registry import ToolRegistry
    from nemor.core.session import Session
    from nemor.core.errors import StoppedByUser

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    url = f"http://127.0.0.1:{args.port}"
    log = output.with_suffix(".server.log").open("w")
    command = [
        sys.executable,
        str(root / "scripts/serve.py"),
        "--model",
        args.model,
        "--cache-type",
        args.cache_type,
        "--ram-rotation",
        str(args.ram_rotation),
        "--port",
        str(args.port),
        "--prefix",
        str(args.prefix),
        "--active",
        str(args.active),
        "--max-output",
        "512",
        "--max-summary",
        "512",
        "--max-sessions",
        "2",
    ]
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    results = {"model": args.model, "command": command, "checks": {}, "responses": {}}
    clients = []

    def save():
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    def checked(name, condition, detail=None):
        results["checks"][name] = bool(condition)
        if detail is not None:
            results["responses"][name] = detail
        save()
        print(name, "PASS" if condition else "FAIL", flush=True)
        assert condition, name

    def request(messages, session="basic", **kw):
        body = {
            "model": "vcortex-shadow",
            "messages": messages,
            "session_id": session,
            "temperature": 0,
            "max_tokens": 64,
            **kw,
        }
        r = requests.post(url + "/v1/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return r.json()

    def status():
        r = requests.get(url + "/v1/shadow/sessions", timeout=300)
        r.raise_for_status()
        return r.json()

    def config(sid, stream=True):
        cfg = {
            "server": url,
            "model": "vcortex-shadow",
            "context_size": None,
            "max_tokens": 256,
            "stream": stream,
            "reasoning": {"enabled": False},
            "sampling": {"temperature": 0},
            "_session_id": sid,
            "max_iterations": 4,
            "request_timeout": 300,
            "system_prompt": "Отвечай кратко. Если нужен код проекта, обязательно вызови lookup_code. Не выдумывай результат инструмента.",
        }
        cfg["_llm_client"] = get_llm_client(cfg)
        clients.append(cfg["_llm_client"])
        return cfg

    try:
        deadline = time.monotonic() + 180
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Backend exited: {proc.returncode}; see {log.name}")
            try:
                if requests.get(url + "/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Backend startup")
            time.sleep(0.2)
        msgs = [
            {"role": "system", "content": "Отвечай кратко."},
            {"role": "user", "content": "Сколько будет 2+2?"},
        ]
        first = request(msgs)
        checked("nonstream", "4" in first["choices"][0]["message"]["content"], first)
        msgs += [first["choices"][0]["message"], {"role": "user", "content": "А 3+3?"}]
        second = request(msgs)
        checked(
            "cache_reuse",
            second["cache_reused"]
            and "6" in second["choices"][0]["message"]["content"],
            second,
        )
        changed = request([{"role": "user", "content": "Ответь только словом ДА."}])
        checked("edited_history_reset", not changed["cache_reused"], changed)
        other = request(
            [{"role": "user", "content": "Сколько будет 8+1?"}], session="other"
        )
        checked(
            "session_isolation",
            not other["cache_reused"]
            and "9" in other["choices"][0]["message"]["content"],
            other,
        )
        deltas = []
        cfg = config("nemor-chat")
        answer = chat_completion(
            [{"role": "user", "content": "Напиши одно слово: Привет."}],
            [],
            cfg,
            stream_callback=deltas.append,
        )
        checked(
            "nemor_stream",
            bool(deltas) and "Привет" in answer["choices"][0]["message"]["content"],
            answer,
        )
        calls = []
        registry = ToolRegistry()
        registry.register(
            "lookup_code",
            lambda values, cfg: calls.append(values) or "PROJECT_CODE=ORION-742",
            {
                "type": "function",
                "function": {
                    "name": "lookup_code",
                    "description": "Получить код проекта по имени.",
                    "parameters": {
                        "type": "object",
                        "properties": {"project": {"type": "string"}},
                        "required": ["project"],
                    },
                },
            },
        )
        cfg = config("nemor-agent")
        cfg["tools"] = {"compact_context": False}
        agent = Agent(config=cfg, tools=registry, session=Session("nemor-agent"))
        events = []
        agent.subscribe(events.append)
        asyncio.run(
            agent.prompt(
                "Получи код проекта Orion с помощью lookup_code и напиши полученный код."
            )
        )
        checked(
            "nemor_agent_tool_loop",
            len(calls) == 1
            and "ORION-742" in "".join(e.get("text", "") for e in events),
            {"calls": calls, "events": events},
        )
        state = status()
        checked(
            "tool_result_cache_reuse",
            state["sessions"]["nemor-agent"]["cache_hits"] >= 1,
            state,
        )
        from nemor.app.tools import get_tool_specs

        specs = get_tool_specs({})
        builtin = request(
            [
                {
                    "role": "system",
                    "content": "Отвечай кратко. Не вызывай инструменты для арифметики.",
                },
                {"role": "user", "content": "Сколько будет 2+2?"},
            ],
            session="builtin",
            tools=specs,
        )
        checked(
            "nemor_builtin_schemas",
            "4" in builtin["choices"][0]["message"]["content"],
            {"tool_count": len(specs), "response": builtin},
        )
        tokens = requests.post(
            url + "/tokenize",
            json={"content": "Hello world", "model": "vcortex-shadow"},
            timeout=10,
        )
        checked(
            "nemor_tokenize",
            tokens.ok and bool(tokens.json().get("tokens")),
            tokens.json(),
        )
        # Cancel while prefill is still running; empty SSE deltas must make Nemor's stop_event observable.
        cfg = config("cancel")
        stop = threading.Event()
        timer = threading.Timer(0.6, stop.set)
        timer.start()
        started = time.monotonic()
        canceled = False
        try:
            chat_completion(
                [{"role": "user", "content": "Повторяй слово тест. " * 500}],
                [],
                cfg,
                stop_event=stop,
            )
        except StoppedByUser:
            canceled = True
        finally:
            timer.cancel()
        elapsed = time.monotonic() - started
        state = status()
        checked(
            "cancel",
            canceled and elapsed < 5 and "cancel" not in state["sessions"],
            {"elapsed_s": elapsed, "status": state},
        )
        resumed = request(
            [{"role": "user", "content": "Сколько будет 5+5?"}], session="cancel"
        )
        checked(
            "cancel_recovery",
            "10" in resumed["choices"][0]["message"]["content"]
            and not resumed["cache_reused"],
            resumed,
        )
        generation_stop = threading.Event()
        generation_canceled = False
        streamed = []

        def stop_after_token(text):
            streamed.append(text)
            generation_stop.set()

        try:
            chat_completion(
                [
                    {
                        "role": "user",
                        "content": "Перечисли все числа от 1 до 200, каждое на новой строке.",
                    }
                ],
                [],
                config("generation-cancel"),
                stream_callback=stop_after_token,
                stop_event=generation_stop,
            )
        except StoppedByUser:
            generation_canceled = True
        state = status()
        checked(
            "cancel_generation",
            generation_canceled
            and bool(streamed)
            and "generation-cancel" not in state["sessions"],
            state,
        )
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    request,
                    [{"role": "user", "content": f"Ответь только этим кодом: {code}."}],
                    session=f"parallel-{index}",
                )
                for index, code in enumerate(("ALPHA-317", "BETA-842"))
            ]
            replies = [future.result() for future in futures]
        checked(
            "concurrent_sessions",
            all(
                code in reply["choices"][0]["message"]["content"]
                for code, reply in zip(("ALPHA-317", "BETA-842"), replies)
            ),
            replies,
        )
        # Fill A in manageable turns. Idle long enough for a genuine background summary to finish.
        history = [
            {
                "role": "system",
                "content": "Запоминай факты. На заметки отвечай только OK. На вопрос о коде ответь только кодом.",
            }
        ]
        observed = []
        seen_ready = False
        commits = 0
        for i in range(14):
            note = "Код проекта Vega: VEGA-913. " if i == 0 else ""
            history.append(
                {
                    "role": "user",
                    "content": note
                    + "Фоновая заметка: погода ясная, ветер слабый, облака редкие. "
                    * int(18 * args.active / 2048),
                }
            )
            reply = request(history, session="memory", max_tokens=16)
            history.append(reply["choices"][0]["message"])
            observed.append(reply)
            if (
                reply["shadow"]["shadow_phase"] not in ("empty", "ready")
                and not seen_ready
            ):
                end = time.monotonic() + 120
                while time.monotonic() < end:
                    time.sleep(1)
                    state = status()["sessions"]["memory"]
                    if state["shadow_phase"] == "ready":
                        seen_ready = True
                        break
                    if state["shadow_phase"] == "failed":
                        raise RuntimeError("Background summary failed")
            commits = reply["shadow"]["commits"]
            if commits >= 2:
                break
        history.append(
            {"role": "user", "content": "Какой код проекта Vega? Ответь только кодом."}
        )
        recall = request(history, session="memory", max_tokens=32)
        checked(
            "shadow_rollover_recall",
            seen_ready
            and commits >= 2
            and "VEGA-913" in recall["choices"][0]["message"]["content"],
            {"turns": observed, "recall": recall},
        )
        checked("idle_commit", recall["shadow"]["ready_commits"] >= 1, recall["shadow"])
        state = status()
        checked(
            "bounded_sessions",
            len(state["sessions"]) <= 2 and state["evictions"] > 0,
            state,
        )
        results["completed"] = True
        save()
    except BaseException as exc:
        results["failure"] = str(exc)
        save()
        raise
    finally:
        for client in clients:
            client.close()
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


if __name__ == "__main__":
    main()
