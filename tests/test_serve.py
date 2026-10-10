"""Transport failures must not look like successful model completions."""

import asyncio
import importlib.util
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
import pytest

SPEC = importlib.util.spec_from_file_location(
    "shadow_serve", Path(__file__).parents[1] / "scripts/serve.py"
)
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


class FakeWorker:
    failure = None

    def __init__(self, events):
        self.events = events
        self.releases = []
        self.submissions = []
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True

    async def submit(self, data):
        self.submissions.append(data)
        queue = asyncio.Queue()
        for event in self.events:
            queue.put_nowait(event)
        return "test-id", queue

    async def release(self, request_id, cancel):
        self.releases.append((request_id, cancel))


def body(**extra):
    return {
        "model": "vcortex-shadow",
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


def completed():
    return {
        "event": "done",
        "message": {"role": "assistant", "content": "Привет"},
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "shadow": {"commits": 1},
        "cache_reused": True,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"chat_template_kwargs": {"enable_thinking": 1}},
        {"max_tokens": 2049},
        {"max_tokens": True},
        {"temperature": float("nan")},
        {"stop": [1]},
        {"tools": [None]},
        {"tools": [{"type": "function", "function": None}]},
        {"stream": 1},
        {"session_id": ""},
        {"reasoning_budget": -2},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ],
)
def test_validation(extra):
    with pytest.raises((ValueError, AttributeError)):
        serve.validate(body(**extra), "vcortex-shadow", 2048)


def test_stream_and_nonstream():
    async def scenario():
        worker = FakeWorker(
            [{"event": "delta", "delta": {"content": "Привет"}}, completed()]
        )
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            async with client.post("/v1/chat/completions", json=body()) as response:
                data = await response.json()
                assert response.status == 200
                assert data["choices"][0]["message"]["content"] == "Привет"
                assert data["cache_reused"]
            async with client.post(
                "/v1/chat/completions",
                json=body(stream=True, stream_options={"include_usage": True}),
            ) as response:
                text = await response.text()
                events = [
                    json.loads(line[6:])
                    for line in text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                assert events[0]["choices"][0]["delta"]["content"] == "Привет"
                assert events[1]["choices"][0]["finish_reason"] == "stop"
                assert events[2]["usage"]["total_tokens"] == 5
                assert text.endswith("data: [DONE]\n\n")
        assert worker.closed
        assert worker.releases == [("test-id", False), ("test-id", False)]

    asyncio.run(scenario())


def test_error_after_stream_has_no_success_finish():
    async def scenario():
        worker = FakeWorker(
            [
                {"event": "delta", "delta": {"content": "partial"}},
                {"event": "error", "error": "GPU failed"},
            ]
        )
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            async with client.post(
                "/v1/chat/completions", json=body(stream=True)
            ) as response:
                text = await response.text()
                assert "GPU failed" in text and "[DONE]" not in text
                assert '"finish_reason": "stop"' not in text
        assert worker.releases == [("test-id", True)]

    asyncio.run(scenario())


def test_disconnect_cancels_prefill():
    async def scenario():
        worker = FakeWorker([])
        async with TestClient(
            TestServer(serve.make_app(worker), handler_cancellation=True)
        ) as client:
            response = await client.post("/v1/chat/completions", json=body(stream=True))
            assert b"data:" in await response.content.readline()
            response.close()
            for _ in range(30):
                if worker.releases:
                    break
                await asyncio.sleep(0.05)
            assert worker.releases == [("test-id", True)]

    asyncio.run(scenario())


def test_timeout_releases_request():
    async def scenario():
        worker = FakeWorker([])
        async with TestClient(
            TestServer(serve.make_app(worker, timeout=0.1))
        ) as client:
            async with client.post("/v1/chat/completions", json=body()) as response:
                assert response.status == 504
        assert worker.releases == [("test-id", True)]

    asyncio.run(scenario())


def test_invalid_request_never_reaches_worker():
    async def scenario():
        worker = FakeWorker([])
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            async with client.post(
                "/v1/chat/completions", json=body(max_tokens=-1)
            ) as response:
                assert response.status == 400
        assert not worker.submissions

    asyncio.run(scenario())


def test_worker_crash_unblocks_pending_requests():
    async def scenario():
        import sys

        worker = serve.Worker(
            [
                sys.executable,
                "-u",
                "-c",
                'import json,sys; print(json.dumps({"event":"ready"}),flush=True); sys.stdin.readline(); sys.exit(7)',
            ]
        )
        await worker.start()
        request_id, queue = await worker.submit({"op": "chat"})
        event = await asyncio.wait_for(queue.get(), 3)
        assert event["event"] == "error" and "(7)" in event["error"]
        await worker.release(request_id, True)
        await worker.close()
        assert not worker.pending

    asyncio.run(scenario())


def test_worker_queue_limit():
    async def scenario():
        import sys
        from aiohttp import web

        worker = serve.Worker(
            [
                sys.executable,
                "-u",
                "-c",
                'import json,sys; print(json.dumps({"event":"ready"}),flush=True); list(sys.stdin)',
            ],
            max_pending=1,
        )
        await worker.start()
        request_id, _ = await worker.submit({"op": "chat"})
        try:
            with pytest.raises(web.HTTPServiceUnavailable):
                await worker.submit({"op": "chat"})
        finally:
            await worker.release(request_id, True)
            await worker.close()

    asyncio.run(scenario())


def test_context_event_precedes_content_and_survives_nonstream():
    async def scenario():
        context = {"schema_version": 1, "context_id": "new", "revision": 2}
        done = {**completed(), "context_state": context}
        worker = FakeWorker(
            [
                {"event": "context", "context_state": context},
                {"event": "delta", "delta": {"content": "Привет"}},
                done,
            ]
        )
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            async with client.post(
                "/v1/chat/completions", json=body(stream=True)
            ) as response:
                data = [
                    json.loads(line[6:])
                    for line in (await response.text()).splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                assert data[0]["choices"] == [] and data[0]["context_state"] == context
                assert data[1]["choices"][0]["delta"]["content"] == "Привет"
            async with client.post("/v1/chat/completions", json=body()) as response:
                assert (await response.json())["context_state"] == context

    asyncio.run(scenario())


def test_lan_allowlist_uses_peer_not_forwarded_header():
    async def scenario():
        worker = FakeWorker([])
        async with TestClient(
            TestServer(serve.make_app(worker, allowed_clients={"192.0.2.76"}))
        ) as client:
            async with client.get(
                "/health", headers={"X-Forwarded-For": "192.0.2.76"}
            ) as response:
                assert response.status == 403
        assert not worker.submissions

    asyncio.run(scenario())


def test_thinking_and_long_output_contract():
    request = body(max_tokens=16000, chat_template_kwargs={"enable_thinking": True},
                   reasoning_budget=-1, id_slot=0)
    request["messages"].append({"role": "assistant", "content": "answer", "reasoning_content": "thought"})
    assert serve.validate(request, "vcortex-shadow", 16000) is request
    for extra in ({"id_slot": True}, {"id_slot": -1}, {"id_slot": 16},
                  {"id_slot": 0, "session_id": "conflict"},
                  {"reasoning_budget": 0, "reasoning_budget_tokens": 32}):
        with pytest.raises(ValueError):
            serve.validate(body(**extra), "vcortex-shadow", 16000)


def test_reasoning_stream_is_separate_from_content():
    async def scenario():
        done = completed()
        done["message"]["reasoning_content"] = "Let me think"
        worker = FakeWorker([
            {"event": "delta", "delta": {"reasoning_content": "Let me think"}},
            {"event": "delta", "delta": {"content": "Привет"}}, done])
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            async with client.post("/v1/chat/completions", json=body(stream=True, chat_template_kwargs={"enable_thinking": True})) as r:
                chunks = [json.loads(l[6:]) for l in (await r.text()).splitlines() if l.startswith("data: {")]
                assert chunks[0]["choices"][0]["delta"] == {"reasoning_content": "Let me think"}
                assert chunks[1]["choices"][0]["delta"] == {"content": "Привет"}
            async with client.post("/v1/chat/completions", json=body()) as r:
                assert (await r.json())["choices"][0]["message"]["reasoning_content"] == "Let me think"
    asyncio.run(scenario())


def test_slot_protocol_and_filename_boundaries():
    async def scenario():
        worker = FakeWorker([{"event": "done", "result": {"n_saved": 15, "n_written": 100}}])
        async with TestClient(TestServer(serve.make_app(worker))) as client:
            for action, op in (("save", "save"), ("restore", "restore"), ("erase", "reset")):
                async with client.post(f"/slots/0?action={action}", json={"filename": "session_abc.bin"}) as r:
                    assert r.status == 200
                    assert worker.submissions[-1] == {"op": op, "id_slot": 0, "filename": "session_abc.bin"}
            for filename in ("../escape", "/tmp/escape", ".", "a/b", "a\\b", "", "x" * 161):
                async with client.post("/slots/0?action=save", json={"filename": filename}) as r:
                    assert r.status == 400
            worker.events = [{"event": "error", "error": "Checkpoint not found"}]
            async with client.post("/slots/0?action=restore", json={"filename": "absent.bin"}) as r:
                assert r.status == 400
        assert all(not cancel for _, cancel in worker.releases)
    asyncio.run(scenario())


def image_body(url=None):
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG")
    url = url or "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return body(messages=[{"role": "user", "content": [
        {"type": "text", "text": "What color?"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}])


@pytest.mark.parametrize("url", [
    "https:///image.png", "file:///tmp/image.png",
    "data:image/png;base64,not base64!", "data:image/png;base64,SGVsbG8=",
    "data:image/svg+xml;base64,PHN2Zz4=",
])
def test_invalid_image_input(url):
    with pytest.raises(ValueError):
        serve.validate(image_body(url), "vcortex-shadow", 2048, vision=True)


def test_image_capability_and_history_protocol():
    async def scenario():
        payload = image_body()
        for enabled in (False, True):
            worker = FakeWorker([completed()])
            async with TestClient(TestServer(serve.make_app(worker, vision=enabled))) as client:
                async with client.get("/v1/capabilities") as response:
                    assert (await response.json())["vision"] is enabled
                async with client.post("/v1/chat/completions", json=payload) as response:
                    assert response.status == (200 if enabled else 400)
                if enabled:
                    assert worker.submissions[0]["messages"] == payload["messages"]
                    # A saved user image can be replayed on the next text turn.
                    followup = image_body()
                    followup["messages"] += [{"role": "assistant", "content": "Red"},
                                             {"role": "user", "content": "Describe its shape."}]
                    async with client.post("/v1/chat/completions", json=followup) as response:
                        assert response.status == 200
                else:
                    assert not worker.submissions
    asyncio.run(scenario())


def test_image_role_and_reserved_marker():
    request = image_body()
    request["messages"][0]["role"] = "system"
    with pytest.raises(ValueError, match="user messages"):
        serve.validate(request, "vcortex-shadow", 2048, vision=True)
    with pytest.raises(ValueError, match="Reserved"):
        serve.validate(body(messages=[{"role": "user", "content": "<__media__>"}]), "vcortex-shadow", 2048, vision=True)


def test_extended_chat_validation():
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    for choice in ("required", {"type": "function", "function": {"name": "lookup"}}):
        value = body(tools=[tool], tool_choice=choice, parallel_tool_calls=True,
                     stop=["END", "Конец"], reasoning_budget_tokens=32,
                     response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}})
        assert serve.validate(value, "vcortex-shadow", 2048) is value
    for extra in ({"tool_choice": "required"},
                  {"tools": [tool], "tool_choice": {"type": "function", "function": {"name": "missing"}}},
                  {"parallel_tool_calls": 1}, {"stop": False}, {"stop": [""]},
                  {"reasoning_budget_tokens": 2**31},
                  {"reasoning_budget": 32, "thinking_budget_tokens": 64},
                  {"response_format": {"type": "json_object", "schema": []}},
                  {"response_format": {"type": "json_schema", "json_schema": {"schema": []}}}):
        with pytest.raises(ValueError):
            serve.validate(body(**extra), "vcortex-shadow", 2048)


def test_remote_image_is_resolved_before_worker():
    async def scenario():
        import base64
        from aiohttp import web
        from PIL import Image
        import io
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(buffer, format='PNG')
        source = web.Application()
        async def picture(request):
            return web.Response(body=buffer.getvalue(), content_type='image/png')
        source.router.add_get('/picture', picture)
        worker = FakeWorker([completed()])
        async with TestClient(TestServer(source)) as origin, TestClient(TestServer(serve.make_app(worker, vision=True))) as client:
            payload = body(messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(origin.make_url('/picture'))}}]}])
            async with client.post('/v1/chat/completions', json=payload) as response:
                assert response.status == 200, await response.text()
            url = worker.submissions[0]['messages'][0]['content'][0]['image_url']['url']
            assert base64.b64decode(url.split(',', 1)[1]) == buffer.getvalue()
            payload['messages'][0]['content'][0]['image_url']['url'] = str(origin.make_url('/missing'))
            async with client.post('/v1/chat/completions', json=payload) as response:
                assert response.status == 400
            assert len(worker.submissions) == 1
    asyncio.run(scenario())


def test_video_and_image_format_contract():
    import base64
    import io
    from PIL import Image
    for fmt in ('PNG', 'JPEG', 'WEBP', 'BMP', 'GIF'):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'blue').save(buffer, format=fmt)
        payload = image_body(base64.b64encode(buffer.getvalue()).decode())
        serve.validate(payload, 'vcortex-shadow', 2048, vision=True)
        part = next(p for p in payload['messages'][0]['content'] if p['type'] == 'image_url')
        assert part['image_url']['url'].startswith('data:image/')
    for media in ({'data': 'Ynl0ZXM='}, {'url': 'data:video/mp4;base64,Ynl0ZXM='}, {'url': 'https://example.org/video.mp4'}):
        request = body(messages=[{'role': 'user', 'content': [{'type': 'input_video', 'input_video': media}]}])
        serve.validate(request, 'vcortex-shadow', 2048, vision=True)
        assert request['messages'][0]['content'][0]['input_video']['data']
    for media in ({}, {'data': ''}, {'data': '!'}, {'url': 'file:///tmp/video.mp4'}):
        with pytest.raises(ValueError):
            serve.validate(body(messages=[{'role': 'user', 'content': [{'type': 'input_video', 'input_video': media}]}]), 'vcortex-shadow', 2048, vision=True)
