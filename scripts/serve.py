#!/usr/bin/env python3
"""Local OpenAI-compatible transport for the single-owner shadow GPU worker."""

import argparse
import base64
import binascii
import io
import warnings
import asyncio
import contextlib
import json
import logging
import ipaddress
import math
from pathlib import Path
import time
import uuid

from aiohttp import web
from PIL import Image

LOG = logging.getLogger("shadow")


def validate_image(part):
    image = part.get("image_url")
    if not isinstance(image, dict) or not isinstance(image.get("url"), str):
        raise ValueError("image_url must contain a data URL")
    if image.get("detail", "auto") not in {"auto", "low", "high"}:
        raise ValueError("Invalid image detail")
    url = image["url"]
    header, separator, payload = url.partition(",")
    if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64"}:
        raise ValueError("Only inline PNG/JPEG data URLs are supported; remote URLs are not fetched")
    if len(payload) > 12 * 1024 * 1024:
        raise ValueError("Image exceeds 8 MiB")
    try:
        data = base64.b64decode(payload, validate=True)
        if not data or len(data) > 8 * 1024 * 1024:
            raise ValueError("Image exceeds 8 MiB or is empty")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.format not in {"PNG", "JPEG"} or decoded.width * decoded.height > 16_000_000:
                    raise ValueError("Expected PNG/JPEG with at most 16 million pixels")
                decoded.verify()
    except (OSError, SyntaxError, binascii.Error, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or oversized image") from exc


def validate(body, model, max_output, vision=False):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    allowed = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "session_id",
        "id_slot",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
        "reasoning_budget",
        "n",
    }
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"Unsupported request fields: {', '.join(sorted(unknown))}")
    if body.get("model", model) != model:
        raise ValueError(f"Use model {model!r}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    image_count = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("Invalid message role")
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Invalid content part")
                if part.get("type") == "image_url":
                    if not vision:
                        raise ValueError("Images are unsupported: configure --mmproj")
                    if msg["role"] != "user":
                        raise ValueError("Images are supported in user messages only")
                    image_count += 1
                    if image_count > 32:
                        raise ValueError("At most 32 images per request")
                    validate_image(part)
                elif part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise ValueError("Only text and image_url content parts are supported")
                elif "<__media__>" in part["text"]:
                    raise ValueError("Reserved media marker in text")
        elif content is not None and not isinstance(content, str):
            raise ValueError("Expected text or a content array")
        elif isinstance(content, str) and "<__media__>" in content:
            raise ValueError("Reserved media marker in text")
        if msg.get("images"):
            raise ValueError("Images are unsupported")
        if msg.get("reasoning_content") is not None and not isinstance(msg["reasoning_content"], str):
            raise ValueError("reasoning_content must be a string")
    if body.get("n", 1) != 1 or body.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("Only n=1 and tool_choice=auto/none are supported")
    template = body.get("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"} or ("enable_thinking" in template and type(template["enable_thinking"]) is not bool):
        raise ValueError("Only boolean chat_template_kwargs.enable_thinking is supported")
    if type(body.get("reasoning_budget", -1)) is not int or body.get("reasoning_budget", -1) not in (-1, 0):
        raise ValueError("Only unlimited (-1) or disabled (0) reasoning budget is supported")
    if body.get("reasoning_budget") == 0 and template.get("enable_thinking", False):
        raise ValueError("reasoning_budget=0 requires enable_thinking=false")
    if "id_slot" in body:
        if type(body["id_slot"]) is not int or not 0 <= body["id_slot"] < 16:
            raise ValueError("id_slot must be an integer in [0, 15]")
        if "session_id" in body:
            raise ValueError("Use id_slot or session_id, not both")
    if "session_id" in body and (
        not isinstance(body["session_id"], str)
        or not 1 <= len(body["session_id"]) <= 256
    ):
        raise ValueError("session_id must contain 1..256 characters")
    for key, low, high, integer in (
        ("max_tokens", 1, max_output, True),
        ("temperature", 0, 5, False),
        ("top_p", 0, 1, False),
        ("min_p", 0, 1, False),
        ("top_k", 0, 100000, True),
        ("repeat_penalty", 0.01, 10, False),
        ("frequency_penalty", -2, 2, False),
        ("presence_penalty", -2, 2, False),
        ("seed", 0, 2**32 - 1, True),
    ):
        if key in body:
            value = body[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
                or (integer and not isinstance(value, int))
            ):
                raise ValueError(
                    f"Invalid {key}; expected {'integer' if integer else 'number'} in [{low}, {high}]"
                )
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ValueError("stream must be boolean")
    if body.get("stream_options", {}) not in (
        {},
        {"include_usage": True},
        {"include_usage": False},
    ):
        raise ValueError("Unsupported stream_options")
    if "tools" in body:
        if not isinstance(body["tools"], list):
            raise ValueError("tools must be an array")
        for tool in body["tools"]:
            if (
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or not isinstance(tool.get("function"), dict)
                or not isinstance(tool.get("function", {}).get("name"), str)
            ):
                raise ValueError("Invalid function tool")
    return body


class Worker:
    def __init__(self, command, max_pending=8):
        self.command = command
        self.max_pending = max_pending
        self.pending = {}
        self.proc = None
        self.reader = None
        self.ready = None
        self.failure = None
        self.write_lock = asyncio.Lock()

    async def start(self):
        self.ready = asyncio.get_running_loop().create_future()
        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        self.reader = asyncio.create_task(self.read())
        await asyncio.wait_for(asyncio.shield(self.ready), 300)

    async def read(self):
        try:
            while line := await self.proc.stdout.readline():
                event = json.loads(line)
                if event.get("event") == "ready":
                    self.ready.set_result(event)
                    continue
                target = self.pending.get(event.get("id"))
                if target is not None:
                    # Bounded by --max-output and max_pending; do not block GPU dispatch on a slow peer.
                    target.put_nowait(event)
            raise RuntimeError(f"Worker exited ({await self.proc.wait()})")
        except Exception as exc:
            self.failure = str(exc)
            if not self.ready.done():
                self.ready.set_exception(exc)
            for target in self.pending.values():
                target.put_nowait({"event": "error", "error": self.failure})

    async def send(self, data):
        if self.failure or self.proc.returncode is not None:
            raise RuntimeError(self.failure or "Worker stopped")
        async with self.write_lock:
            self.proc.stdin.write(
                (json.dumps(data, ensure_ascii=False) + "\n").encode()
            )
            await self.proc.stdin.drain()

    async def submit(self, data):
        if len(self.pending) >= self.max_pending:
            raise web.HTTPServiceUnavailable(text="Foreground queue is full")
        request_id = uuid.uuid4().hex
        target = asyncio.Queue()
        self.pending[request_id] = target
        try:
            await self.send({**data, "id": request_id})
        except Exception:
            self.pending.pop(request_id, None)
            raise
        return request_id, target

    async def release(self, request_id, cancel):
        if cancel:
            with contextlib.suppress(Exception):
                await self.send({"op": "cancel", "id": request_id})
        self.pending.pop(request_id, None)

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        if self.reader:
            await self.reader


def error_response(message, status=400):
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status == 400 else "server_error",
            }
        },
        status=status,
    )


def make_app(
    worker, model="vcortex-shadow", max_output=2048, timeout=600, allowed_clients=None, vision=False
):
    @web.middleware
    async def peer_access(request, handler):
        if allowed_clients is not None and request.remote not in allowed_clients:
            return error_response("Client address is not allowed", 403)
        return await handler(request)

    app = web.Application(client_max_size=(32 if vision else 8) * 1024 * 1024, middlewares=[peer_access])

    async def lifetime(app):
        try:
            await worker.start()
            yield
        finally:
            await worker.close()

    app.cleanup_ctx.append(lifetime)

    async def health(request):
        return web.json_response(
            {
                "status": "error" if worker.failure else "ok",
                "backend": "vcortex-shadow",
            },
            status=503 if worker.failure else 200,
        )

    async def models(request):
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "owned_by": "local"}],
            }
        )

    async def capabilities(request):
        return web.json_response(
            {
                "protocol": 0,
                "services": {"llm": True},
                "backend": "vcortex-shadow",
                "chat": True,
                "streaming": True,
                "tools": True,
                "vision": vision,
                "reasoning": True,
                "session_id": True,
                "shadow_memory": True,
            }
        )

    async def next_event(request, target):
        while True:
            if request.transport is None or request.transport.is_closing():
                raise ConnectionResetError("Client disconnected")
            try:
                return await asyncio.wait_for(target.get(), 0.5)
            except asyncio.TimeoutError:
                yield_heartbeat = {"event": "heartbeat"}
                return yield_heartbeat

    async def chat(request):
        try:
            body = validate(await request.json(), model, max_output, vision)
        except (ValueError, TypeError, AttributeError) as exc:
            return error_response(str(exc))
        try:
            request_id, target = await worker.submit({**body, "op": "chat"})
        except RuntimeError as exc:
            return error_response(str(exc), 503)
        stream = body.get("stream", False)
        response = None
        complete = False
        created = int(time.time())
        envelope = {
            "id": "chatcmpl-" + request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
        deadline = time.monotonic() + timeout

        async def sse(value):
            await asyncio.wait_for(
                response.write(
                    ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()
                ),
                10,
            )

        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("Request deadline exceeded")
                event = await next_event(request, target)
                if event["event"] == "error":
                    if response is None:
                        return error_response(event["error"], 500)
                    await sse(
                        {"error": {"message": event["error"], "type": "server_error"}}
                    )
                    # No success finish_reason: Nemor correctly rejects an interrupted stream.
                    await response.write_eof()
                    return response
                if stream and response is None:
                    response = web.StreamResponse(
                        headers={
                            "Content-Type": "text/event-stream; charset=utf-8",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        }
                    )
                    await response.prepare(request)
                if event["event"] == "done":
                    complete = True
                    if not stream:
                        return web.json_response(
                            {
                                **envelope,
                                "object": "chat.completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": event["message"],
                                        "finish_reason": event["finish_reason"],
                                    }
                                ],
                                "usage": event["usage"],
                                "shadow": event["shadow"],
                                "mtp": event.get("mtp", {"enabled": False}),
                                "context_state": event.get("context_state"),
                                "cache_reused": event["cache_reused"],
                            }
                        )
                    await sse(
                        {
                            **envelope,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": event["finish_reason"],
                                }
                            ],
                            "shadow": event["shadow"],
                            "mtp": event.get("mtp", {"enabled": False}),
                            "context_state": event.get("context_state"),
                            "cache_reused": event["cache_reused"],
                        }
                    )
                    if body.get("stream_options", {}).get("include_usage"):
                        await sse({**envelope, "choices": [], "usage": event["usage"]})
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                    return response
                if stream and event["event"] == "context":
                    await sse(
                        {
                            **envelope,
                            "choices": [],
                            "context_state": event["context_state"],
                        }
                    )
                elif stream:
                    # Empty deltas let Nemor observe stop_event during queueing/prefill/forced summary.
                    await sse(
                        {
                            **envelope,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": event.get("delta", {}),
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except (TimeoutError, RuntimeError) as exc:
            if response is None:
                return error_response(str(exc), 504)
            with contextlib.suppress(ConnectionError):
                await sse({"error": {"message": str(exc), "type": "server_error"}})
                await response.write_eof()
            return response
        finally:
            await worker.release(request_id, cancel=not complete)

    async def status(request):
        request_id, target = await worker.submit({"op": "status"})
        complete = False
        try:
            event = await asyncio.wait_for(target.get(), timeout)
            complete = True
            return web.json_response(event.get("result", event))
        finally:
            await worker.release(request_id, not complete)

    async def reset(request):
        request_id, target = await worker.submit(
            {"op": "reset", "session_id": request.match_info["session"]}
        )
        complete = False
        try:
            event = await asyncio.wait_for(target.get(), timeout)
            complete = True
            return web.json_response(event.get("result", event))
        finally:
            await worker.release(request_id, not complete)

    async def inspect_prompt(request):
        try:
            body = validate(await request.json(), model, max_output, vision)
        except (ValueError, TypeError, AttributeError) as exc:
            return error_response(str(exc))
        request_id, target = await worker.submit({**body, "op": "inspect"})
        complete = False
        try:
            event = await asyncio.wait_for(target.get(), timeout)
            complete = True
            if event["event"] == "error":
                return error_response(event["error"], 400)
            return web.json_response(event["result"])
        finally:
            await worker.release(request_id, not complete)

    async def slot_action(request):
        try:
            slot = int(request.match_info["slot"])
            action = request.query.get("action")
            if not 0 <= slot < 16 or action not in {"save", "restore", "erase"}:
                raise ValueError("Invalid slot/action")
            body = await request.json() if request.can_read_body else {}
            if not isinstance(body, dict):
                raise ValueError("Expected an object")
            filename = body.get("filename", "")
            if action != "erase" and (not isinstance(filename, str) or not filename or len(filename) > 160 or filename.startswith(".") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in filename)):
                raise ValueError("Invalid checkpoint filename")
        except (ValueError, TypeError) as exc:
            return error_response(str(exc))
        request_id, target = await worker.submit({"op": "reset" if action == "erase" else action, "id_slot": slot, "filename": filename})
        complete = False
        try:
            event = await asyncio.wait_for(target.get(), timeout)
            complete = True
            if event["event"] == "error":
                return error_response(event["error"], 400)
            return web.json_response(event["result"])
        finally:
            await worker.release(request_id, not complete)

    async def tokenize(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("content"), str):
                raise ValueError("content must be a string")
            if body.get("model", model) != model:
                raise ValueError("Unknown model")
            if body.get("add_special", False):
                raise ValueError("add_special=true is unsupported")
        except (ValueError, TypeError) as exc:
            return error_response(str(exc))
        request_id, target = await worker.submit(
            {
                "op": "tokenize",
                "content": body["content"],
                "parse_special": bool(body.get("parse_special", False)),
            }
        )
        complete = False
        try:
            event = await asyncio.wait_for(target.get(), timeout)
            complete = True
            if event["event"] == "error":
                return error_response(event["error"], 500)
            return web.json_response(event["result"])
        finally:
            await worker.release(request_id, not complete)

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/slots/{slot}", slot_action)
    app.router.add_post("/v1/shadow/inspect", inspect_prompt)
    app.router.add_get("/v1/shadow/sessions", status)
    app.router.add_delete("/v1/shadow/sessions/{session}", reset)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", help="Compatible qwen35 vision projector GGUF")
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--mmproj-gpu", type=int, choices=(0, 1), default=1, help="Use 0 to encode images on CPU and keep GPU memory for the language model")
    parser.add_argument("--model-name", default="vcortex-shadow")
    parser.add_argument("--slot-save-path", default="", help="Private directory for full shadow session checkpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allow-client",
        action="append",
        default=[],
        help="Allowed peer IP; required for LAN binding. Loopback is always allowed.",
    )
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--context", type=int, help="Shared total token budget; --prefix bounds prepared prefix scratch")
    parser.add_argument(
        "--worker",
        default=str(
            Path(__file__).resolve().parents[1] / "build/bin/llama-shadow-worker"
        ),
    )
    for name, value in [
        ("prefix", 8192),
        ("active", 8192),
        ("max-output", 2048),
        ("max-sessions", 2),
        ("quantum", 64),
        ("max-summary", 512),
        ("gpu-layers", 99),
    ]:
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--cache-type", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--shared-context", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ram-rotation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--mtp-draft", type=int, choices=range(9), default=0)
    parser.add_argument("--trigger", type=float, default=0.7)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.context is not None:
        if args.context <= args.prefix:
            parser.error("--context must exceed the maximum prepared prefix")
        args.active = args.context - args.prefix
        args.shared_context = 1
    try:
        host = ipaddress.ip_address(args.host)
        allowed = {str(ipaddress.ip_address(address)) for address in args.allow_client}
    except ValueError as exc:
        parser.error(str(exc))
    if not host.is_loopback and not allowed:
        parser.error("LAN binding requires at least one --allow-client IP")
    allowed.update({"127.0.0.1", "::1"})
    if not 1 <= args.max_pending <= 32 or args.timeout <= 0:
        parser.error("max-pending must be 1..32 and timeout must be positive")
    command = [args.worker, "--model", args.model]
    for name in (
        "prefix",
        "active",
        "max-output",
        "max-sessions",
        "quantum",
        "max-summary",
        "gpu-layers",
        "trigger",
        "cache-type",
        "ram-rotation",
        "mtp-draft",
        "shared-context",
    ):
        command += ["--" + name, str(getattr(args, name.replace("-", "_")))]
    if args.mmproj:
        command += ["--mmproj", args.mmproj, "--image-max-tokens", str(args.image_max_tokens), "--mmproj-gpu", str(args.mmproj_gpu)]
    if args.slot_save_path:
        command += ["--slot-save-path", args.slot_save_path]
    logging.basicConfig(level=logging.INFO)
    web.run_app(
        make_app(
            Worker(command, args.max_pending),
            args.model_name,
            args.max_output,
            args.timeout,
            allowed,
            vision=bool(args.mmproj),
        ),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
