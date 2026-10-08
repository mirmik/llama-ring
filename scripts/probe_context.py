#!/usr/bin/env python3
"""Isolated CUDA sizing run; run with the production model service stopped."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--active", type=int, required=True)
    p.add_argument("--cache-type", choices=("f16", "q8_0"), required=True)
    p.add_argument("--prefix", type=int, default=6144)
    p.add_argument("--ram-rotation", type=int, choices=(0,1), default=0)
    p.add_argument("--worker")
    p.add_argument("--allow-recall-miss", action="store_true")
    p.add_argument("--summary", type=int, default=512)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8098)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=False)
    url = f"http://127.0.0.1:{a.port}"
    command = [sys.executable, str(root / "scripts/serve.py"), "--model", a.model,
               "--port", str(a.port), "--prefix", str(a.prefix), "--ram-rotation", str(a.ram_rotation), "--active", str(a.active),
               "--max-output", "1024", "--max-summary", str(a.summary),
               "--max-sessions", "1", "--cache-type", a.cache_type]
    if a.worker:
        command += ["--worker", a.worker]
    result = {"command": command, "active": a.active, "cache_type": a.cache_type,
              "summary": a.summary, "completed": False}
    start = time.monotonic()
    server = monitor = None
    with (a.output / "server.log").open("w") as log, (a.output / "vram.csv").open("w") as vram:
        try:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.free",
                                        "--format=csv,noheader,nounits", "-lms", "200"], stdout=vram)
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {server.returncode}")
                try:
                    if requests.get(url + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("Server startup")
            with (a.output / "probe.log").open("w") as probe_log:
                subprocess.run([sys.executable, str(root / "scripts/probe_remote.py"),
                                "--url", url, "--output", str(a.output / "probe.json"),
                                "--repeat", str(36 * max(1, a.active // 4096)),
                                "--max-turns", "24"] + (["--allow-recall-miss"] if a.allow_recall_miss else []), stdout=probe_log, stderr=subprocess.STDOUT,
                               timeout=2400, check=True)
            # Real SSE disconnect after content, then wait for the canceled
            # session to disappear. Isolate it from the recall session.
            before = requests.get(url + "/v1/shadow/sessions", timeout=30).json()
            with requests.post(url + "/v1/chat/completions", json={
                "model": "vcortex-shadow", "session_id": "sizing-cancel",
                "messages": [{"role": "user", "content": "Count from 1 to 10000, one number per line."}],
                "max_tokens": 1024, "temperature": 0, "stream": True,
            }, stream=True, timeout=600) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: {"):
                        item = json.loads(line[6:])
                        if any(c.get("delta", {}).get("content") for c in item.get("choices", [])):
                            break
                else:
                    raise RuntimeError("No content before cancellation")
            for _ in range(100):
                after = requests.get(url + "/v1/shadow/sessions", timeout=30).json()
                if after["cancellations"] > before["cancellations"] and "sizing-cancel" not in after["sessions"]:
                    break
                time.sleep(.1)
            else:
                raise RuntimeError("Cancellation did not release session")
            result["cancellation"] = after
            result["completed"] = True
        except BaseException as exc:
            result["error"] = repr(exc)
            raise
        finally:
            for proc in (server, monitor):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            vram.flush()
            samples = [line.split(",") for line in (a.output / "vram.csv").read_text().splitlines()]
            if samples:
                result["peak_used_mib"] = max(int(row[1]) for row in samples)
                result["min_free_mib"] = min(int(row[2]) for row in samples)
            result["elapsed_s"] = time.monotonic() - start
            (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
