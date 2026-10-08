# API and optional Nemor / llm-proxy integration

The HTTP entry point is `scripts/serve.py`; its default model alias is
`vcortex-shadow`. Core inference does not depend on Nemor or llm-proxy.
Build and start it using the [quickstart](README.md).

## HTTP surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Worker health |
| `GET /v1/models`, `GET /v1/capabilities` | Model alias and supported features |
| `POST /v1/chat/completions` | Chat, JSON or SSE |
| `POST /tokenize` | Tokenize text |
| `POST /v1/shadow/inspect` | Render a chat request and check prefix fit without generation or session creation |
| `GET /v1/shadow/sessions` | Session diagnostics |
| `DELETE /v1/shadow/sessions/{session_id}` | Reset a named session |
| `POST /slots/{id}?action=save\|restore\|erase` | Explicit slot checkpoint operations |

Chat supports function tools (`tool_choice: auto/none`), one completion,
`temperature`, `top_p`, `top_k`, `min_p`, repeat/frequency/presence penalties and
`seed`. Thinking uses `chat_template_kwargs: {"enable_thinking": true}`;
`reasoning_content` is supported in both history and responses. Keep the returned
assistant message, including reasoning and tool calls, in subsequent history.
`reasoning_budget` supports -1 (no separate budget) or 0 with thinking disabled.
Images require a compatible `--mmproj` and inline PNG/JPEG data URLs. Required
tool choice, arbitrary stop sequences and response schemas are rejected. This is a subset of the OpenAI API, not a drop-in replacement
for every client or llama-server endpoint.

Keep the complete transcript on the client. Use a stable `session_id`, or
`id_slot` when an external manager owns slots; do not send both. Changes to
previous messages, protected instructions, tools or thinking can rebuild state.
A successful response includes `context_state` and runtime diagnostics.
SSE carries context metadata before answer chunks. Clients should display the
actual `context_tokens`/`context_capacity` rather than count the full transcript
as the current window. `context_id` changes on rebuild; `revision` changes on
compaction. Tool arguments are emitted after JSON validation, not as partially
executable fragments.

Default limits: 8 MiB request body (32 MiB with vision), queue length 8, request timeout 600 seconds.
A disconnected stream is canceled. No authentication/TLS is provided; use a
trusted local client or proxy. Diagnostic endpoints expose session information.

## Slot persistence

Start with `--slot-save-path /path/to/private/checkpoints`. Send chat requests
with `"id_slot":0`, then explicitly save or restore:

```bash
curl http://127.0.0.1:8097/slots/0?action=save \
  -H 'Content-Type: application/json' -d '{"filename":"conversation-1"}'
curl http://127.0.0.1:8097/slots/0?action=restore \
  -H 'Content-Type: application/json' -d '{"filename":"conversation-1"}'
```

Filenames must be simple basenames inside the configured directory. The saved
state includes shadow work; restoring it does not require replaying all tokens.
Configuration/model compatibility is checked. A checksum catches corruption;
files from ordinary llama-server are not compatible. A model or runner upgrade
may require rebuilding sessions. Direct named-session LRU does **not** save to
disk automatically. RAM branch rotation and disk session switching are distinct.

## Nemor

The example [backend settings](config/nemor-shadow.json) use a direct loopback
URL without `/v1`. Set the server URL for your installation. In the character,
use `context_management: "runner"` and `send_session_id: true`; disable
client-side automatic compaction. Enable `read_image` only when the backend
reports `vision: true`; this requires `--mmproj`. Nemor's saved `_images` and
`read_image` results are serialized as inline `image_url` parts by its client and
can be replayed on later turns. The original history need not be rewritten.
The example text-only character disables image tools until vision is configured.
Nemor needs a version that understands `context_state` and runner-managed
context. The [shadow-lab character](config/shadow-lab.character.json) is an
optional experiment profile, not a prerequisite.

[Atom](config/atom-shadow.patch.json) and
[subagent](config/atom-subagent-shadow.patch.json) examples contain only fields
to merge into existing characters. `model-vcortex-shadow` is a local profile
name to create or replace with your own. Do not replace a whole character with
these fragments. The client's original transcript remains its responsibility.

## Deployment examples

- [systemd user unit](deploy/llama-ring.service): replace `/path/to` paths before
  installing. It binds to loopback and uses the shared-window settings.
- [llm-proxy runtime](deploy/llm-proxy-shadow.json): replace paths and configure
  the proxy's own environment. `{python}`, `{name}`, `{port}`, `{slots}` and
  `{runtime_kv}` are proxy placeholders. Its SlotManager can save one session,
  load another into the sole slot, and restore the first on return.

Choose one process manager for the device. Neither example automatically stops
other model services. Optional integration probes require separate checkouts:
`python scripts/test_nemor.py --help` and
`python scripts/probe_atom_slots.py --help`. Their external client contracts may
change independently; the core runner and its CI do not import these projects.
