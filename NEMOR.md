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

Chat supports function tools (`tool_choice: auto/none/required` or a named
function), parallel tool calls, one completion,
`temperature`, `top_p`, `top_k`, `min_p`, repeat/frequency/presence penalties and
`seed`. Thinking uses `chat_template_kwargs: {"enable_thinking": true}`;
`reasoning_content` is supported in both history and responses. Keep the returned
assistant message, including reasoning and tool calls, in subsequent history.
`reasoning_budget`, `reasoning_budget_tokens` and `thinking_budget_tokens` accept
-1 (unlimited) or a nonnegative token budget. Flash-Next defaults to thinking
enabled; set `enable_thinking: false` to disable it. Other architectures retain
their previous default (disabled). `response_format` accepts `json_object` and
`json_schema`; `stop` accepts a string or array and is honored in JSON and SSE.
Images require a compatible `--mmproj`; raster image data URLs, raw base64 and
HTTP(S) URLs are accepted. `input_video` accepts `data` or `url` and uses the
upstream FFmpeg decoder (4 FPS, timestamps and temporal merging).
Remote media are fetched before inference with a 30-second timeout and the
same size limits as inline images. This is a subset of the OpenAI API, not a drop-in replacement
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
- [Aurora Flash-Next runtime](deploy/llm-proxy-flash-next.json): concrete IQ4_XS
  profile for the existing `model-layout` alias `qwen3.8-flash-next-iq4-xs`,
  port 8103, native vision projector, shared context 262144, protected-prefix
  reserve 24576, summary limit 8192, answer limit 32768 and one resident slot.
  It sets `GGML_VK_DISABLE_MMVQ=1`: Vulkan's automatic quantized matvec mode
  exceeded the measured tolerance for shifting a video context. This preserves
  Q8 KV while using floating-point matvec inputs. Keep this setting when
  starting the Flash-Next vision worker outside the proxy as well.
  It uses `build-qsa-mtp`, enables sequence-private QSA caching and MTP3,
  and requires the separate Q8 head under `llama-ring/.models/`.
  The head path can be changed with `--mtp-model`.
  Merge its runtime entry and inference timeout into the existing proxy config; it is not a complete
  replacement config. Preserve existing autoload settings selected by
  `model-layout`. Before switching, stop the old runtime and archive its native
  checkpoint directory; those files cannot be restored by this runner. The
  model alias, proxy URL, credentials and Nemor connection profile stay valid.
  Atom/Atom-subagent already use runner context management in the inspected
  installation; Tet also needs `context_management: "runner"` when using ring
  memory. Large first requests can require a longer client/proxy timeout than
  short cached turns; the runner profile allows 1800 seconds. For these
  characters, merge `"context_management": "runner"`, `"send_session_id": true`
  and `"request_timeout": 1800` for a manual deployment.
- [model-layout profiles](deploy/model-layout/README.md): `flash-ring` and
  `27b-ring` select separate runtime aliases and Nemor profiles, with dedicated
  checkpoint directories. They switch context management and timeouts together
  with the connection profile; `flash` and `27b` retain the native runtimes.

Choose one process manager for the device. Neither example automatically stops
other model services. Optional integration probes require separate checkouts:
`python scripts/test_nemor.py --help` and
`python scripts/probe_atom_slots.py --help`. Their external client contracts may
change independently; the core runner and its CI do not import these projects.
