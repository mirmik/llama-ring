# llama-ring

An experimental inference runner for studying long-running language-model
sessions with a bounded working context. Built on a pinned, patched **llama.cpp**.
No training or weight modification is required.

The runner summarizes earlier context using a branch of the already computed
cache. Summary work runs between foreground requests. When the working window
fills, the runner installs the prepared summary and shifts the recent tail,
keeping its existing attention cache and, for hybrid models, recurrent state.

**Research prototype / experimental alpha.** This is an implementation for
experiments, not evidence of lossless memory or reliable infinite context.
Summaries discard details; long-horizon quality and recurrent-state stability
remain research questions. See [architecture and limitations](docs/architecture.md).

## Build

Target: Linux, C++17 compiler, CMake 3.20+, Git, Python **3.10.12+**
(3.12 recommended for the bootstrap script), and OpenSSL development headers.
On Debian/Ubuntu, the system packages are `build-essential cmake git libssl-dev
python3-venv`. GPU builds additionally need the relevant SDK/toolchain.
Video and WebP decoding also require `ffmpeg` and `ffprobe` on `PATH`.

Run all commands below from this repository's root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/bootstrap.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --parallel 4
```

For **CPU**, omit `-DGGML_CUDA=ON` and pass `--gpu-layers 0` when starting
the runner. For **Vulkan**, use `-DGGML_VULKAN=ON` instead; the Vulkan SDK
and `glslc` must be installed. Requesting a GPU with no available device is an
error, rather than a silent CPU fallback. Other backends are unverified here.

[deps.json](deps.json) pins the upstream revision and patches. Bootstrap
extracts its own copy into `.deps/llama.cpp` and does not change other checkouts.
For offline bootstrap, use `python scripts/bootstrap.py --source /path/to/llama.cpp`
with a Git checkout containing the pinned commit. Bootstrap refuses to overwrite
an existing `.deps/llama.cpp`.

When upgrading an existing checkout, keep the old dependency/build directories
and create new ones: `python scripts/bootstrap.py --destination .deps/llama-new`,
then configure with `-DLLAMA_SOURCE=$PWD/.deps/llama-new` in a new build directory.
The current pin changes native snapshot layout; old saved slots must be rebuilt
from their transcripts.

## Start a conversation

Provide your own GGUF. Supported architectures are **qwen3** (full attention)
and **qwen35** (hybrid, with optional images), plus experimental
**qwen4exp** (Qwen3.8-Flash-Next, with optional images and a separate MTP head); the model's metadata, not its filename,
determines its architecture. Tested families include Qwen3-4B-Instruct-2507,
Qwen3.5-derived 4B and 27B models. Other architectures are not implemented.
Model weights and their licenses are separate from this project.

Flash-Next needs substantially more model memory than the 27B configuration.
The IQ4_XS shards occupy about 87.25 GiB on disk. The loader explicitly keeps
architecture-marked lookup tables lazy, including the large PLE table, even when
Vulkan's automatic mode would load them in full. This does not bound memory used
by the remaining weights or GPU allocations. Start with a small context and
measure memory on the intended machine. ROCmFP4-FAST files with custom tensor
types 101/104 are not supported by this pin.

For the supplied Aurora QSA/MTP profile, build its separate directory:

```bash
python scripts/bootstrap.py --destination .deps/llama-qsa-mtp
cmake -S . -B build-qsa-mtp -DLLAMA_SOURCE="$PWD/.deps/llama-qsa-mtp" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-qsa-mtp --parallel 4
```

Download `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from
[ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/blob/main/mtp-Qwen3.8-Flash-Next-Q8_0.gguf)
into `.models/`, or set a different `--mtp-model` path. The tested file's SHA256
is `6be8a94e2d94107b7f38fdc957c3460ffeb6ff6302edef92c50738682f9dc0e0`.

For GPU runs with Q8 KV and MTP, the runner disables Vulkan MMVQ by default:
its additional activation quantization exceeded the measured error bound between
single-token and speculative decoding. Other backends ignore this Vulkan setting.
Flash-Next supports `--mtp-draft 3 --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`.
Use a compatible ggml-org MTP head, not the custom ROCmFP4 sidecar.
See [validation procedures](docs/experiments.md).
For Flash-Next vision/video on Vulkan, start with `GGML_VK_DISABLE_MMVQ=1`:
the automatic matvec mode exceeded the measured video cache-shift tolerance.
The [Aurora proxy profile](deploy/llm-proxy-flash-next.json) includes this setting.

```bash
python scripts/serve.py \
  --model /path/to/model.gguf \
  --context 32768 --prefix 8192 \
  --max-summary 2048 --max-output 2048 \
  --max-sessions 1 --ram-rotation 1 --cache-type q8_0
```

The default API model name is `vcortex-shadow` (a compatibility alias, independent
of the GGUF filename). Wait for model loading, then in another terminal:

```bash
curl http://127.0.0.1:8097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vcortex-shadow","session_id":"demo","messages":[{"role":"user","content":"Explain how a ring buffer works."}],"max_tokens":256,"temperature":0}'
```

For subsequent turns, send the same `session_id` and the **complete message
history**, including previous assistant/tool messages. The runner reuses cached
history while maintaining a smaller internal context. Set `"stream":true` for
SSE. Plain OpenAI-compatible clients can use `/v1/chat/completions`; the extra
session and context-state fields require client support for persistent memory.
See [API and integrations](NEMOR.md).

The server defaults to loopback. It has **no authentication or TLS**. For LAN
use, explicitly set `--host` and `--allow-client IP`, or put it behind a trusted
proxy. The peer allowlist is not a multi-user authorization system.

## Images (qwen35 and qwen4exp)

Add a compatible projector to the launch command:

```bash
python scripts/serve.py --model /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf --image-max-tokens 1024 \
  --context 32768 --prefix 8192 --max-summary 2048 \
  --ram-rotation 1 --cache-type q8_0 --max-sessions 1
```

Send user `content` as OpenAI-style text/image parts, for example:

```json
[{"type":"text","text":"Describe this image."},
 {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
```

PNG, JPEG, WebP, GIF, BMP and other supported raster images are accepted as
inline base64 data or HTTP(S) URLs. Limits:
8 MiB decoded per image, 16 million pixels, 32 images and 32 MiB request body.
`detail` is accepted but the projector's token limit controls image resolution.
For Flash-Next, the default (`--image-max-tokens -1`) takes resolution limits
from the projector. qwen35 retains its previous default of 1024 tokens. An
explicit limit overrides the default for either architecture.
Use `--mmproj-gpu 0` to encode images on CPU and keep GPU memory for the language
model. The default uses GPU when the language model does. `vision` in
`/v1/capabilities` is true only when a projector is configured and loaded.

Image embeddings enter the actual model context. Images are kept as atomic
chunks when compacting; a chunk in the retained tail keeps its cached visual
state and shifted M-RoPE coordinates. Once summarized, exact visual details may
be lost. A compaction can happen slightly early to fit an incoming image without
splitting it. RAM rotation and disk snapshots can be used with images. MTP drafts the text
answer from target hidden states after visual prefill.
Old text-only checkpoint files must be rebuilt after this format change.

Video uses llama-server's content format:
`{"type":"input_video","input_video":{"data":"BASE64","format":"mp4"}}`.
`input_video.url` also accepts an HTTP(S) URL or a base64 data URL. The upstream
FFmpeg helper samples at 4 FPS and supplies timestamps and temporal frame merging.
Each resulting image chunk is atomic during compaction; the entire clip need
not fit at once. Video data has the same 8 MiB per-item limit. Audio input is
not supported by these vision projectors.

## Memory settings

| Option | Meaning |
| --- | --- |
| `--context 32768` | Shared token budget for protected instructions, summary and recent history. Enables the current layout. |
| `--prefix 8192` | Maximum prepared prefix and scratch reserve; unused space is available to history. |
| `--max-summary 2048` | Maximum generated summary length. Leave room for system instructions and tool schemas in `--prefix`. |
| `--trigger 0.7` | Start preparing a summary at 70% of the shared working window. Commit it only when the window fills. |
| `--ram-rotation 1` | Park private summary/answer branches in host RAM between uses. The active context still has to fit on the device. |
| `--cache-type q8_0` | Quantized KV cache; `f16` is also supported. |
| `--max-sessions 1` | Maximum resident sessions. More sessions need more memory. |
| `--slot-save-path checkpoints` | Enable explicit full-session save/restore endpoints. No automatic disk eviction for direct `session_id` clients. |
| `--mtp-draft 3` | Optional speculative answers using a NextN head. Default is off. |
| `--mtp-model FILE` | Separate Flash-Next MTP GGUF; qwen35 uses its embedded head. Requires a positive draft count. |

The prefix must satisfy `instruction_tokens + max_summary + 128 <= prefix`.
Use `POST /v1/shadow/inspect` with a chat request to check the rendered prompt
before generation. Large tool schemas may require a larger prefix reserve.
Device allocation includes scratch space **in addition** to the working window.
Start small and measure your model; the example is not a VRAM guarantee.

MTP accelerates foreground generation only. Its draft cache starts at the current
answer suffix, so acceptance and speed depend on the task. Background summaries
use ordinary decoding. Draft settings can change across a checkpoint restore:
only target state is persisted, and draft KV starts fresh for each answer.
Target weights, projector, cache type and memory layout must still match.

Flash-Next caches QSA pooled keys in sequence-private F32 buffers. Branches
sharing raw keys keep independent pool groupings; edits and restore invalidate
affected derived entries. These buffers are rebuilt rather than serialized.
Their size per indexer layer is `4 * indexer_head_size *
(n_ctx / pool_size * n_seq_max + 1)` bytes, in addition to the KV cache.
Set `VCORTEX_QSA_CACHE=0` before startup to use the previous full-recomputation
path for comparisons. Caching is enabled by default in ring mode.

## Tests and experiments

```bash
ctest --test-dir build --output-on-failure
python -m pytest -q tests
```

These are CLI startup checks and model-free HTTP/protocol, scoring and packaging
tests. CI runs them on CPU; it does not download model weights or validate model
quality. For model-dependent checks and benchmark tools, see
[experiments](docs/experiments.md). Fixed P/A research CLIs remain available but
are separate from the shared-window HTTP path described above.

Known limits include no audio input, partial OpenAI API compatibility, summary
failure when generation does not finish within its token limit, and unproven
quality after many compression cycles. Under continuous load there is no idle
compute: the next foreground request must wait for unfinished summary work when
the window fills. Checkpoint formats are experimental and version/configuration
specific. Linux CPU, Vulkan and CUDA have been exercised; portability elsewhere
is not claimed.

## Source release and license

The code is [MIT licensed](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md).
Contributions and reproducible failure reports are welcome. Include the source
revision, GGUF metadata/quantization, backend, launch flags and a synthetic
reproducer; avoid posting private conversations or checkpoints.

For a minimal source export, including new files before they are committed:

```bash
python scripts/export.py --output dist/llama-ring
```

This copies the explicit [release manifest](RELEASE_FILES.txt) and writes a
sibling `llama-ring.tar.gz`. It excludes downloaded dependencies, weights,
checkpoints and laboratory runs. The exported directory can become a standalone
Git repository; publication is a separate step.
