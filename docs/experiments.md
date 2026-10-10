# Experiments and validation

Run commands from the repository root. Store private outputs under `runs/`
(ignored by Git). Weights and benchmark datasets are supplied separately.
The short CPU CI suite tests startup and model-free protocols, not inference
quality. Model tests below may take minutes and allocate substantial device/RAM
memory. Stop competing workloads yourself when necessary.

## Native state checks

```bash
mkdir -p runs
build/bin/llama-ring --model /path/to/model.gguf \
  --self-test 1 --output runs/ring-checks.json
build/bin/llama-shadow --model /path/to/model.gguf \
  --self-test 1 --output runs/shadow-checks.json
build/bin/llama-shadow --model /path/to/model.gguf \
  --ram-self-test 1 --output runs/ram-checks.json
build/bin/llama-session-test /path/to/model.gguf 99
```

The first three commands accept `--gpu-layers 0` for CPU. The session test's
second positional argument is GPU layers (use 0 for CPU). Session checks cover
pending-summary phases, fixed/shared layouts and RAM rotation. For a compatible
qwen35 model with a single NextN head, a GPU build can additionally run:

```bash
build/bin/llama-mtp-test /path/to/model-with-mtp.gguf 3
build/bin/llama-session-test /path/to/model-with-mtp.gguf 99 3
```

MTP tests compare generation, restored state and grown/shrunk prefixes with
ordinary decoding. Quantized batching changes numerical results: teacher-forced
logits are checked with KL bounds and an independent batching control; exact
greedy equality is reported separately. Do not interpret a passing test as
bitwise equivalence for arbitrary hardware or prompts.

The MTP probe also accepts GPU layers and cache type:
`llama-mtp-test MODEL 3 0 q8_0` (CPU), or `MODEL 3 99 f16` (GPU/f16).
`VCORTEX_MTP_DIAGNOSTIC=1` records per-token KL and replays the exact verifier
batches without an MTP head, separately controlling hidden-state export and
rejected-token rollback. `VCORTEX_MTP_INITIAL_ONLY=1` limits that diagnostic
to the initial case and labels its output accordingly; omit it for the full
eight-case acceptance check.

For a negative control of the Vulkan Q8 workaround,
`VCORTEX_MTP_VULKAN_AUTO=1` retains the backend's original automatic MMVQ
selection (with both `GGML_VK_FORCE_MMVQ` and `GGML_VK_DISABLE_MMVQ` unset).
This reproduces the failing numerical case on the tested Radeon 8060S. Normal
Q8 MTP runs select the floating-input path; model weights and KV remain quantized.

## Synthetic streaming CLIs

For Flash-Next, run the QSA probe before a streaming experiment:

```bash
build/bin/llama-qsa-test /path/to/first-shard.gguf 99 f16 2208 baseline
build/bin/llama-qsa-test /path/to/first-shard.gguf 99 q8_0 2208 memory
```

The baseline compares ordinary upstream pooling with ring pooling on the same
weights, prompt and teacher-forced continuation. The memory mode also checks
different batch partitions, uniform shifts, branches with different QSA block
boundaries, native disk restoration, RAM branch export/import and ten prepared
prefix replacements with growing/shrinking prefixes and full active eviction.
Using more than 2048 prompt tokens exercises sparse selection. The prepared
prefix test supplies a known fact; it does not test the quality of generated
summaries. Use the streaming CLI for that separate experiment.

To compare the sequence-private QSA cache with the previous ring implementation,
set `VCORTEX_QSA_REFERENCE_RING=1` for the QSA probe. It creates the reference
with `VCORTEX_QSA_CACHE=0` and the tested contexts with caching enabled.

Flash-Next MTP and the three-arm decode benchmark use a separate head:

```bash
GGML_VK_DISABLE_MMVQ=1 build/bin/llama-mtp-test TARGET 3 99 q8_0 MTP_HEAD
GGML_VK_DISABLE_MMVQ=1 build/bin/llama-qsa-mtp-bench TARGET MTP_HEAD 16000 256
GGML_VK_DISABLE_MMVQ=1 build/bin/llama-qsa-mtp-bench TARGET MTP_HEAD 16000 256 oracle
```

The MTP test includes checkpoint migration with draft counts 0 → 3 → 0.
The benchmark reports history prefill, suffix preparation, decode tokens/s and
draft acceptance separately for code and Russian prose. The optional `oracle`
mode runs only the MTP arm, traces its logits, and teacher-forces the same tokens
through independent serial and four-token target decoders. It checks the same
numerical envelope as the MTP regression test. Tracing changes timings: keep
oracle runs separate from throughput measurements. Greedy free-running texts
need not remain identical when serial and batched numerical rounding differs.
Every oracle arm clears memory and restores the same main snapshot before
forking: deleting the answer branch alone can change physical KV placement and
confound the numerical comparison. The main snapshot must remain byte-identical
after MTP. Use `diagnostic` instead of `oracle` to also replay the exact verifier
batches, independently toggling hidden-state export and rejected-token rollback.
Diagnostic mode reports failures without failing the process; inspect its
`oracle.passed` fields. Ordinary `oracle` mode exits nonzero on a failed bound.

`llama-qsa-test MODEL 99 q8_0 717 summary fixture.json` records three summary
requests against both a cached branch with a fixed P/A gap and a fresh contiguous
upstream context. The fixture uses the streaming CLI schema. This diagnostic
reports generated text and logits; it does not assert summary quality or
equivalence between different position layouts.

On shared-memory GPUs, monitor system available RAM as well as process/cgroup
memory: device allocations may not be included in the cgroup's memory counter.
Lazy PLE avoids eager loading of that table but is not a whole-model memory cap.

```bash
python scripts/make_fixture.py --out runs/fixture.json
build/bin/llama-ring --model /path/to/model.gguf \
  --input runs/fixture.json --output runs/ring.json \
  --prefix 4096 --active 8192 --evict 1024 --summarizer cached
build/bin/llama-shadow --help
```

`llama-ring` is the earlier fixed P/A, append-only-summary experiment. It uses
short greedy answers and a fact-oriented synthetic fixture; it is not the HTTP
runner's current summary policy. `llama-shadow` exercises background preparation
and replacement offline. Read each CLI's help for its specific options. JSON
results with `completed: true` indicate completed runs, not necessarily correct
recall. Factual recall, state mechanics and speed are separate outcomes.

For a running HTTP backend, `scripts/probe_remote.py --help` describes a synthetic
multi-turn/compaction probe. Optional probe dependencies are installed with
`python -m pip install -r requirements-experiments.txt`. Exact fact retention is
not guaranteed by general summarization; `--allow-recall-miss` records misses
separately from mechanical checks.

## MemoryAgentBench pilot

When preparing a Flash-Next fixture, pass `--architecture qwen4exp` to
`scripts/memory_bench.py prepare`. The helper reads the architecture's current
summary instruction from `src/shadow.cpp` and records its source hash. The
default remains `qwen35` for the existing pilot protocol.

`scripts/memory_bench.py` prepares a selected, pinned pilot and scores
`llama-memory-bench` output. This is an adapted streaming protocol, not an
implementation of every official benchmark condition. It separates gold answers
from model input and keeps failed questions in score denominators.

Supply the upstream MemoryAgentBench Git checkout at revision
`538026089d1a8a8eff05121d0db89b388f360eba`, and the files
`Conflict_Resolution.parquet` and `Accurate_Retrieval.parquet` from
`ai-hyz/MemoryAgentBench` dataset revision
`7ea066982b140a19337e17e60d45d4076e042faf`. Install the experiment dependencies:

```bash
python -m pip install -r requirements-experiments.txt
python scripts/memory_bench.py prepare \
  --upstream /path/to/MemoryAgentBench \
  --data /path/to/parquet-directory --output runs/fixture-bench
build/bin/llama-memory-bench --help
python scripts/memory_bench.py score --help
```

Preparation records source revisions, file hashes, selection seed and the current
HTTP summary instruction with its actual source hash. It copies upstream license
and scoring/template provenance. Generated inputs and gold data are not shipped
with this repository. Consult the data/model licenses before redistribution.

## Reporting results

Record source and upstream revisions, model metadata/quantization, device and
backend, all flags, input/seed, completion/failure counts and raw output. Separate
prefill, generation and branch transfer time. Lazy RoPE work is charged to the
next decode, not merely the metadata shift. `ring-speed` is an optional target
(`-DRING_BUILD_SPEED_PROBE=ON`). The historical `scripts/run_speed.py` expects
`build-stock/bin/stock-speed` (compile `src/speed_probe.cpp` against unpatched
llama.cpp without `RING_SPEED_RUNTIME`) and `build-stock/bin/llama-bench`, both
built separately at the pinned upstream revision. It does not establish general
model-quality parity.

The project originated in a local research lab. Source releases deliberately
omit private transcripts, deployment history and raw lab artifacts. The included
programs are reproducible mechanisms and probes; there is no claim of a reviewed
paper, broad benchmark ranking or validated infinite-context quality.

## Native vision checks

```bash
build/bin/llama-vision-test /path/to/model.gguf /path/to/mmproj.gguf /path/to/messages.json
```

`messages.json` is an OpenAI-style user message array containing one inline image.
This test compares positive/negative M-RoPE translations with a separately
prefilled reference and checks RAM/disk round trips, using Q8 KV. The optional
last argument selects GPU layers (0 for CPU). The token/color checks are
synthetic mechanism checks, not a vision-quality benchmark. Python HTTP tests
also cover image validation, capability reporting and history replay.
