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

## Synthetic streaming CLIs

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
