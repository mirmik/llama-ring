# Architecture and research scope

## Shared working window

The HTTP runner places tokens contiguously:

```text
protected instructions + tool schemas | current summary | recent history
```

With `--context TOTAL`, there is no fixed unused P/A gap. The prefix occupies
its actual length, and history receives the remainder of the shared budget.
`--prefix MAX_PREFIX` bounds a prepared replacement and its scratch allocation.
The older fixed layout remains available through `--prefix P --active A`
without `--context` or `--shared-context 1`.

At the configured threshold (normally 70% of the shared window), the runner
captures a cache branch of the prefix and the history accumulated so far. It
asks the model `Суммаризируй предыдущий текст.` (“Summarize the preceding text.”).
This is a general summary, without a list of fact categories to preserve.
Original system instructions and tool definitions are kept separately and
reinserted unchanged into the replacement prefix.

One worker thread owns the contexts and compute device. Summary preparation
advances only while the foreground queue is empty, in quanta of up to 64 prefill
tokens or one generated token. This is cooperative idle scheduling, not a
simultaneous low-priority GPU stream. A ready candidate waits until capacity is
reached. If it is not ready then, the foreground request waits for completion.

On commit, the summarized part is removed and the unsummarized tail is moved by
`new_prefix_length - old_history_start - discarded_position_extent`.
For text the extent equals the token count; an image's extent is its M-RoPE
grid extent, rather than the number of occupied KV cells.
Attention keys receive a lazy RoPE shift before the next decode. Cached tail
states are retained; they are not recomputed under the new prefix. New tokens
therefore see the replacement summary together with states formed under the old
context. This mixed history is the experiment's central approximation.

The model receives a memory-revision notice. Clients receive `context_state`
with identity, revision, sizes and reason (`created`, `rebuilt`, `compacted`,
`reused`). A new identity denotes a rebuilt session; an increased revision
with the same identity denotes compaction.

## Hybrid state and upstream patches

For qwen35, full-attention KV can be shifted, but recurrent state cannot be
reconstructed from an arbitrary slice of KV. The runner retains the tail's
recurrent state, keeps a separate prefix state, and snapshots complete branch
states. Discarded history may continue influencing recurrent layers. Bounded
storage does not imply bounded effective history or long-horizon stability.

The [pinned patches](../deps.json) provide opt-in noncontiguous positions,
access to hybrid memory components, qwen35 position shifts and
quantized partial-RoPE shifts. The runner enables
`VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS=1` itself. These patches do not make arbitrary
models or other llama.cpp applications safe to use with sparse positions.
For qwen35 images, all three M-RoPE axes receive the same translation; relative
patch-grid coordinates are preserved. Token counts and temporal position extents
are tracked separately: many image KV cells can share one temporal coordinate.
Only complete image chunks can be evicted. General nonuniform axis edits and
video/audio are not supported. The ordinary attention gate is preserved.

## RAM rotation and disk snapshots

`--ram-rotation 1` keeps the live main sequence and one physical scratch sequence
on the device. Private branch KV/recurrent state is parked in host RAM and
restored on branch switches. Consecutive idle steps keep the summary branch
resident. Shared history is not transferred on every generated token.

The approximate requested KV allocation in this mode is:

```text
context + max(max_prefix, max_summary + 1024, max_output + 512)
```

Model weights, recurrent buffers, MTP state, compute buffers and allocator
alignment add further memory. This is branch rotation, not arbitrary paging of
the main context or CPU attention.

Explicit slot checkpoints contain native state, parked RAM branches, token
ledger, layout, context identity and pending summary work/logits. They use
checksums, compatibility checks, reserved disk space and atomic replacement.
They are private files (0600), but are not encrypted. Treat them as conversation
data. This format is incompatible with ordinary llama-server snapshots and is
not a stable long-term storage format. Direct `session_id` eviction is in-memory
LRU; a client or proxy must save before eviction for durable continuity.

## MTP

A compatible qwen35 model can use its single NextN head for speculative answers.
The target model verifies candidates through the ordinary sampler/grammar chain;
rejected tokens are not emitted. Attention and recurrent state roll back together.
Draft state is request-local and starts at the generation suffix. Background
summaries do not use MTP. Quantized single-token and batched kernels may yield
slightly different logits; exact cross-backend determinism is not promised.

## What is and is not established

Local synthetic tests have exercised cache isolation, shifts, RAM round trips,
checkpoint restoration, tools/thinking and MTP on selected qwen3/qwen35 models.
These tests establish mechanics for their inputs. They do not establish
lossless summarization, universal speedups, broad model support, or indefinite
coherence beyond the model's trained context range.

Failure to produce a completed summary within `--max-summary` remains an error;
there is no robust retry/recovery policy yet. Sustained foreground load can
starve background work. Changing earlier messages, system/tools or thinking
settings can force a rebuild. No video/audio, arbitrary stop strings, required tool
choice or response JSON-schema constraints are implemented. Client history and
the runner's token ledger for matching that history can continue growing even
though the working token window is bounded. This is not a constant-total-RAM
guarantee for arbitrarily long sessions.

## Image identity and lifecycle

`mtmd` renders image embeddings with the model's matching projector. Subsequent
requests tokenize the multimodal history but encode/decode only its new suffix.
A SHA-256 ledger of original image bytes prevents reusing the same placeholder
IDs for changed images. A changed earlier image forces a rebuild.

Snapshots include image spans, position extents and identity hashes along with
the full native visual KV/recurrent state. Raw image files are not separately
stored by the runner. The client still sends the original history, including
images. A shadow snapshot is taken only at an image boundary; summaries see its
cached visual state. Cancellation is checked before/after encoder execution and
between decoder batches; a single vision-encoder call cannot be preempted.
