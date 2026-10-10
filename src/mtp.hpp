#pragma once
#include "runtime.hpp"
#include "sampling.h"
#include "speculative.h"
#include <functional>

namespace ring {
// Request-local draft KV: no stale hidden state survives branch rotation,
// a summary commit, cancellation, or disk restore. Target KV is authoritative.
// Draft attention starts at the generation suffix (a cold proposal cache).
class MtpAnswer {
    Context & target;
    int seq, n_max;
    std::unique_ptr<llama_context, decltype(&llama_free)> draft{nullptr, llama_free};
    common_speculative_ptr spec;
    uint64_t steps = 0, proposed = 0, accepted = 0, rollbacks = 0;
    void decode(const Tokens & tokens, int position, bool all_logits);
public:
    // Optional read-only numerical oracle for tests; never set by the server.
    std::function<void(const float *)> observe_logits;
    std::function<void(const Tokens &, int)> observe_verification;
    MtpAnswer(Context & target, int capacity);
    ~MtpAnswer();
    void prefill(const Tokens & suffix, int position);
    // last was sampled/accepted but not decoded. Returns sampled tokens,
    // including a bonus/mismatch token not yet decoded, and advances position.
    Tokens next(common_sampler * sampler, llama_token last, int & position, int remaining);
    json stats() const;
};
}
