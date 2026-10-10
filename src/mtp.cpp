#include "mtp.hpp"
#include "llama-ext.h"
#include <algorithm>

namespace ring {
MtpAnswer::MtpAnswer(Context & ctx, int capacity) : target(ctx), seq(ctx.physical(1)), n_max(ctx.model.mtp_draft) {
    require(n_max > 0, "MTP is disabled");
    auto p = llama_context_default_params();
    p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    p.ctx_other = ctx.ptr.get();
    p.n_ctx = capacity;
    p.n_batch = p.n_ubatch = ctx.batch_size;
    p.n_seq_max = 2;
    p.n_outputs_max = std::max(2, n_max + 1);
    p.kv_unified = true;
    p.n_threads = p.n_threads_batch = 8;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.type_k = p.type_v = ctx.model.cache_type;
    draft.reset(llama_init_from_model(ctx.model.mtp_model ? ctx.model.mtp_model.get() : ctx.model.ptr.get(), p));
    require(bool(draft), "Cannot create MTP context");
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.draft.ctx_tgt = ctx.ptr.get();
    params.draft.ctx_dft = draft.get();
    params.draft.n_max = n_max;
    params.draft.backend_sampling = false;
    spec.reset(common_speculative_init(params, 2));
    require(bool(spec), "Cannot initialize MTP driver");
}
MtpAnswer::~MtpAnswer() {
    spec.reset();
    llama_set_embeddings_nextn(target.ptr.get(), false, false);
}
void MtpAnswer::decode(const Tokens & tokens, int position, bool all_logits) {
    require(!tokens.empty() && tokens.size() <= size_t(target.batch_size), "Invalid MTP batch");
    auto batch = llama_batch_init(tokens.size(), 0, 1);
    try {
        batch.n_tokens = tokens.size();
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq;
            batch.logits[i] = all_logits || i + 1 == batch.n_tokens;
        }
        require(llama_decode(target.ptr.get(), batch) == 0, "MTP target decode failed");
        // Discard speculative head KV before teacher-forcing target hidden rows.
        require(llama_memory_seq_rm(llama_get_memory(draft.get()), seq, position, -1), "MTP draft trim failed");
        common_batch processed(target.ptr.get());
        for (int i = 0; i < batch.n_tokens; ++i) processed.add(batch.token[i], batch.pos[i], seq, batch.logits[i]);
        require(common_speculative_process(spec.get(), processed), "MTP hidden-state update failed");
    } catch (...) { llama_batch_free(batch); throw; }
    llama_batch_free(batch);
}
void MtpAnswer::prefill(const Tokens & suffix, int position) { decode(suffix, position, false); }
Tokens MtpAnswer::next(common_sampler * sampler, llama_token last, int & position, int remaining) {
    require(remaining > 0, "Empty MTP generation budget");
    Tokens proposal;
    auto & dp = common_speculative_get_draft_params(spec.get(), seq);
    dp.drafting = true; dp.n_max = std::min(n_max, remaining - 1);
    dp.pos0 = position; dp.id_last = last; dp.result = &proposal; dp.prompt = nullptr;
    if (dp.n_max > 0) common_speculative_draft(spec.get());
    else dp.drafting = false;
    // Enforce the output budget even if a driver ignores the per-sequence cap.
    if (proposal.size() > size_t(remaining - 1)) proposal.resize(remaining - 1);
    Tokens verification{last}; verification.insert(verification.end(), proposal.begin(), proposal.end());
    decode(verification, position, true);
    Tokens result;
    size_t matched = 0;
    for (size_t i = 0; i <= proposal.size(); ++i) {
        if(observe_logits) observe_logits(llama_get_logits_ith(target.ptr.get(),i));
        const auto token = common_sampler_sample(sampler, target.ptr.get(), i);
        common_sampler_accept(sampler, token, true);
        result.push_back(token);
        if (llama_vocab_is_eog(target.model.vocab, token) || i == proposal.size() || token != proposal[i]) break;
        ++matched;
    }
    common_speculative_accept(spec.get(), seq, result.size() - 1);
    if (observe_verification) observe_verification(verification,result.size());
    const int keep = position + result.size();
    require(llama_memory_seq_rm(target.memory(), seq, keep, -1), "MTP target rollback failed");
    require(llama_memory_seq_rm(llama_get_memory(draft.get()), seq, keep, -1), "MTP draft rollback failed");
    ++steps; proposed += proposal.size(); accepted += matched;
    rollbacks += verification.size() > result.size();
    position = keep;
    return result;
}
json MtpAnswer::stats() const {
    return {{"enabled",true},{"max_draft",n_max},{"steps",steps},{"proposed",proposed},
        {"accepted",accepted},{"rollbacks",rollbacks},{"draft_cache","request_local"}};
}
}
