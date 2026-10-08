#include "runtime.hpp"
#include <iostream>

namespace ring {
json hybrid_self_test(Model & model) {
    require(model.hybrid, "Hybrid test requires qwen35");
    json result = {{"architecture", model.architecture}, {"name", model.name},
                   {"policy", "preserve_stream_recurrence_separate_prefix"}};
    const auto p = model.tokens(message("system", "Remember facts. Answer code questions with four digits only. If unknown, answer UNKNOWN."));
    const auto delta = model.tokens(message("system", "Memory: the code of Linden is 2850."));
    const auto question = model.tokens(message("user", "What is the code of Linden?") + model.generation_prompt());
    {
        Context baseline(model, 2048);
        Tokens all = p;
        all.insert(all.end(), delta.begin(), delta.end());
        all.insert(all.end(), question.begin(), question.end());
        auto answer = baseline.generate(all, 0, 32);
        result["ordinary_context_baseline"] = answer.metrics();
        std::cerr << result.dump(2) << std::endl;
        require(answer.eos && answer.text == "2850", "Ordinary model baseline failed before ring test");
    }
    Stream stream(model, 512, 256);
    stream.append_prefix(p);
    auto prefix_state = stream.ctx.state(3);
    const auto one = model.tokens(" weather").front();
    stream.append(Tokens(128, one));
    require(stream.ctx.state(3) == prefix_state, "A overwrote prefix state");
    result["prefix_recurrence_isolated_from_active"] = true;
    auto recurrence = stream.ctx.recurrent_state();
    auto old_a = stream.ctx.range_state(512, -1);
    auto old_p = stream.ctx.range_state(0, p.size());
    stream.append_prefix(delta);
    require(recurrence == stream.ctx.recurrent_state(), "P append overwrote A recurrence");
    require(old_a == stream.ctx.range_state(512, -1), "P append overwrote A attention");
    require(old_p == stream.ctx.range_state(0, p.size()), "P append changed old P attention");
    result["prefix_append_preserves_active_recurrence_and_old_kv"] = true;
    auto before = stream.ctx.state();
    auto answer = stream.query(question);
    require(before == stream.ctx.state(), "Query modified persistent hybrid state");
    result["answer_after_prefix_append"] = answer.metrics();
    std::cerr << result.dump(2) << std::endl;
    require(answer.text == "2850" && answer.eos, "New prefix was not read");
    result["query_branch_preserves_hybrid_state"] = true;
    bool overflow = false;
    const auto p_before = stream.ctx.state(3);
    try { stream.append_prefix(Tokens(513, one)); } catch (const std::runtime_error &) { overflow = true; }
    require(overflow && before == stream.ctx.state() && p_before == stream.ctx.state(3), "Overflow changed state");
    result["prefix_overflow_atomic"] = true;
    Context reference(model, 2048);
    reference.restore(before);
    require(llama_memory_seq_rm(reference.attention_memory(), 0, 512, 576), "Reference eviction failed");
    llama_memory_seq_add(reference.attention_memory(), 0, 0, 512, 64);
    recurrence = stream.ctx.recurrent_state();
    stream.evict(64);
    // Recurrent tensors are unchanged; restore only the shifted position for
    // the serialized comparison, then put its new position back.
    llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, 64);
    require(recurrence == stream.ctx.recurrent_state(), "Eviction changed recurrent tensors");
    llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, -64);
    result["eviction_preserves_recurrent_tensors"] = true;
    answer = stream.query(question);
    result["answer_after_shift"] = answer.metrics();
    auto absolute = reference.generate(question, 640, 32);
    result["text_rope_rebase_vs_absolute"] = compare_logits(answer.first_logits, absolute.first_logits);
    require(result["text_rope_rebase_vs_absolute"]["kl"].get<double>() < 1e-3 &&
            result["text_rope_rebase_vs_absolute"]["top1_equal"].get<bool>(), "Text M-RoPE shift mismatch");
    require(answer.eos && answer.text == "2850", "Shift lost appended memory");
    stream.append(Tokens(192, one));
    auto size = stream.ctx.state().size();
    for (int i = 0; i < 8; ++i) {
        stream.evict(64);
        stream.append(Tokens(64, one));
        require(stream.ctx.state().size() == size, "Hybrid state grows with evictions");
    }
    result["bounded_state_after_8_evictions"] = {{"serialized_bytes", size}, {"active_tokens", stream.active.size()}};
    recurrence = stream.ctx.recurrent_state();
    stream.evict(256);
    require(llama_memory_seq_pos_max(stream.ctx.recurrent_memory(), 0) == 511, "Full eviction did not rebase recurrent tail");
    llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, 256);
    require(recurrence == stream.ctx.recurrent_state(), "Full eviction changed recurrent tensors");
    llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, -256);
    stream.append(Tokens(64, one));
    result["full_eviction_and_refill"] = true;
    // Recompute the complete saved prefix independently with identical batch
    // partition. Compare its next-token distribution to the persistent P branch.
    reference.remove(0);
    reference.decode(p, 0);
    reference.decode(delta, p.size());
    auto isolated = reference.generate(question, p.size() + delta.size(), 16);
    auto saved = stream.ctx.generate(question, stream.prefix.size(), 16, 3);
    result["prefix_branch_vs_independent"] = compare_logits(saved.first_logits, isolated.first_logits);
    require(result["prefix_branch_vs_independent"]["kl"].get<double>() < 1e-3 && saved.text == isolated.text,
            "Prefix recurrence does not match independent prefill");
    Summarizer summary(model, "Код объекта Кедр: 2247.", 2048, true);
    auto state = summary.ctx.state();
    auto generated = summary.summarize(model.tokens("Код объекта Липа: 9980. Погода ясная."));
    require(state == summary.ctx.state(), "Summary branch changed recurrent prefix state");
    result["summary"] = generated.metrics();
    require(generated.eos && generated.text.find("9980") != std::string::npos, "Summary did not retain code");
    summary.append_memory(generated.text);
    state = summary.ctx.state();
    generated = summary.summarize(model.tokens("Код объекта Липа: 9980. Погода ясная."));
    require(state == summary.ctx.state() && generated.eos, "Updated summary branch corrupted state");
    result["repeat_summary"] = generated.metrics();
    result["passed"] = true;
    return result;
}
} // namespace ring
