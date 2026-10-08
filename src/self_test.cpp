#include "runtime.hpp"
#include <cstdlib>
#include <iostream>

namespace ring {
json self_test(Model & model) {
    json result;
    const auto p = model.tokens(message("system", "Remember facts. Answer with the four-digit code only."));
    const auto delta = model.tokens(message("system", "Memory: the code of Linden is 2850."));
    const auto a = model.tokens(message("user", std::string(240, 'x') + "\nThe weather is mild.") + message("assistant", "OK."));
    const auto q = model.tokens(message("user", "What is the code of Linden?") + "<|im_start|>assistant\n");
    Stream stream(model, 512, 512);
    stream.append_prefix(p);
    setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "0", 1);
    bool rejected = false;
    const auto pristine = stream.ctx.state();
    try { stream.append(a); } catch (const std::runtime_error &) { rejected = true; }
    setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "1", 1);
    require(rejected && stream.ctx.state() == pristine, "Strict mode must reject gaps without mutation");
    result["strict_mode_rejects_gap_atomically"] = true;
    stream.append(a);
    auto old_p = stream.ctx.range_state(0, p.size());
    auto old_a = stream.ctx.range_state(512, -1);
    stream.append_prefix(delta);
    require(old_p == stream.ctx.range_state(0, p.size()), "Old P changed");
    require(old_a == stream.ctx.range_state(512, -1), "A changed during prefix append");
    result["old_prefix_and_active_kv_byte_identical"] = true;
    Context reference(model, 2048);
    reference.decode(p, 0);
    reference.decode(delta, p.size());
    auto expected = reference.generate(q, p.size() + delta.size(), 16);
    auto actual = stream.ctx.generate(q, stream.prefix.size(), 16, 0, stream.prefix.size());
    result["prefix_append_vs_isolated"] = compare_logits(actual.first_logits, expected.first_logits);
    require(result["prefix_append_vs_isolated"]["kl"].get<double>() < 1e-4, "Prefix attended to future A");
    auto all = stream.ctx.state();
    auto answer = stream.query(q);
    require(all == stream.ctx.state(), "Query branch mutated persistent cache");
    require(answer.text == "2850" && answer.eos, "New prefix is not visible to new A queries");
    result["query_branch_preserves_cache"] = true;
    result["new_prefix_answer"] = answer.metrics();
    bool overflow = false;
    try { stream.append_prefix(Tokens(513, p.front())); } catch (const std::runtime_error &) { overflow = true; }
    require(overflow && all == stream.ctx.state(), "Prefix overflow mutated cache");
    stream.append_prefix({});
    require(all == stream.ctx.state(), "Empty append mutated cache");
    result["overflow_and_empty_append_atomic"] = true;
    const int drop = 16;
    require(a.size() > size_t(drop), "Test A too short");
    reference.restore(all);
    reference.remove(0, 512, 512 + drop);
    // Equivalent coordinates: keep A absolute and move P forward by drop.
    llama_memory_seq_add(reference.memory(), 0, 0, 512, drop);
    stream.evict(drop);
    auto shifted = stream.query(q, 1);
    auto absolute = reference.generate(q, 512 + a.size(), 1);
    result["rebase_vs_absolute"] = compare_logits(shifted.first_logits, absolute.first_logits);
    require(result["rebase_vs_absolute"]["kl"].get<double>() < 1e-4 &&
            result["rebase_vs_absolute"]["top1_equal"].get<bool>(), "Position shift is inconsistent");
    Stream loop(model, 256, 128);
    loop.append_prefix(p);
    const auto fill = Tokens(32, model.tokens(" weather").front());
    size_t full_size = 0;
    for (int i = 0; i < 20; ++i) {
        if (loop.active.size() == 128) loop.evict(32);
        loop.append(fill);
        if (i == 3) full_size = loop.ctx.state().size();
        if (i >= 3) require(loop.ctx.state().size() == full_size, "Live KV state grows across evictions");
    }
    result["bounded_cache_after_16_evictions"] = {{"serialized_bytes", full_size}, {"active_tokens", loop.active.size()}};
    Summarizer cached(model, "Known: code of Cedar is 3143.", 2048, true);
    Summarizer fresh(model, "Known: code of Cedar is 3143.", 2048, false);
    auto block = model.tokens("Код объекта Липа: 2850. Погода ясная.");
    auto before = cached.ctx.state();
    auto c = cached.summarize(block, 96), f = fresh.summarize(block, 96);
    require(before == cached.ctx.state(), "Summary branch mutated persistent P");
    result["summary_cached_vs_fresh"] = compare_logits(c.first_logits, f.first_logits);
    result["summary_cached_vs_fresh"]["same_tokens"] = c.tokens == f.tokens;
    result["summary_cached_vs_fresh"]["cached"] = c.metrics();
    result["summary_cached_vs_fresh"]["fresh"] = f.metrics();
    // Quantized kernels can change with prefill batch shape. Separately compare
    // an independently recomputed prefix with the same prefill partition.
    require(c.eos && f.eos, "Summary did not finish");
    fresh.ctx.decode(fresh.prefix, 0);
    fresh.cached = true;
    auto matched = fresh.summarize(block, 96);
    result["summary_same_partition"] = compare_logits(c.first_logits, matched.first_logits);
    result["summary_same_partition"]["same_tokens"] = c.tokens == matched.tokens;
    std::cerr << result.dump(2) << std::endl;
    require(result["summary_same_partition"]["kl"].get<double>() < 1e-4, "Summary cache reuse mismatch");
    cached.append_memory(c.text); fresh.append_memory(c.text);
    before = cached.ctx.state();
    c = cached.summarize(block, 96); f = fresh.summarize(block, 96);
    require(before == cached.ctx.state(), "Second summary mutated persistent P");
    result["summary_after_append"] = compare_logits(c.first_logits, f.first_logits);
    result["summary_after_append"]["same_tokens"] = c.tokens == f.tokens;
    require(result["summary_after_append"]["kl"].get<double>() < 1e-4 && c.eos && f.eos,
            "Summary prefix append is inconsistent");
    result["passed"] = true;
    return result;
}
} // namespace ring
