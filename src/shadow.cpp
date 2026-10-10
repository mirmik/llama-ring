#include "shadow.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>

namespace ring {
static const std::string instruction = "Суммаризируй предыдущий текст.";
static const std::string indexed_instruction =
    "Сейчас составь краткое связное резюме всего предыдущего диалога для продолжения разговора. "
    "Выведи только резюме, без подтверждения выполнения.";
static int trigger_tokens(double trigger, int capacity) {
    require(std::isfinite(trigger) && trigger > 0 && trigger < 1, "Invalid trigger");
    int value = int(std::ceil(trigger * capacity));
    require(value < capacity, "Trigger must leave tail capacity");
    return value;
}

Shadow::Shadow(Stream & s, std::string b, double trigger, int q, int limit)
    : stream(s), base(std::move(b)), threshold(trigger_tokens(trigger, s.active_capacity)), quantum(q), max_summary(limit), trigger(trigger) {
    require(s.ctx.ram_rotation || llama_n_seq_max(s.ctx.ptr.get()) >= 5, "Shadow requires five sequence slots");
    require(q > 0 && q <= 512 && limit > 0 && limit <= s.prefix_capacity && limit <= 65536, "Invalid shadow quantum/summary limit");
}
void Shadow::refresh_layout() {
    threshold = stream.flexible ? std::max(1,int(std::ceil(trigger*stream.total_capacity))-stream.prefix_capacity) : trigger_tokens(trigger,stream.active_capacity);
    require(threshold<stream.active_capacity,"Trigger must leave tail capacity");
}
int Shadow::extra_capacity(int prefix, int summary) {
    return std::max(prefix, summary + suffix_capacity);
}
Shadow::~Shadow() {
    // The candidate owns sequence 2, including when unfinished at EOF.
    stream.ctx.remove(2);
    stream.ctx.remove(4);
}
std::string Shadow::status() const {
    switch (phase) {
        case Phase::empty: return "empty";
        case Phase::prefill: return "prefill";
        case Phase::generate: return "generate";
        case Phase::prefix: return "prefix";
        case Phase::ready: return "ready";
        case Phase::failed: return "failed";
    }
    throw std::runtime_error("Invalid phase");
}
bool Shadow::pending() const { return phase != Phase::empty; }
bool Shadow::start_if_due(bool force) {
    if (pending() || stream.active.empty() || (!force && stream.active.size() < size_t(threshold))) return false;
    require(stream.active.size() <= size_t(stream.active_capacity), "Snapshot exceeds capacity");
    auto start = Clock::now();
    cut = stream.active.size();
    cut_position = stream.next_position();
    snapshot_p = stream.prefix;
    snapshot_a = stream.discarded(cut);
    auto & model = stream.ctx.model;
    // The cut may be inside a user/tool/assistant message. Close that message,
    // then switch the cached conversation into an internal summarization task.
    // No archive is rendered/tokenized again: seq4 sees exactly the frozen P+A.
    // Flash-Next also ignores the terse mid-dialogue system request in a fresh
    // upstream context. Use an explicit user task; preserve custom instructions.
    prompt = model.tokens("<|im_end|>\n" + message(model.indexed ? "user" : "system",
        instruction_override.empty() ? (model.indexed ? indexed_instruction : instruction) : instruction_override) + model.generation_prompt());
    require(prompt.size() <= suffix_capacity, "Summary instruction exceeds suffix reserve");
    stream.ctx.remove(2);
    stream.ctx.remove(4);
    // seq_cp shares attention cells and freezes the recurrent tail; subsequent
    // decodes copy recurrent tensors on write. This must happen before A grows.
    stream.ctx.copy(0, 4);
    cursor = 0;
    output.clear(); candidate.clear(); next_logits.clear();
    phase = Phase::prefill;
    current = {{"generation",++generation},{"cut",cut},{"prefix_before_tokens",snapshot_p.size()},
               {"max_summary_tokens",max_summary},
               {"summary_mode","cached_branch"},{"reused_tokens",snapshot_p.size()+snapshot_a.size()},
               {"snapshot_position",cut_position},{"prompt_tokens",prompt.size()},{"idle_steps",0},{"forced_steps",0},
               {"idle_compute_s",0.0},{"forced_compute_s",0.0},{"max_quantum_s",0.0},
               {"prefill_s",0.0},{"decode_s",0.0},{"prefix_prepare_s",0.0},
               {"eos",false},{"steps",json::array()},{"snapshot_s",seconds(start)}};
    return true;
}
void Shadow::advance() {
    auto & model = stream.ctx.model;
    if (phase == Phase::prefill) {
        int end = std::min<int>(cursor + quantum, prompt.size());
        bool last = end == int(prompt.size());
        stream.ctx.decode(Tokens(prompt.begin()+cursor, prompt.begin()+end), cut_position + cursor, 4, last);
        cursor = end;
        if (last) { next_logits = stream.ctx.logits(); phase = Phase::generate; }
    } else if (phase == Phase::generate) {
        const auto token = llama_token(std::max_element(next_logits.begin(), next_logits.end()) - next_logits.begin());
        output.push_back(token);
        if (llama_vocab_is_eog(model.vocab, token)) {
            current["eos"] = true;
            current["summary"] = trim(model.text(output));
            current["summary_tokens"] = output;
            if (const char * path = std::getenv("VCORTEX_SHADOW_TRACE")) {
                std::ofstream trace(path,std::ios::app);
                require(bool(trace),"Cannot open shadow trace");
                trace << json({{"generation",generation},{"prefix",model.text(snapshot_p,true)},
                    {"fragment",model.text(snapshot_a,true)},{"prompt",model.text(prompt,true)},
                    {"summary",current["summary"]}}).dump() << '\n';
            }
            auto text = current["summary"].get<std::string>();
            require(!text.empty(), "Empty shadow summary");
            candidate = prefix_builder ? prefix_builder(text) : model.tokens(message("system", base + (text == "NONE" ? "" : "\n\nПамять:\n" + text)));
            require(candidate.size() <= size_t(stream.prefix_limit), "Shadow prefix overflow; old memory preserved");
            // Release the frozen archive before reserving candidate P cells.
            stream.ctx.remove(4);
            current["candidate_tokens"] = candidate.size();
            cursor = 0; phase = Phase::prefix;
        } else {
            require(output.size() < size_t(max_summary), "Shadow summary did not reach EOS; old memory preserved");
            stream.ctx.decode({token}, cut_position + prompt.size() + output.size() - 1, 4, true);
            next_logits = stream.ctx.logits();
        }
    } else if (phase == Phase::prefix) {
        int end = std::min<int>(cursor + quantum, candidate.size());
        stream.ctx.decode(Tokens(candidate.begin()+cursor, candidate.begin()+end), cursor, 2);
        cursor = end;
        if (end == int(candidate.size())) {
            phase = Phase::ready;
            current["ready_at_active_tokens"] = stream.active.size();
        }
    }
}
bool Shadow::step(bool foreground_waiting) {
    if (foreground_waiting || phase == Phase::empty || phase == Phase::ready) return false;
    return run_step(false);
}
bool Shadow::run_step(bool forced) {
    require(phase != Phase::failed, "Shadow job failed; cannot advance");
    const auto stage = status();
    auto start = Clock::now();
    try { advance(); }
    catch (const std::exception & e) {
        phase = Phase::failed; current["failure"] = e.what();
        stream.ctx.remove(2);
        stream.ctx.remove(4);
        throw;
    }
    const double duration = seconds(start);
    const std::string key = stage == "prefill" ? "prefill_s" : stage == "generate" ? "decode_s" : "prefix_prepare_s";
    current[key] = current[key].get<double>() + duration;
    current["max_quantum_s"] = std::max(current["max_quantum_s"].get<double>(), duration);
    auto count_key = forced ? "forced_steps" : "idle_steps";
    auto time_key = forced ? "forced_compute_s" : "idle_compute_s";
    current[count_key] = current[count_key].get<int>() + 1;
    current[time_key] = current[time_key].get<double>() + duration;
    current["steps"].push_back({{"phase",stage},{"seconds",duration},{"active_tokens",stream.active.size()},{"forced",forced}});
    return true;
}
json Shadow::at_capacity(const std::function<void()> & checkpoint, int required_space) {
    require(stream.active.size() + required_space >= size_t(stream.active_capacity), "Cannot commit before capacity");
    require(pending() && phase != Phase::failed, "Missing valid shadow job");
    require(stream.prefix == snapshot_p && stream.discarded(cut) == snapshot_a, "Stale shadow snapshot");
    current["ready_before_capacity"] = phase == Phase::ready;
    auto start = Clock::now();
    while (phase != Phase::ready) {
        if (checkpoint) checkpoint();
        run_step(true);
    }
    if (checkpoint) checkpoint();
    current["foreground_wait_s"] = seconds(start);
    current["prefix_replace_s"] = stream.replace_prefix_from(2, candidate, stream.flexible?cut:0);
    current["evict_metadata_s"] = stream.flexible?0.0:stream.evict(cut);
    refresh_layout();
    current["active_after"] = stream.active.size();
    current["prefix_after_tokens"] = stream.prefix.size();
    current["commit_s_excluding_lazy_rotation"] = current["prefix_replace_s"].get<double>() + current["evict_metadata_s"].get<double>();
    history.push_back(current);
    phase = Phase::empty;
    return history.back();
}
Tokens record_tokens(Model & model, const json & item, int block_size) {
    if (!block_size) return model.tokens(message("user", item.at("text")) + message("assistant", "Принято."));
    auto ids = model.tokens("<|im_start|>user\n" + item.at("text").get<std::string>() + "\n");
    const auto tail = model.tokens("<|im_end|>\n" + message("assistant", "Принято."));
    const auto filler = model.tokens(" Фоновая заметка: погода ясная, ветер слабый, облака редкие.");
    require(block_size > 0 && ids.size() + tail.size() <= size_t(block_size), "Invalid block_tokens");
    while (ids.size() + tail.size() < size_t(block_size)) {
        size_t n = std::min(filler.size(), block_size - ids.size() - tail.size());
        ids.insert(ids.end(), filler.begin(), filler.begin()+n);
    }
    ids.insert(ids.end(), tail.begin(), tail.end()); return ids;
}
}
