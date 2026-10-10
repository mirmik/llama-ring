#pragma once
#include "llama.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ring {
using json = nlohmann::ordered_json;
using Tokens = std::vector<llama_token>;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start);
void require(bool condition, const std::string & message);
std::string trim(std::string text);
std::string message(const std::string & role, const std::string & text);
struct Model {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> ptr{nullptr, llama_model_free};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> mtp_model{nullptr, llama_model_free};
    const llama_vocab * vocab;
    std::string architecture, name;
    bool hybrid = false, indexed = false;
    ggml_type cache_type;
    int mtp_draft = 0;
    Model(const std::string & path, int gpu_layers, const std::string & cache = "f16", int mtp_draft = 0, const std::string & mtp_path = "");
    Tokens tokens(const std::string & text, bool special = true) const;
    std::string text(const Tokens & ids, bool special = false) const;
    std::string generation_prompt() const;
};
struct Generation {
    std::string text;
    Tokens tokens;
    bool eos = false;
    double prefill_s = 0, decode_s = 0;
    int prefill_tokens = 0;
    std::vector<float> first_logits;
    json metrics() const;
};
// Logical branches: 0 live, 1 answer, 2 candidate P, 3 prefix, 4 shadow.
// RAM rotation maps branches 1..4 onto one physical GPU sequence.
// No persistent branch may be alive while moving positions on sequence 0.
struct Context {
    Model & model;
    std::unique_ptr<llama_context, decltype(&llama_free)> ptr{nullptr, llama_free};
    int batch_size;
    const bool ram_rotation;
    struct Parked { std::vector<uint8_t> bytes; int shared_end = 0; };
    std::array<Parked,5> parked;
    int resident = -1;
    uint64_t exports = 0, imports = 0, export_bytes = 0, import_bytes = 0, peak_ram_bytes = 0;
    double export_s = 0, import_s = 0;
    Context(Model & model, int capacity, int batch_size = 512, int sequences = 4, bool ram_rotation = false);
    // RAM mode: logical 0 is physical 0; all other branches lease physical 1.
    // Shared attention cells belong to main and must not change while borrowed.
    int physical(int seq);
    void park();
    void share_prefix(int seq, int end);
    void rename_branch(int src, int dst);
    json rotation_stats() const;
    llama_memory_t memory() const;
    llama_memory_t attention_memory() const;
    llama_memory_t recurrent_memory() const;
    void remove(int seq, int from = -1, int to = -1);
    void copy(int src, int dst, int from = -1, int to = -1);
    void decode(const Tokens & ids, int position, int seq = 0, bool logits = false);
    std::vector<float> logits();
    Generation generate(const Tokens & prompt, int position, int limit,
                        int source = 0, int prefix_end = -1);
    std::vector<uint8_t> state(int seq = 0);
    std::vector<uint8_t> recurrent_state(int seq = 0);
    std::vector<uint8_t> range_state(int from, int to);
    void restore(const std::vector<uint8_t> & state, int seq = 0);
};
struct Stream {
    Context ctx;
    int prefix_capacity, active_capacity;
    const int prefix_limit, total_capacity;
    bool flexible = false;
    void enable_flexible();
    Tokens prefix;
    std::deque<llama_token> active;
    struct ImageSpan { int begin, count, positions; };
    std::vector<ImageSpan> images;
    // KV occupancy and M-RoPE position extent differ for images.
    int position_at(int token_offset) const;
    int next_position() const { return prefix_capacity + position_at(active.size()); }
    void erase_active(int count);
    bool started = false;
    Stream(Model & model, int prefix_capacity, int active_capacity, int extra_capacity = 0,
           int generation_capacity = 1024, int sequences = 4, bool ram_rotation = false);
    double append_prefix(const Tokens & ids);
    double append(const Tokens & ids);
    Tokens discarded(int count) const;
    double evict(int count);
    double replace_prefix_from(int sequence, const Tokens & ids, int evict_count = 0);
    Generation query(const Tokens & prompt, int limit = 32);
};
struct Summarizer {
    Context ctx;
    Tokens prefix;
    bool cached;
    Summarizer(Model & model, const std::string & initial_memory, int capacity, bool cached);
    Generation summarize(const Tokens & discarded, int limit = 256);
    double append_memory(const std::string & addition);
};
json compare_logits(const std::vector<float> & a, const std::vector<float> & b);
json self_test(Model & model);
json hybrid_self_test(Model & model);
} // namespace ring
