#pragma once
#include "runtime.hpp"
#include <functional>

namespace ring {
// One caller owns GPU dispatch. Foreground requests run before idle_step().
class Shadow {
public:
    enum class Phase { empty, prefill, generate, prefix, ready, failed };
    Stream & stream;
    const std::string base;
    int threshold;
    const int quantum, max_summary;
    const double trigger;
    void refresh_layout();
    Phase phase = Phase::empty;
    Tokens snapshot_p, snapshot_a, prompt, output, candidate;
    int cut = 0, cut_position = 0, cursor = 0, generation = 0;
    std::vector<float> next_logits;
    json current, history = json::array();
    std::function<Tokens(const std::string &)> prefix_builder;
    std::string instruction_override;
    Shadow(Stream & stream, std::string base, double trigger = .7, int quantum = 64, int max_summary = 512);
    ~Shadow();
    // Candidate P and shadow suffix/output never coexist. Foreground reserve is separate.
    static constexpr int suffix_capacity = 1024;
    static int extra_capacity(int prefix, int summary);
    bool start_if_due(bool force = false);
    bool step(bool foreground_waiting);
    json at_capacity(const std::function<void()> & checkpoint = {}, int required_space = 0);
    std::string status() const;
    bool pending() const;
private:
    void advance();
    bool run_step(bool forced);
};
json shadow_self_test(Model & model);
json ram_self_test(Model & model);
Tokens record_tokens(Model & model, const json & item, int block_size);
}
