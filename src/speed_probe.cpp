#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef RING_SPEED_RUNTIME
#include "runtime.hpp"
using ring::Model;
using ring::Context;
#else
struct Model {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> ptr{nullptr, llama_model_free};
    const llama_vocab * vocab;
    Model(const std::string & path, int layers) {
        ggml_backend_load_all();
        bool gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto t = ggml_backend_dev_type(ggml_backend_dev_get(i));
            gpu |= t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        if (!gpu) throw std::runtime_error("GPU unavailable");
        auto p = llama_model_default_params(); p.n_gpu_layers = layers;
        ptr.reset(llama_model_load_from_file(path.c_str(), p));
        if (!ptr) throw std::runtime_error("Model load failed");
        vocab = llama_model_get_vocab(ptr.get());
    }
};
// Public upstream API, with the same allocation and synchronization as Context.
struct Context {
    Model & model;
    std::unique_ptr<llama_context, decltype(&llama_free)> ptr{nullptr, llama_free};
    Context(Model & m, int capacity) : model(m) {
        auto p = llama_context_default_params();
        p.n_ctx = capacity; p.n_batch = p.n_ubatch = 512;
        p.n_seq_max = 4; p.kv_unified = true;
        p.n_threads = p.n_threads_batch = 8;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        p.type_k = p.type_v = GGML_TYPE_F16; p.no_perf = false;
        ptr.reset(llama_init_from_model(m.ptr.get(), p));
        if (!ptr) throw std::runtime_error("Context failed");
    }
    void decode(const std::vector<llama_token> & ids, int pos, int seq = 0, bool logits = false) {
        auto b = llama_batch_init(512, 0, 1);
        for (size_t start = 0; start < ids.size(); start += 512) {
            b.n_tokens = std::min<size_t>(512, ids.size() - start);
            for (int i = 0; i < b.n_tokens; ++i) {
                b.token[i] = ids[start+i]; b.pos[i] = pos + start + i;
                b.n_seq_id[i] = 1; b.seq_id[i][0] = seq;
                b.logits[i] = logits && start+i+1 == ids.size();
            }
            if (llama_decode(ptr.get(), b)) {
                llama_batch_free(b); throw std::runtime_error("Decode failed");
            }
        }
        llama_synchronize(ptr.get()); llama_batch_free(b);
    }
    std::vector<float> logits() {
        auto data = llama_get_logits_ith(ptr.get(), -1);
        if (!data) throw std::runtime_error("No logits");
        std::vector<float> result(data, data + llama_vocab_n_tokens(model.vocab));
        if (!std::all_of(result.begin(), result.end(), [](float x){return std::isfinite(x);}))
            throw std::runtime_error("Nonfinite logits");
        return result;
    }
};
#endif
using json = nlohmann::ordered_json;
using Tokens = std::vector<llama_token>;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static Tokens tokens(Model & m, int count) {
    std::mt19937 rng(31);
    Tokens ids(count);
    for (auto & t : ids) t = rng() % llama_vocab_n_tokens(m.vocab);
    return ids;
}
static std::vector<uint8_t> save(Context & c) {
    llama_synchronize(c.ptr.get());
    std::vector<uint8_t> s(llama_state_seq_get_size(c.ptr.get(), 0));
    if (llama_state_seq_get_data(c.ptr.get(), s.data(), s.size(), 0) != s.size())
        throw std::runtime_error("Save failed");
    return s;
}
static void reset(Context & c, const std::vector<uint8_t> & s) {
    llama_memory_clear(llama_get_memory(c.ptr.get()), false);
    if (!s.empty() && llama_state_seq_set_data(c.ptr.get(), s.data(), s.size(), 0) != s.size())
        throw std::runtime_error("Restore failed");
    llama_synchronize(c.ptr.get());
}
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: speed-probe MODEL OUTPUT.json");
#ifdef RING_SPEED_RUNTIME
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "1", 1);
        const char * engine = "ring";
#else
        unsetenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS"); unsetenv("VCORTEX_QWEN35_TEXT_SHIFT");
        const char * engine = "stock";
#endif
        Model model(argv[1], 99);
        json out = {{"engine",engine},{"model",argv[1]},{"seed",31},{"repeats",3},
                    {"warmups_per_case",1},{"n_ctx",13312},{"n_seq_max",4},
                    {"n_batch",512},{"n_ubatch",512},{"threads",8},{"cases",json::array()}};
        auto write = [&] { std::ofstream f(argv[2]); f << out.dump(2) << '\n'; if(!f) throw std::runtime_error("Output failed"); };
        {
            Context ctx(model, 13312);
            for (int depth : {0, 7235, 8259}) {
                reset(ctx, {});
                if (depth) ctx.decode(tokens(model, depth), 0);
                auto state = depth ? save(ctx) : std::vector<uint8_t>{};
                for (const std::string mode : {"pp", "tg"}) {
                    if (depth == 7235 && mode == "tg") continue;
                    int count = mode == "pp" ? 1024 : 128;
                    auto input = tokens(model, count);
                    json item = {{"mode",mode},{"depth",depth},{"tokens",count},{"samples",json::array()}};
                    for (int rep = -1; rep < 3; ++rep) {
                        reset(ctx, state);
                        Tokens generated;
                        auto start = Clock::now();
                        if (mode == "pp") ctx.decode(input, depth);
                        else {
                            auto token = input[0];
                            for (int i = 0; i < count; ++i) {
                                ctx.decode({token}, depth+i, 0, true);
                                auto logits = ctx.logits();
                                token = std::max_element(logits.begin(), logits.end()) - logits.begin();
                                generated.push_back(token);
                            }
                        }
                        double duration = elapsed(start);
                        if (rep >= 0) item["samples"].push_back({{"seconds",duration},{"tokens_s",count/duration},{"generated",generated}});
                    }
                    out["cases"].push_back(item); write();
                    std::cerr << engine << ' ' << mode << " depth=" << depth << " complete\n";
                }
            }
        }
#ifdef RING_SPEED_RUNTIME
        {
            ring::Stream stream(model, 4096, 8192);
            stream.append_prefix(tokens(model, 67));
            stream.append(tokens(model, 8192));
            json item = {{"mode","evict_append"},{"depth",7235},{"tokens",1024},{"samples",json::array()}};
            auto block = tokens(model, 1024);
            for (int rep = -1; rep < 3; ++rep) {
                auto start = Clock::now();
                double meta = stream.evict(1024);
                double append = stream.append(block);
                double duration = elapsed(start);
                if (rep >= 0) item["samples"].push_back({{"seconds",duration},{"tokens_s",1024/duration},
                    {"metadata_s",meta},{"append_including_rotation_s",append}});
            }
            out["cases"].push_back(item); write();
        }
#endif
        out["completed"] = true; write();
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
