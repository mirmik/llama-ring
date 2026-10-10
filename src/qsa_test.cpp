#include "runtime.hpp"
#include "checkpoint.hpp"
#include "shadow.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace ring;

int main(int argc, char ** argv) {
    try {
        require(argc >= 2, "Usage: llama-qsa-test MODEL [gpu_layers=99] [cache=f16] [tokens=256] [baseline|memory|summary] [fixture.json]");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "1", 1);
        Model model(argv[1], argc > 2 ? std::stoi(argv[2]) : 99, argc > 3 ? argv[3] : "f16");
        require(model.indexed, "This test requires qwen4exp");
        const int count = argc > 4 ? std::stoi(argv[4]) : 256;
        const std::string mode = argc > 5 ? argv[5] : "baseline";
        require(count >= 64 && count <= 8192 && (mode == "baseline" || mode == "memory" || mode == "summary"), "Invalid test parameters");
        json results = json::array();
        auto emit = [&](json item) { results.push_back(item); std::cout << item.dump() << std::endl; };
        auto compare = [&](const std::string & name, const std::vector<float> & a, const std::vector<float> & b, double tolerance = 1e-3) {
            auto item = compare_logits(a, b); item["test"] = name; emit(item);
            require(item["kl"].get<double>() < tolerance && item["top1_equal"].get<bool>(), "Logits mismatch: " + name);
        };
        if (mode == "summary") {
            require(argc > 6, "Summary diagnostic requires a fixture");
            std::ifstream input(argv[6]); json fixture; input >> fixture;
            auto prefix = model.tokens(message("system",fixture.at("system")));
            Tokens archive;
            for (const auto & block : fixture.at("blocks")) {
                auto ids = record_tokens(model,block,fixture.at("block_tokens"));
                archive.insert(archive.end(),ids.begin(),ids.end());
                if (archive.size() >= size_t(count)) break;
            }
            require(archive.size() >= size_t(count), "Fixture is shorter than the snapshot");
            archive.resize(count);
            Tokens full = prefix; full.insert(full.end(),archive.begin(),archive.end());
            setenv("VCORTEX_QWEN4_RING","1",1);
            Context frozen(model,4096,128,5);
            frozen.decode(prefix,0); frozen.decode(archive,1024);
            const auto before = frozen.state();
            for (int variant = 0; variant < 3; ++variant) {
                const std::string role = variant == 0 ? "system" : "user";
                const std::string instruction = variant < 2 ? "Суммаризируй предыдущий текст." :
                    "Сейчас составь краткое связное резюме всего предыдущего диалога для продолжения разговора. Выведи только резюме, без подтверждения выполнения.";
                auto suffix = model.tokens("<|im_end|>\n"+message(role,instruction)+model.generation_prompt());
                auto cached = frozen.generate(suffix,1024+archive.size(),512);
                require(frozen.state() == before,"Summary changed the live state");
                // Fresh, contiguous upstream context, without sequence copies or ring pooling.
                setenv("VCORTEX_QWEN4_RING","0",1);
                Context fresh(model,4096,128,5);
                full.insert(full.end(),suffix.begin(),suffix.end());
                Generation control;
                auto started = Clock::now(); fresh.decode(full,0,0,true);
                control.first_logits = fresh.logits(); control.prefill_tokens = full.size();
                control.prefill_s = seconds(started); started = Clock::now();
                auto current = control.first_logits;
                for (int i = 0; i < 512; ++i) {
                    auto token = llama_token(std::max_element(current.begin(),current.end())-current.begin());
                    control.tokens.push_back(token);
                    if (llama_vocab_is_eog(model.vocab,token)) { control.eos = true; break; }
                    fresh.decode({token},full.size()+i,0,true); current = fresh.logits();
                }
                control.decode_s = seconds(started); control.text = trim(model.text(control.tokens));
                full.resize(prefix.size()+archive.size());
                setenv("VCORTEX_QWEN4_RING","1",1);
                emit({{"test","summary_role"},{"role",role},{"instruction",instruction},{"cached",cached.metrics()},{"fresh",control.metrics()},
                      {"first_logits",compare_logits(control.first_logits,cached.first_logits)}});
            }
            emit({{"passed",true},{"mode",mode}}); return 0;
        }
        Tokens prompt = model.tokens(message("system", "Answer briefly. The secret code of Linden is 2850.") + "<|im_start|>user\n");
        const auto filler = model.tokens("The weather is clear. A bird flies over the garden. ");
        const auto question = model.tokens("\nWhat is the secret code of Linden? Reply with the number only.<|im_end|>\n" + model.generation_prompt());
        while (prompt.size() + question.size() < size_t(count)) prompt.push_back(filler[prompt.size() % filler.size()]);
        prompt.insert(prompt.end(), question.begin(), question.end());
        const int capacity = std::max(2048, int(prompt.size()) * 3 + 512);
        // By default compare with upstream; optionally use the old ring path
        // to isolate cache parity beyond the 2048-token sparse-selection cutoff.
        const bool old_ring = std::getenv("VCORTEX_QSA_REFERENCE_RING") != nullptr;
        setenv("VCORTEX_QWEN4_RING", old_ring ? "1" : "0", 1);
        if (old_ring) setenv("VCORTEX_QSA_CACHE", "0", 1);
        Context stock(model, capacity, 128, 5);
        auto start = Clock::now();
        stock.decode(prompt, 0, 0, true);
        const auto ordinary = stock.logits();
        emit({{"test", "ordinary_prefill"}, {"tokens", prompt.size()}, {"seconds", seconds(start)}});
        setenv("VCORTEX_QWEN4_RING", "1", 1);
        if (old_ring) setenv("VCORTEX_QSA_CACHE", "1", 1);
        Context ring(model, capacity, 128, 5);
        start = Clock::now();
        ring.decode(prompt, 0, 0, true);
        emit({{"test", "ring_prefill"}, {"tokens", prompt.size()}, {"seconds", seconds(start)}});
        compare("ring_vs_upstream", ordinary, ring.logits());
        const auto base = ring.state();
        Tokens generated;
        auto logits = ordinary;
        start = Clock::now();
        for (int i = 0; i < 16; ++i) {
            const auto token = llama_token(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (llama_vocab_is_eog(model.vocab, token)) break;
            generated.push_back(token);
            stock.decode({token}, prompt.size() + i, 0, true);
            ring.decode({token}, prompt.size() + i, 0, true);
            logits = stock.logits();
            compare("decode_" + std::to_string(i), logits, ring.logits());
        }
        emit({{"test", "answer"}, {"text", model.text(generated)}, {"tokens", generated}, {"paired_decode_seconds", seconds(start)}});
        require(!generated.empty(), "Empty ordinary generation");
        if (mode == "baseline") { emit({{"passed", true}, {"mode", mode}}); return 0; }

        Context partitioned(model, capacity, 31, 5);
        partitioned.decode(prompt, 0, 0, true);
        compare("batch31_vs_batch128", ordinary, partitioned.logits());
        const auto probe = model.tokens(" 2850");
        ring.restore(base);
        ring.decode(probe, prompt.size(), 0, true);
        const auto expected = ring.logits();
        for (int shift : {1, 2, 3, 4, 37}) {
            ring.restore(base);
            llama_memory_attention_seq_add(ring.memory(), 0, -1, -1, shift);
            llama_memory_seq_add(ring.recurrent_memory(), 0, -1, -1, shift);
            ring.decode(probe, prompt.size() + shift, 0, true);
            compare("uniform_shift_" + std::to_string(shift), expected, ring.logits());
        }
        ring.restore(base);
        const auto disk = std::filesystem::temp_directory_path() / ("llama-qsa-" + std::to_string(getpid()) + ".bin");
        save_checkpoint(ring, disk, {{"test", "qsa"}});
        ring.remove(0);
        { Checkpoint checkpoint(disk); checkpoint.restore(ring); }
        ring.decode(probe, prompt.size(), 0, true);
        compare("disk_roundtrip", expected, ring.logits());
        std::filesystem::remove(disk);

        // Different prefix lengths make the same shared raw cells belong to different QSA pools.
        // A branch decode must never overwrite the main branch's raw/derived state.
        for (int cut : {1, 2, 3, 4}) {
            ring.restore(base); ring.remove(1);
            ring.copy(0, 1);
            require(llama_memory_attention_seq_rm(ring.memory(), 1, 0, cut), "Branch trim failed");
            auto main = ring.state();
            ring.decode(probe, prompt.size(), 1, true);
            const auto branch_logits = ring.logits();
            require(main == ring.state(), "Branch pooling changed main state");
            Context isolated(model, capacity, 128, 5);
            isolated.restore(base);
            require(llama_memory_attention_seq_rm(isolated.memory(), 0, 0, cut), "Control trim failed");
            isolated.decode(probe, prompt.size(), 0, true);
            compare("partial_branch_" + std::to_string(cut), isolated.logits(), branch_logits);
            ring.remove(1);
        }

        {
            Context rotated(model, capacity, 128, 5, true);
            rotated.restore(base);
            rotated.copy(0, 4);
            const auto main = rotated.state();
            rotated.decode(probe, prompt.size(), 4, true);
            compare("ram_branch_first", expected, rotated.logits());
            const auto branch = rotated.state(4);
            rotated.park();
            require(rotated.state(4) == branch, "RAM roundtrip changed private branch bytes");
            rotated.park();
            rotated.decode(probe, prompt.size() + probe.size(), 4, true);
            const auto resumed = rotated.logits();
            ring.restore(base);
            ring.decode(probe, prompt.size());
            ring.decode(probe, prompt.size() + probe.size(), 0, true);
            compare("ram_branch_resumed", ring.logits(), resumed);
            require(rotated.state() == main, "RAM branch modified main state");
            require(rotated.rotation_stats()["imports"].get<int>() >= 2, "RAM test did not import a branch");
            emit({{"test", "ram_roundtrip"}, {"rotation", rotated.rotation_stats()}});
        }

        const auto prefix = model.tokens(message("system", "Remember: Linden code is 2850."));
        for (bool rotate : {false, true}) {
            Stream stream(model, 128, std::max(256, count), 512, 128, 5, rotate);
            stream.append_prefix(prefix);
            stream.enable_flexible();
            for (int cycle = 0; cycle < 5; ++cycle) {
                Tokens active(stream.active_capacity - stream.active.size(), filler.front());
                stream.append(active);
                auto recurrence = stream.ctx.recurrent_state();
                Tokens next = prefix;
                next.insert(next.end(), cycle % 2 ? 3 : 8, filler.front());
                stream.ctx.decode(next, 0, 2);
                const int cut = cycle == 4 ? stream.active.size() : 33 + cycle;
                const int old_position = llama_memory_seq_pos_max(stream.ctx.recurrent_memory(), 0);
                stream.replace_prefix_from(2, next, cut);
                const int new_position = llama_memory_seq_pos_max(stream.ctx.recurrent_memory(), 0);
                llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, old_position - new_position);
                require(recurrence == stream.ctx.recurrent_state(), "Compaction changed recurrent tensors");
                llama_memory_seq_add(stream.ctx.recurrent_memory(), 0, -1, -1, new_position - old_position);
                stream.append(probe);
                const auto saved = stream.ctx.state();
                auto answer = stream.query(question, 8);
                require(saved == stream.ctx.state(), "Query changed live state");
                require(answer.eos && answer.text == "2850", "Prepared memory fact was not read");
                emit({{"test", "compaction"}, {"ram_rotation", rotate}, {"cycle", cycle}, {"active", stream.active.size()},
                      {"bytes", saved.size()}, {"answer", answer.metrics()}, {"rotation", stream.ctx.rotation_stats()}});
            }
        }
        emit({{"passed", true}, {"mode", mode}});
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
