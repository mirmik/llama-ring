#include "runtime.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

using namespace ring;
int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "llama-ring --model MODEL.gguf --self-test 1 --output checks.json\n"
                    "llama-ring --model MODEL.gguf --input stream.json --output results.json\n"
                    "  [--prefix 4096] [--active 8192] [--evict 1024]\n"
                    "  [--summarizer cached|fresh|none] [--gpu-layers 99] [--cache-type f16|q8_0]\n";
                return 0;
            }
            require(i + 1 < argc && key.rfind("--", 0) == 0, "Expected --option value");
            args[key] = argv[++i];
        }
        auto value = [&](const std::string & k, std::string fallback) { return args.count(k) ? args.at(k) : fallback; };
        require(args.count("--model") && args.count("--output"), "--model and --output required (see --help)");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "1", 1);
        Model model(args.at("--model"), std::stoi(value("--gpu-layers", "99")), value("--cache-type", "f16"));
        if (args.count("--self-test")) {
            auto result = model.hybrid ? hybrid_self_test(model) : self_test(model);
            std::ofstream(args.at("--output")) << result.dump(2) << '\n';
            std::cout << result.dump(2) << '\n';
            return 0;
        }
        require(args.count("--input"), "--input required");
        std::ifstream input(args.at("--input"));
        require(bool(input), "Cannot open input");
        json data = json::parse(input);
        const int p = std::stoi(value("--prefix", "4096"));
        const int a = std::stoi(value("--active", "8192"));
        const int eviction = std::stoi(value("--evict", "1024"));
        const auto mode = value("--summarizer", "cached");
        require(p > 0 && a > 0 && eviction > 0 && eviction <= a, "Invalid capacity/eviction");
        require(mode == "cached" || mode == "fresh" || mode == "none", "Unknown summarizer mode");
        auto start = Clock::now();
        Stream stream(model, p, a);
        const std::string base = data.at("system");
        stream.append_prefix(model.tokens(message("system", base)));
        std::unique_ptr<Summarizer> summarizer;
        if (mode != "none") summarizer = std::make_unique<Summarizer>(model, base, 2*p + eviction + 2048, mode == "cached");
        json result = {{"model", args.at("--model")}, {"architecture", model.architecture}, {"model_name", model.name},
                       {"hybrid_policy", model.hybrid ? "preserve_stream_recurrence_separate_prefix" : "none"},
                       {"mode", mode}, {"prefix_capacity", p},
                       {"active_capacity", a}, {"evict_tokens", eviction}, {"events", json::array()},
                       {"probes", json::array()}, {"blocks", json::array()}};
        auto save = [&]() {
            result["elapsed_s"] = seconds(start);
            result["prefix_tokens"] = stream.prefix.size();
            result["active_tokens"] = stream.active.size();
            result["prefix_text"] = model.text(stream.prefix, true);
            std::ofstream out(args.at("--output"));
            require(bool(out), "Cannot write output");
            out << result.dump(2) << '\n';
        };
        int index = 0, total_evicted = 0;
        for (const auto & item : data.at("blocks")) {
            Tokens ids;
            const int block_size = data.value("block_tokens", 0);
            if (block_size) {
                ids = model.tokens("<|im_start|>user\n" + item.at("text").get<std::string>() + "\n");
                const auto tail = model.tokens("<|im_end|>\n" + message("assistant", "Принято."));
                const auto filler = model.tokens(" Фоновая заметка: погода ясная, ветер слабый, облака редкие.");
                require(ids.size() + tail.size() <= size_t(block_size), "Record exceeds block_tokens");
                while (ids.size() + tail.size() < size_t(block_size)) {
                    size_t count = std::min(filler.size(), block_size - ids.size() - tail.size());
                    ids.insert(ids.end(), filler.begin(), filler.begin() + count);
                }
                ids.insert(ids.end(), tail.begin(), tail.end());
            } else {
                ids = model.tokens(message("user", item.at("text")) + message("assistant", "Принято."));
            }
            double append_s = 0;
            for (size_t offset = 0; offset < ids.size();) {
                if (stream.active.size() == size_t(a)) {
                    const auto evicted = stream.discarded(eviction);
                    json event = {{"before_block", index}, {"evicted_start", total_evicted}, {"count", eviction},
                                  {"discarded_text", model.text(evicted)}, {"prefix_before", stream.prefix.size()}};
                    std::string delta;
                    if (summarizer) {
                        auto generated = summarizer->summarize(evicted);
                        event["summary"] = generated.metrics();
                        if (!generated.eos) {
                            result["failure"] = "Summary did not reach EOS; stream not advanced";
                            result["pending_event"] = event;
                            save();
                            throw std::runtime_error(result["failure"].get<std::string>());
                        }
                        delta = generated.text == "NONE" ? "" : generated.text;
                    }
                    auto addition = delta.empty() ? Tokens{} : model.tokens(message("system", "Дополнение памяти:\n" + delta));
                    if (stream.prefix.size() + addition.size() > size_t(p)) {
                        result["failure"] = "Prefix full; stream not advanced";
                        result["pending_event"] = event;
                        save();
                        throw std::runtime_error(result["failure"].get<std::string>());
                    }
                    // Append P first: pending rotation must not contaminate its timing.
                    event["prefix_append_s"] = stream.append_prefix(addition);
                    event["summary_memory_append_s"] = summarizer ? summarizer->append_memory(delta) : 0;
                    event["evict_metadata_s"] = stream.evict(eviction);
                    event["prefix_after"] = stream.prefix.size();
                    total_evicted += eviction;
                    result["events"].push_back(event);
                    save();
                }
                const size_t count = std::min(ids.size() - offset, a - stream.active.size());
                append_s += stream.append(Tokens(ids.begin() + offset, ids.begin() + offset + count));
                offset += count;
            }
            result["blocks"].push_back({{"index", index}, {"tokens", ids.size()}, {"append_s_including_pending_shift", append_s}});
            if (item.contains("questions")) for (const auto & question : item.at("questions")) {
                const auto prompt = model.tokens(message("user", question.at("text")) + model.generation_prompt());
                auto answer = stream.query(prompt, 32);
                auto probe = answer.metrics();
                probe["block"] = index;
                probe["question"] = question;
                if (question.contains("expected")) probe["correct"] = answer.text == question.at("expected").get<std::string>();
                result["probes"].push_back(probe);
            }
            ++index;
            save();
            std::cout << "block=" << index << " P=" << stream.prefix.size() << " A=" << stream.active.size()
                      << " evictions=" << result["events"].size() << std::endl;
        }
        result["completed"] = true;
        save();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
