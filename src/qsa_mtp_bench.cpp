#include "mtp.hpp"
#include "llama-ext.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>

using namespace ring;

// Same weights, prompts, target rollback capacity and sampler in every arm.
// Separate contexts also include the cost of initially constructing each cache.
int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "Usage: llama-qsa-mtp-bench MODEL MTP_MODEL [prompt_tokens=16000] [output_tokens=256] [oracle]");
        const int count = argc > 3 ? std::stoi(argv[3]) : 16000;
        const int limit = argc > 4 ? std::stoi(argv[4]) : 256;
        const bool diagnostic = argc > 5 && std::string(argv[5]) == "diagnostic";
        const bool oracle = diagnostic || (argc > 5 && std::string(argv[5]) == "oracle");
        require(count >= 0 && count <= 131072 && limit > 0 && limit <= 2048, "Invalid benchmark size");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS", "1", 1);
        Model model(argv[1], 99, "q8_0", 3, argv[2]);
        Tokens history = model.tokens(message("system", "You are a helpful assistant."));
        const auto filler = model.tokens("The garden contains a stone path, a wooden bench, and several apple trees. ");
        while (history.size() < size_t(count)) history.insert(history.end(), filler.begin(), filler.end());
        const std::vector<std::pair<std::string, std::string>> questions = {
            {"code", "Write Python code for an LRU cache with get, put, capacity eviction, and a few usage examples. Explain how it works."},
            {"russian", "Подробно объясни, как устроен кеш процессора, зачем нужны его уровни и что происходит при промахе. Приведи несколько примеров."},
        };
        for (const auto & arm : {"baseline", "qsa", "qsa_mtp"}) {
            if (oracle && std::string(arm) != "qsa_mtp") continue;
            setenv("VCORTEX_QSA_CACHE", std::string(arm) == "baseline" ? "0" : "1", 1);
            Context ctx(model, std::max(4096, int(history.size()) + limit + 1024), 256, 2, true);
            auto start = Clock::now();
            ctx.decode(history, 0);
            const double prefill = seconds(start);
            for (const auto & question : questions) {
                ctx.remove(1);
                const auto main_state = oracle ? ctx.state() : std::vector<uint8_t>();
                auto reset_oracle = [&] {
                    ctx.remove(1); ctx.remove(0);
                    llama_memory_clear(ctx.memory(), true);
                    ctx.restore(main_state);
                };
                if (oracle) reset_oracle();
                ctx.copy(0, 1);
                const auto suffix = model.tokens(message("user", question.second) + model.generation_prompt());
                int position = history.size();
                start = Clock::now();
                std::unique_ptr<MtpAnswer> mtp;
                if (std::string(arm) == "qsa_mtp") {
                    mtp = std::make_unique<MtpAnswer>(ctx, limit + 1024);
                    mtp->prefill(suffix, position);
                } else ctx.decode(suffix, position, 1, true);
                position += suffix.size();
                common_params_sampling params;
                params.temp = 0; params.seed = 123; params.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
                common_sampler_ptr sampler(common_sampler_init(model.ptr.get(), params));
                Tokens tokens;
                std::deque<llama_token> queued;
                std::vector<std::vector<float>> trace;
                std::vector<std::pair<Tokens,int>> steps;
                auto observe = [&](const float * logits) {
                    trace.emplace_back(logits, logits + llama_vocab_n_tokens(model.vocab));
                };
                if (oracle && mtp) mtp->observe_logits = observe;
                if (diagnostic && mtp) mtp->observe_verification = [&](const Tokens & ids,int kept) { steps.emplace_back(ids,kept); };
                const double first = seconds(start);
                start = Clock::now();
                while (tokens.size() < size_t(limit)) {
                    llama_token token;
                    if (queued.empty()) {
                        if (oracle) observe(llama_get_logits_ith(ctx.ptr.get(), -1));
                        token = common_sampler_sample(sampler.get(), ctx.ptr.get(), -1);
                        common_sampler_accept(sampler.get(), token, true);
                    } else { token = queued.front(); queued.pop_front(); }
                    tokens.push_back(token);
                    if (llama_vocab_is_eog(model.vocab, token) || tokens.size() == size_t(limit)) break;
                    if (mtp) {
                        if (queued.empty()) {
                            auto next = mtp->next(sampler.get(), token, position, limit - tokens.size());
                            queued.insert(queued.end(), next.begin(), next.end());
                        }
                    } else ctx.decode({token}, position++, 1, true);
                }
                const double elapsed = seconds(start);
                const auto stats = mtp ? mtp->stats() : json(nullptr);
                mtp.reset();
                json accuracy;
                if (oracle) {
                    require(ctx.state() == main_state,"MTP changed the main branch state");
                    require(trace.size() == tokens.size(), "Oracle row count mismatch");
                    reset_oracle(); ctx.copy(0, 1);
                    ctx.decode(suffix, history.size(), 1, true);
                    int pos = history.size() + suffix.size();
                    double max_kl = 0, max_regret = 0;
                    std::vector<std::vector<float>> serial;
                    auto kl = [](const float * p, const float * q, size_t n) {
                        const double mp = *std::max_element(p,p+n), mq = *std::max_element(q,q+n);
                        double zp=0,zq=0,result=0;
                        for (size_t j=0;j<n;++j) {
                            require(std::isfinite(p[j]) && std::isfinite(q[j]),"Nonfinite oracle logits");
                            zp+=std::exp(p[j]-mp); zq+=std::exp(q[j]-mq);
                        }
                        for (size_t j=0;j<n;++j) {
                            const double lp=p[j]-mp-std::log(zp),lq=q[j]-mq-std::log(zq);
                            result+=std::exp(lp)*(lp-lq);
                        }
                        return result;
                    };
                    for (size_t i=0;i<tokens.size();++i) {
                        const auto * ref=llama_get_logits_ith(ctx.ptr.get(),-1);
                        const size_t n=trace[i].size();
                        max_kl=std::max(max_kl,kl(ref,trace[i].data(),n));
                        max_regret=std::max(max_regret,double(*std::max_element(ref,ref+n)-ref[tokens[i]]));
                        serial.emplace_back(ref,ref+n);
                        if (i+1<tokens.size()) ctx.decode({tokens[i]},pos++,1,true);
                    }
                    reset_oracle(); ctx.copy(0,1);
                    ctx.decode(suffix,history.size(),1,true);
                    pos=history.size()+suffix.size();
                    double control=0;
                    auto batch=llama_batch_init(4,0,1);
                    for (size_t i=0;i+1<tokens.size();) {
                        batch.n_tokens=std::min<size_t>(4,tokens.size()-i-1);
                        for(int j=0;j<batch.n_tokens;++j) {
                            batch.token[j]=tokens[i+j];batch.pos[j]=pos+j;
                            batch.n_seq_id[j]=1;batch.seq_id[j][0]=ctx.physical(1);batch.logits[j]=true;
                        }
                        require(llama_decode(ctx.ptr.get(),batch)==0,"Oracle batch decode failed");
                        for(int j=0;j<batch.n_tokens;++j) control=std::max(control,
                            kl(serial[i+j+1].data(),llama_get_logits_ith(ctx.ptr.get(),j),serial[i+j+1].size()));
                        pos+=batch.n_tokens;i+=batch.n_tokens;
                    }
                    llama_batch_free(batch);
                    accuracy={{"max_kl",max_kl},{"max_greedy_regret",max_regret},{"batch_control_max_kl",control},
                        {"passed",max_kl<0.02 && max_kl<std::max(0.002,3*control)}};
                    if (diagnostic) {
                        accuracy["replays"]=json::array();
                        for (bool hidden : {false,true}) for (bool rollback : {false,true}) {
                            reset_oracle();ctx.copy(0,1);
                            llama_set_embeddings_nextn(ctx.ptr.get(),hidden,false);
                            ctx.decode(suffix,history.size(),1,true);
                            pos=history.size()+suffix.size();
                            double spec_kl=kl(trace[0].data(),llama_get_logits_ith(ctx.ptr.get(),-1),trace[0].size());
                            double serial_kl=kl(serial[0].data(),llama_get_logits_ith(ctx.ptr.get(),-1),serial[0].size());
                            size_t row=1,worst=0;
                            auto replay=llama_batch_init(4,0,1);
                            for(const auto & step:steps) {
                                replay.n_tokens=rollback?step.first.size():step.second;
                                for(int j=0;j<replay.n_tokens;++j) {
                                    replay.token[j]=step.first[j];replay.pos[j]=pos+j;
                                    replay.n_seq_id[j]=1;replay.seq_id[j][0]=ctx.physical(1);replay.logits[j]=true;
                                }
                                require(llama_decode(ctx.ptr.get(),replay)==0,"Replay failed");
                                for(int j=0;j<step.second;++j,++row) {
                                    const auto * ref=llama_get_logits_ith(ctx.ptr.get(),j);
                                    const double d=kl(trace.at(row).data(),ref,trace[row].size());
                                    if(d>spec_kl) {spec_kl=d;worst=row;}
                                    serial_kl=std::max(serial_kl,kl(serial[row].data(),ref,serial[row].size()));
                                }
                                pos+=step.second;
                                require(llama_memory_seq_rm(ctx.memory(),ctx.physical(1),pos,-1),"Replay rollback failed");
                            }
                            llama_batch_free(replay);
                            llama_set_embeddings_nextn(ctx.ptr.get(),false,false);
                            require(row==trace.size(),"Replay row count mismatch");
                            accuracy["replays"].push_back({{"hidden",hidden},{"rollback",rollback},{"vs_spec_kl",spec_kl},
                                {"vs_serial_kl",serial_kl},{"worst_spec_row",worst}});
                        }
                    }
                }
                std::cout << json({{"arm",arm},{"question",question.first},{"prompt_tokens",history.size()+suffix.size()},
                    {"history_prefill_s",prefill},{"suffix_ttft_s",first},{"decode_s",elapsed},
                    {"tokens",tokens},{"text",model.text(tokens)},{"decode_tps",(tokens.size()-1)/elapsed},
                    {"mtp",stats},{"oracle",accuracy}}).dump() << std::endl;
                require(!oracle || diagnostic || accuracy.at("passed").get<bool>(),"MTP exceeds batching error envelope");
            }
        }
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
