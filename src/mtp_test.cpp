#include "mtp.hpp"
#include "server_session.hpp"
#include "llama-ext.h"
#include <iostream>
#include <unistd.h>
#include <cmath>
using namespace ring;
using Verification = std::vector<std::pair<Tokens,int>>;

// Replay the exact verifier batches without running an MTP head. Separately
// control hidden-state export and rejected suffixes / recurrent rollback.
static std::vector<std::vector<float>> replay_reference(Session & s,const Verification & steps,bool hidden,bool rollback) {
    auto & ctx=s.stream.ctx;ctx.remove(1);ctx.copy(0,1);
    int pos=s.stream.next_position();
    auto suffix=ctx.model.tokens(message("user","Write a short Python function to compute Fibonacci numbers.")+ctx.model.generation_prompt());
    llama_set_embeddings_nextn(ctx.ptr.get(),hidden,false);
    auto b=llama_batch_init(ctx.model.mtp_draft+1,0,1);
    std::vector<std::vector<float>> rows;
    try {
        ctx.decode(suffix,pos,1,true);pos+=suffix.size();rows.push_back(ctx.logits());
        for(const auto & step:steps) {
            b.n_tokens=rollback?step.first.size():step.second;
            for(int i=0;i<b.n_tokens;++i) {b.token[i]=step.first[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=ctx.physical(1);b.logits[i]=true;}
            require(llama_decode(ctx.ptr.get(),b)==0,"Replay decode failed");
            for(int i=0;i<step.second;++i) {const auto * p=llama_get_logits_ith(ctx.ptr.get(),i);rows.emplace_back(p,p+llama_vocab_n_tokens(ctx.model.vocab));}
            pos+=step.second;
            require(llama_memory_seq_rm(ctx.memory(),ctx.physical(1),pos,-1),"Replay rollback failed");
        }
    } catch(...) {llama_batch_free(b);llama_set_embeddings_nextn(ctx.ptr.get(),false,false);throw;}
    llama_batch_free(b);llama_set_embeddings_nextn(ctx.ptr.get(),false,false);ctx.remove(1);return rows;
}

// Independent batching control: only target decode, no MTP head or rollback.
static std::vector<std::vector<float>> batched_reference(Session & s,const Tokens & tokens) {
    auto & ctx=s.stream.ctx;ctx.remove(1);ctx.copy(0,1);
    int pos=s.stream.prefix_capacity+s.stream.active.size();
    auto suffix=ctx.model.tokens(message("user","Write a short Python function to compute Fibonacci numbers.")+ctx.model.generation_prompt());
    ctx.decode(suffix,pos,1,true);pos+=suffix.size();
    std::vector<std::vector<float>> rows{ctx.logits()};
    const int n=ctx.model.mtp_draft+1;
    auto b=llama_batch_init(n,0,1);
    try {
        for(size_t offset=0;offset+1<tokens.size();) {
            b.n_tokens=std::min<size_t>(n,tokens.size()-offset-1);
            for(int i=0;i<b.n_tokens;++i) {b.token[i]=tokens[offset+i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=ctx.physical(1);b.logits[i]=true;}
            require(llama_decode(ctx.ptr.get(),b)==0,"Batch reference decode failed");
            for(int i=0;i<b.n_tokens;++i) {const auto * p=llama_get_logits_ith(ctx.ptr.get(),i);rows.emplace_back(p,p+llama_vocab_n_tokens(ctx.model.vocab));}
            offset+=b.n_tokens;pos+=b.n_tokens;
        }
    } catch(...) {llama_batch_free(b);throw;}
    llama_batch_free(b);ctx.remove(1);return rows;
}
static json compare(const std::vector<std::vector<float>> & actual,const std::vector<std::vector<float>> & ref,const Tokens & tokens) {
    require(actual.size()==ref.size(),"Oracle row count differs");
    double max_abs=0,max_rms=0,max_regret=0,max_kl=0;
    size_t worst_kl_index=0;
    json per_row=json::array();
    for(size_t i=0;i<actual.size();++i) {
        double sum=0,zp=0,zq=0,kl=0;
        const double mp=*std::max_element(ref[i].begin(),ref[i].end());
        const double mq=*std::max_element(actual[i].begin(),actual[i].end());
        for(size_t j=0;j<ref[i].size();++j) {
            double d=actual[i][j]-ref[i][j];require(std::isfinite(d),"Nonfinite oracle logits");
            max_abs=std::max(max_abs,std::abs(d));sum+=d*d;
            zp+=std::exp(ref[i][j]-mp);zq+=std::exp(actual[i][j]-mq);
        }
        for(size_t j=0;j<ref[i].size();++j) {
            double lp=ref[i][j]-mp-std::log(zp),lq=actual[i][j]-mq-std::log(zq);
            kl+=std::exp(lp)*(lp-lq);
        }
        if(kl>max_kl) {max_kl=kl;worst_kl_index=i;}
        per_row.push_back(kl);max_rms=std::max(max_rms,std::sqrt(sum/ref[i].size()));
        max_regret=std::max(max_regret,mp-ref[i][tokens[i]]);
    }
    return {{"max_abs",max_abs},{"max_rms",max_rms},{"max_greedy_regret",max_regret},{"max_kl",max_kl},{"worst_kl_index",worst_kl_index},{"kl_by_token",per_row}};
}

static Tokens answer(Session & s, bool speculative, json & metrics, std::vector<std::vector<float>> * trace=nullptr, const Tokens * forced=nullptr, Verification * steps=nullptr) {
    auto & ctx=s.stream.ctx;
    ctx.remove(1);ctx.copy(0,1);
    int pos=s.stream.prefix_capacity+s.stream.active.size();
    const auto suffix=ctx.model.tokens(message("user","Write a short Python function to compute Fibonacci numbers.")+ctx.model.generation_prompt());
    std::unique_ptr<MtpAnswer> mtp;
    if(speculative) {mtp=std::make_unique<MtpAnswer>(ctx,1024);mtp->prefill(suffix,pos);}
    else ctx.decode(suffix,pos,1,true);
    pos+=suffix.size();
    auto observe=[&](const float * logits){if(trace) trace->emplace_back(logits,logits+llama_vocab_n_tokens(ctx.model.vocab));};
    if(mtp) mtp->observe_logits=observe;
    if(mtp && steps) mtp->observe_verification=[&](const Tokens & batch,int accepted){steps->emplace_back(batch,accepted);};
    common_params_sampling params;params.temp=0;params.seed=123;params.samplers={COMMON_SAMPLER_TYPE_TEMPERATURE};
    common_sampler_ptr sampler(common_sampler_init(ctx.model.ptr.get(),params));
    Tokens result;std::deque<llama_token> queue;
    auto start=Clock::now();
    while(result.size()<(forced?forced->size():96)) {
        llama_token token;
        if(queue.empty()) {observe(llama_get_logits_ith(ctx.ptr.get(),-1));token=forced?forced->at(result.size()):common_sampler_sample(sampler.get(),ctx.ptr.get(),-1);common_sampler_accept(sampler.get(),token,true);}
        else {token=queue.front();queue.pop_front();}
        result.push_back(token);
        if(llama_vocab_is_eog(ctx.model.vocab,token)) break;
        if(result.size()<96) {
            if(mtp) {if(queue.empty()) {auto next=mtp->next(sampler.get(),token,pos,96-result.size());queue.insert(queue.end(),next.begin(),next.end());}}
            else ctx.decode({token},pos++,1,true);
        }
    }
    metrics={{"seconds",seconds(start)},{"tokens",result.size()},{"mtp",mtp?mtp->stats():json(nullptr)}};
    mtp.reset();ctx.remove(1);return result;
}
int main(int argc,char **argv) {
    try {
        require(argc>=2,"Usage: llama-mtp-test MODEL [draft=3] [gpu_layers=99] [cache=q8_0] [mtp_model]");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS","1",1);
        Model model(argv[1],argc>3?std::stoi(argv[3]):99,argc>4?argv[4]:"q8_0",argc>2?std::stoi(argv[2]):3,argc>5?argv[5]:"");
        json cases=json::array();
        auto path=std::filesystem::temp_directory_path()/("mtp-test-"+std::to_string(getpid()));
        for (bool rotate : {false,true}) {
            const int drafts=model.mtp_draft;
            auto make=[&](bool init){return std::make_unique<Session>(model,512,1024,512,32,128,.7,rotate,message("system","You are a helpful assistant."),"mtp-migration",[] {},init,true);};
            model.mtp_draft=0;
            auto before=make(true);
            before->append(model.tokens(message("user","Remember the code 2850.")),[]{});
            const auto state=before->stream.ctx.state();
            save_checkpoint(before->stream.ctx,path,before->snapshot({{"mtp_draft",0}}));
            json normal,fast;
            const auto expected=answer(*before,false,normal);
            before.reset(); model.mtp_draft=drafts;
            Checkpoint saved(path); auto after=make(false); after->restore(saved);
            require(after->stream.ctx.state()==state,"Enabling MTP changed saved target state");
            const auto actual=answer(*after,true,fast);
            require(actual==expected,"MTP migration changed greedy continuation");
            // Also check disabling MTP after an actual speculative request.
            save_checkpoint(after->stream.ctx,path,after->snapshot({{"mtp_draft",drafts}}));
            after.reset(); model.mtp_draft=0;
            Checkpoint again(path); auto plain=make(false); plain->restore(again);
            require(plain->stream.ctx.state()==state,"Disabling MTP changed saved target state");
            require(answer(*plain,false,normal)==expected,"MTP rollback migration changed continuation");
            model.mtp_draft=drafts;
            std::cout<<json({{"test","mtp_checkpoint_migration"},{"rotation",rotate},{"passed",true}}).dump()<<std::endl;
        }
        for(bool rotate:{false,true}) {
            auto make=[&](bool init){return std::make_unique<Session>(model,512,1024,512,32,128,.7,rotate,message("system","You are a helpful assistant."),"mtp-test",[] {},init,true);};
            auto s=make(true);
            s->append(model.tokens(message("user",std::string(100,'a'))),[]{});
            for(const auto stage: {"initial","restored","grown","shrunk"}) {
                if(std::string(stage)=="restored") {
                    auto meta=s->snapshot({{"mtp",model.mtp_draft}});save_checkpoint(s->stream.ctx,path,meta);
                    s.reset();Checkpoint saved(path);s=make(false);s->restore(saved);
                }
                if(std::string(stage)=="grown" || std::string(stage)=="shrunk") {
                    auto ids=model.tokens(message("system",std::string(stage)=="grown"?"You are a helpful assistant. Prior conversation was summarized. The topic is Python functions.":"You are helpful."));
                    s->stream.ctx.remove(2);s->stream.ctx.decode(ids,0,2);
                    s->stream.replace_prefix_from(2,ids,4);
                }
                json normal,fast;
                auto expected=answer(*s,false,normal);
                // The first decode also materializes pending lazy RoPE shifts.
                auto main=s->stream.ctx.state(0);
                std::vector<std::vector<float>> spec_logits,serial_logits;
                Verification steps;
                auto actual=answer(*s,true,fast,&spec_logits,nullptr,&steps);
                json oracle;answer(*s,false,oracle,&serial_logits,&actual);
                const auto numerical=compare(spec_logits,serial_logits,actual);
                const auto control=compare(batched_reference(*s,actual),serial_logits,actual);
                if(std::getenv("VCORTEX_MTP_DIAGNOSTIC")) {
                    json diagnostics;
                    for(bool hidden:{false,true}) for(bool rollback:{false,true}) {
                        const auto replay=replay_reference(*s,steps,hidden,rollback);
                        diagnostics.push_back({{"hidden",hidden},{"rollback",rollback},{"vs_mtp",compare(replay,spec_logits,actual)},
                            {"vs_serial",compare(replay,serial_logits,actual)}});
                    }
                    std::cout<<json({{"stage",stage},{"rotation",rotate},{"steps",steps},{"replays",diagnostics},{"normal_tokens",expected},{"mtp_tokens",actual}}).dump()<<std::endl;
                }
                std::cerr<<json({{"stage",stage},{"rotation",rotate},{"oracle",numerical},{"batch_control",control},{"tokens_equal",actual==expected}}).dump()<<'\n';
                // Compare against independent ordinary batching, not a bitwise
                // serial oracle: quantized GEMM and GEMV use different kernels.
                require(numerical.at("max_kl").get<double>()<0.02 &&
                    numerical.at("max_kl").get<double>()<std::max(0.002,3*control.at("max_kl").get<double>()),"MTP exceeds ordinary batching error envelope");
                require(main==s->stream.ctx.state(0),"Answer altered main state");
                cases.push_back({{"rotation",rotate},{"stage",stage},{"tokens_equal",actual==expected},{"oracle",numerical},{"batch_control",control},{"normal",normal},{"fast",fast}});
                if(std::getenv("VCORTEX_MTP_INITIAL_ONLY")) {
                    std::cout<<json({{"passed",true},{"scope","initial_only"},{"cases",cases}}).dump()<<std::endl;
                    return 0;
                }
            }
        }
        std::filesystem::remove(path);
        std::cout<<json({{"passed",true},{"cases",cases}}).dump(2)<<'\n';
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
