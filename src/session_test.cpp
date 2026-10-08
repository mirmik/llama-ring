#include "server_session.hpp"
#include <algorithm>
#include <iostream>
#include <unistd.h>
int main(int argc,char ** argv) {
    try {
        require(argc>=2,"Usage: llama-session-test MODEL [gpu_layers] [mtp_draft]");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS","1",1);
        Model model(argv[1],argc>2?std::stoi(argv[2]):99,"q8_0",argc>3?std::stoi(argv[3]):0);
        const auto path=std::filesystem::temp_directory_path()/("vcortex-session-"+std::to_string(getpid())+".bin");
        json result=json::array();
        for(bool shared:{false,true}) for(bool rotate:{false,true}) {
            auto make=[&](bool initialize){return std::make_unique<Session>(model,2048,512,64,32,512,.7,rotate,message("system","You are a helpful assistant."),"test",[] {},initialize,shared);};
            auto s=make(true);
            // Fill to the trigger with a real conversation; freeze a shadow branch.
            auto tokens=model.tokens(message("user","Remember: the project name is CEDAR. Summarize briefly."));
            Tokens input;
            while(input.size()<size_t(s->shadow.threshold)) input.insert(input.end(),tokens.begin(),tokens.end());input.resize(s->shadow.threshold);
            s->append(input,[]{});s->ledger=input;s->context_id="test-context";
            for(int stage=0;stage<4;++stage) {
                if(stage==1) while(s->shadow.phase==Shadow::Phase::prefill) s->shadow.step(false);
                if(stage==2) {
                    // A valid independently evaluated candidate, without relying on model EOS.
                    s->stream.ctx.remove(4);
                    s->shadow.candidate=model.tokens(message("system",std::string(200,' ' )+"Memory: CEDAR."));
                    while(s->shadow.candidate.size()<96) s->shadow.candidate.push_back(tokens.front());
                    s->shadow.cursor=0;s->shadow.phase=Shadow::Phase::prefix;
                    s->shadow.step(false);
                }
                if(stage==3) while(s->shadow.phase==Shadow::Phase::prefix) s->shadow.step(false);
                const auto phase=s->shadow.status();
                auto meta=s->snapshot({{"test",1}});
                save_checkpoint(s->stream.ctx,path,meta);
                auto main=s->stream.ctx.state(0);
                auto prefix=s->stream.ctx.state(3);
                auto expected=s->stream.query(model.tokens(message("user","What is the project?")+model.generation_prompt()),8);
                s.reset();
                Checkpoint saved(path);s=make(false);s->restore(saved);
                require(s->shadow.status()==phase,"Shadow phase lost");
                require(s->stream.ctx.state(0)==main,"Main state differs");
                require(s->stream.ctx.state(3)==prefix,"Prefix state differs");
                auto actual=s->stream.query(model.tokens(message("user","What is the project?")+model.generation_prompt()),8);
                if(actual.tokens!=expected.tokens) std::cerr<<json({{"shared",shared},{"rotation",rotate},{"phase",phase},
                    {"expected",expected.tokens},{"actual",actual.tokens},{"first_logits",compare_logits(expected.first_logits,actual.first_logits)}}).dump()<<'\n';
                require(actual.tokens==expected.tokens,"Continuation differs after restore");
                result.push_back({{"shared",shared},{"rotation",rotate},{"phase",phase},{"bytes",saved.size},{"tokens_equal",true}});
            }
            // Resume and commit a ready prefix after restoration.
            Tokens tail(s->stream.active_capacity-s->stream.active.size(),tokens.front());s->append(tail,[]{});
            s->append({tokens.front()},[]{});require(s->commits==1,"Restored shadow did not commit");
            if(shared) {
                require(s->stream.prefix_capacity==int(s->stream.prefix.size()),"Gap before tail");
                require(s->stream.active_capacity+s->stream.prefix_capacity==s->stream.total_capacity,"Lost shared budget");
                require(llama_memory_seq_pos_max(s->stream.ctx.attention_memory(),0)==int(s->stream.prefix.size()+s->stream.active.size()-1),"Tail position mismatch after growth");
                // Next cycle shrinks the prefix. Check the whole retained tail is
                // still addressable after changing both eviction and base offset.
                s->append(Tokens(s->stream.active_capacity-s->stream.active.size(),tokens.front()),[]{});
                s->stream.ctx.remove(4);s->stream.ctx.remove(2);
                s->shadow.candidate=model.tokens(message("system","CEDAR."));
                s->shadow.phase=Shadow::Phase::prefix;s->shadow.cursor=0;
                while(s->shadow.phase!=Shadow::Phase::ready) s->shadow.step(false);
                s->append({tokens.front()},[]{});
                require(s->commits==2 && s->stream.prefix_capacity==int(s->stream.prefix.size()),"Shrinking prefix failed");
                require(llama_memory_seq_pos_max(s->stream.ctx.attention_memory(),0)==int(s->stream.prefix.size()+s->stream.active.size()-1),"Tail position mismatch after shrink");
                auto answer=s->stream.query(model.tokens(model.generation_prompt()),4);
                require(!answer.tokens.empty(),"No continuation after shrink");
            }
        }
        std::filesystem::remove(path);
        std::cout<<json({{"passed",true},{"cases",result}}).dump(2)<<'\n';return 0;
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
