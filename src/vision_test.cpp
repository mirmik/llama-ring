#include "vision.hpp"
#include "mtmd-helper.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
using namespace ring;
int main(int argc,char **argv) {
    try {
        require(argc>=4,"llama-vision-test MODEL MMPROJ MESSAGES_JSON [gpu_layers] [f16|q8_0] [--shifts-only]");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS","1",1);
        Model model(argv[1],argc>4?std::stoi(argv[4]):99,argc>5?argv[5]:"q8_0");
        Vision vision(model,argv[2],argc<=4 || std::stoi(argv[4])>0,256);
        std::ifstream in(argv[3]);auto input=vision.input(json::parse(in));
        const auto text=message("user",std::string(mtmd_get_marker(vision.ctx.get()))+"What is in the image?");
        auto prompt=vision.tokenize(text,input);
        json results=json::array();
        bool shifts_ok=true;
        // A uniform translation must preserve this already-computed state.
        // Fresh replay also changes prefill rounding/routing, so measure its
        // own translation drift separately instead of attributing it to K-shift.
        for(int delta:{0,37,-37}) {
            Context shifted(model,8192),reference(model,8192);
            llama_pos old_end=0,new_end=0;const int origin=100;
            require(mtmd_helper_eval_chunks(vision.ctx.get(),shifted.ptr.get(),prompt.chunks.get(),origin,0,512,false,&old_end)==0,"Reference image prefill failed");
            auto snapshot=shifted.state();
            auto suffix=model.tokens(model.generation_prompt());
            shifted.decode(suffix,old_end,0,true);
            const auto original=shifted.logits();
            shifted.restore(snapshot);
            llama_memory_attention_seq_add(shifted.memory(),0,origin,-1,delta);
            llama_memory_seq_add(shifted.recurrent_memory(),0,origin,-1,delta);
            require(mtmd_helper_eval_chunks(vision.ctx.get(),reference.ptr.get(),prompt.chunks.get(),origin+delta,0,512,false,&new_end)==0,"Shift control prefill failed");
            shifted.decode(suffix,old_end+delta,0,true);reference.decode(suffix,new_end,0,true);
            auto metric=compare_logits(reference.logits(),shifted.logits());
            auto cached=compare_logits(original,shifted.logits());
            results.push_back({{"delta",delta},{"metrics",metric},
                {"cached_translation",cached},
                {"fresh_translation",compare_logits(original,reference.logits())}});
            std::cerr << results.back().dump() << '\n';
            shifts_ok &= cached.at("kl").get<double>()<0.02;
            if(delta==0) shifts_ok &= metric.at("kl").get<double>()<1e-6;
        }
        // Atomic image accounting, RAM branches, and exact disk round trip.
        for(bool ram:{false,true}) {
            if(argc>6 && std::string(argv[6])=="--shifts-only") break;
            auto check=[]{};
            Session s(model,1024,3072,64,64,256,.7,ram,message("system","Be concise."),"vision-test",check,true,true);
            vision.append(s,prompt,0,check);s.ledger=prompt.tokens;s.image_ledger=input.hashes;
            require(!s.stream.images.empty(),"Image span missing");
            const int end=s.stream.next_position();
            require(end<int(s.stream.prefix.size()+s.stream.active.size()),"Image positions were flattened");
            auto q=model.tokens(model.generation_prompt());
            auto a=s.stream.query(q,8);
            auto metadata=s.snapshot(json::object());
            auto path=std::filesystem::temp_directory_path()/("llama-ring-vision-test-"+std::to_string(getpid())+".checkpoint");
            save_checkpoint(s.stream.ctx,path,metadata);
            {
                Session restored(model,1024,3072,64,64,256,.7,ram,s.immutable,s.signature,check,false,true);
                Checkpoint cp(path);restored.restore(cp);
                require(restored.stream.next_position()==end && restored.image_ledger==input.hashes,"Image metadata restore failed");
                auto b=restored.stream.query(q,8);
                auto metric=compare_logits(a.first_logits,b.first_logits);
                require(metric.at("kl").get<double>()<0.01,"Image restore KL exceeds tolerance");
                results.push_back({{"ram",ram},{"restore",metric},{"answer",b.text}});
            }
            std::filesystem::remove(path);
            // Leave less room than a whole image: commit the old shadow early,
            // then admit the complete grid without cutting it at the context edge.
            auto note=model.tokens(" The weather is clear today.");
            Tokens padding;
            const size_t fill=size_t(s.stream.active_capacity)-s.stream.active.size()-32;
            while(padding.size()<fill) padding.insert(padding.end(),note.begin(),note.end());
            padding.resize(fill);s.append(padding,check);
            const auto commits=s.commits;
            vision.append(s,prompt,0,check);
            require(s.commits>commits && !s.stream.images.empty(),"Image did not force an atomic early commit");
            const auto & image=s.stream.images.back();
            bool rejected=false;
            try { (void)s.stream.position_at(image.begin+1); }
            catch(const std::exception &) { rejected=true; }
            require(rejected,"Image accepted a split boundary");
            results.push_back({{"ram",ram},{"atomic_early_commit",true},{"images",s.stream.images.size()}});
        }
        std::cout<<json({{"completed",shifts_ok},{"shift_reference","original_cached_state"},{"results",results}}).dump(2)<<'\n';
        require(shifts_ok,"Image shift KL exceeds tolerance");
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
