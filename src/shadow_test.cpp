#include "shadow.hpp"
#include <algorithm>
#include <iostream>

namespace ring {
static std::vector<uint8_t> attention_range(Context & ctx, int start, int end) {
    ctx.remove(1);
    llama_memory_seq_cp(ctx.attention_memory(),0,1,start,end);
    auto bytes=ctx.state(1);ctx.remove(1);return bytes;
}
static std::vector<uint8_t> normalized_state(Context & ctx,int source) {
    // Serialization includes sequence IDs; compare both branches through slot 1.
    ctx.remove(1);ctx.copy(source,1);
    auto bytes=ctx.state(1);ctx.remove(1);return bytes;
}
json shadow_self_test(Model & model) {
    json result={{"architecture",model.architecture}};
    const std::string base="Запоминай факты. На вопрос о коде отвечай только четырьмя цифрами. Если неизвестно, ответь UNKNOWN.";
    const auto initial=model.tokens(message("system",base+"\nПамять: Код объекта Липа: 9980."));
    const auto question=[&](std::string name){return model.tokens(message("user","Какой код у объекта "+name+"? Ответь только четырьмя цифрами.")+model.generation_prompt());};
    {
        Stream stream(model,512,512,Shadow::extra_capacity(512,256),1024,5);
        stream.append_prefix(initial);
        // Keep a prompt continuation for the restore oracle even with a short instruction.
        Shadow shadow(stream,base,.7,8,256);
        stream.append(record_tokens(model,{{"text","Код объекта Кедр: 2247."}},shadow.threshold));
        auto before=stream.ctx.state();
        auto old_p=model.hybrid?stream.ctx.state(3):attention_range(stream.ctx,0,512);
        require(shadow.start_if_due() && stream.ctx.state()==before,"Snapshot mutated stream");
        require(!shadow.step(true) && shadow.cursor==0,"Foreground did not block background");
        shadow.step(false);
        const auto frozen=normalized_state(stream.ctx,4);
        const auto snapshot=shadow.prompt;
        const int cursor=shadow.cursor;
        stream.append(record_tokens(model,{{"text","Код объекта Ива: 6666."}},80));
        auto probe=stream.query(question("Ива"));
        require(probe.eos && probe.text=="6666","Foreground query failed during background job");
        require(shadow.cursor==cursor && shadow.prompt==snapshot,"Foreground advanced/mutated shadow job");
        require(normalized_state(stream.ctx,4)==frozen,"Foreground modified frozen summary state");
        before=stream.ctx.state();
        std::vector<uint8_t> full(llama_state_get_size(stream.ctx.ptr.get()));
        require(llama_state_get_data(stream.ctx.ptr.get(),full.data(),full.size())==full.size(),
                "Full control snapshot failed");
        while(shadow.phase==Shadow::Phase::prefill) shadow.step(false);
        const auto first_logits=shadow.next_logits;
        stream.query(question("Ива"));
        require(first_logits==shadow.next_logits,"Foreground overwrote saved summary logits");
        // Continue the frozen state in a different context as a numeric oracle.
        // This also checks that the branch has the same sparse P/A positions.
        {
            Context isolated(model,llama_n_ctx(stream.ctx.ptr.get()),512,5);
            isolated.restore(frozen);
            require(normalized_state(isolated,0)==frozen,"Isolated restore changed snapshot bytes");
            for(int offset=cursor;offset<int(shadow.prompt.size());offset+=shadow.quantum) {
                int end=std::min<int>(offset+shadow.quantum,shadow.prompt.size());
                isolated.decode(Tokens(shadow.prompt.begin()+offset,shadow.prompt.begin()+end),
                    stream.prefix_capacity+shadow.cut+offset,0,end==int(shadow.prompt.size()));
            }
            result["cached_vs_isolated_logits"]=compare_logits(first_logits,isolated.logits());
            std::cerr<<"Cached vs isolated: "<<result["cached_vs_isolated_logits"].dump()<<'\n';
            // Packed sequence restore changes physical attention layout and
            // masked reduction width. Record that diagnostic separately from
            // the identical-layout control (the strict numeric assertion).
            require(llama_state_set_data(isolated.ptr.get(),full.data(),full.size())==full.size(),
                    "Full control restore failed");
            for(int offset=cursor;offset<int(shadow.prompt.size());offset+=shadow.quantum) {
                int end=std::min<int>(offset+shadow.quantum,shadow.prompt.size());
                isolated.decode(Tokens(shadow.prompt.begin()+offset,shadow.prompt.begin()+end),
                    stream.prefix_capacity+shadow.cut+offset,4,end==int(shadow.prompt.size()));
            }
            result["cached_vs_full_control_logits"]=compare_logits(first_logits,isolated.logits());
            std::cerr<<"Cached vs full control: "<<result["cached_vs_full_control_logits"].dump()<<'\n';
            require(result["cached_vs_full_control_logits"]["kl"].get<double>()<1e-5,
                    "Cached branch differs from full snapshot continuation");
        }
        full.clear();full.shrink_to_fit();
        while(shadow.phase!=Shadow::Phase::ready) shadow.step(false);
        require(stream.ctx.state()==before,"Background modified live KV or recurrence");
        require((model.hybrid?stream.ctx.state(3):attention_range(stream.ctx,0,512))==old_p,"Background modified old P");
        const auto ready_state=stream.ctx.state(2);
        probe=stream.query(question("Ива"));
        require(probe.eos && probe.text=="6666" && stream.ctx.state()==before && stream.ctx.state(2)==ready_state,
                "Foreground query damaged ready candidate or live stream");
        require(shadow.current["summary"].get<std::string>().find("9980")!=std::string::npos &&
                shadow.current["summary"].get<std::string>().find("2247")!=std::string::npos &&
                shadow.current["summary"].get<std::string>().find("6666")==std::string::npos,"Summary snapshot coverage incorrect");
        bool rejected=false;
        try{shadow.at_capacity();}catch(const std::runtime_error&){rejected=true;}
        require(rejected && stream.ctx.state()==before,"Early commit was not rejected atomically");
        stream.append(Tokens(512-stream.active.size(),model.tokens(" погода").front()));
        Tokens tail(stream.active.begin()+shadow.cut,stream.active.end());
        const auto candidate_state=stream.ctx.state(2);
        const auto candidate_normalized=normalized_state(stream.ctx,2);
        Context reference(model,3072);
        reference.restore(stream.ctx.state()); reference.restore(candidate_state,2);
        require(llama_memory_seq_rm(reference.attention_memory(),0,0,512+shadow.cut),"Reference drop failed");
        llama_memory_seq_cp(reference.attention_memory(),2,0,0,shadow.candidate.size());
        reference.remove(2);
        llama_memory_seq_add(reference.attention_memory(),0,0,512,shadow.cut);
        const auto recurrence=model.hybrid?stream.ctx.recurrent_state():std::vector<uint8_t>{};
        auto event=shadow.at_capacity();
        require(event["ready_before_capacity"].get<bool>() && event["forced_steps"]==0,"Ready path forced work");
        require(Tokens(stream.active.begin(),stream.active.end())==tail,"Commit changed tail tokens");
        if(model.hybrid) {
            require(candidate_normalized==normalized_state(stream.ctx,3),"New prefix state not installed");
            llama_memory_seq_add(stream.ctx.recurrent_memory(),0,-1,-1,shadow.cut);
            require(recurrence==stream.ctx.recurrent_state(),"Commit changed main recurrent tensors");
            llama_memory_seq_add(stream.ctx.recurrent_memory(),0,-1,-1,-shadow.cut);
        }
        auto rebased=stream.query(question("Ива"));
        auto absolute=reference.generate(question("Ива"),1024,32);
        result["rebase_vs_absolute"]=compare_logits(rebased.first_logits,absolute.first_logits);
        std::cerr << "Rebase check: " << result["rebase_vs_absolute"].dump()
                  << ", rebased=" << rebased.text << ", absolute=" << absolute.text << '\n';
        require(result["rebase_vs_absolute"]["kl"].get<double>()<1e-3 && rebased.text==absolute.text,"Tail rebase mismatch");
        for(const auto & pair:std::vector<std::pair<std::string,std::string>>{{"Липа","9980"},{"Кедр","2247"},{"Ива","6666"}}) {
            auto answer=stream.query(question(pair.first));
            result["answers"][pair.first]=answer.metrics();
            require(answer.eos && answer.text==pair.second,"Old P/snapshot/tail fact lost after swap");
        }
        result["ready_commit"]=event;
        result["foreground_priority_and_snapshot_isolation"]=true;
        result["cached_branch_frozen_during_foreground"]=true;
        require(llama_memory_seq_pos_max(stream.ctx.memory(),4)<0,"Summary branch survived commit");
        result["prepared_prefix_waits_for_capacity"]=true;
        result["tail_and_main_recurrence_preserved"]=true;
    }
    {
        Stream stream(model,512,512,Shadow::extra_capacity(512,256),1024,5);stream.append_prefix(initial);
        Shadow shadow(stream,base,.7,64,256);
        stream.append(record_tokens(model,{{"text","Код объекта Кедр: 2247."}},shadow.threshold));
        shadow.start_if_due();
        const auto frozen=normalized_state(stream.ctx,4);
        stream.append(Tokens(512-stream.active.size(),model.tokens(" погода").front()));
        require(normalized_state(stream.ctx,4)==frozen,"Main-first decode modified snapshot recurrence");
        result["main_first_snapshot_isolation"]=true;
        auto cancel_before=stream.ctx.state();int checkpoints=0;bool canceled=false;
        try { shadow.at_capacity([&] { if(++checkpoints==3) throw std::runtime_error("test cancel"); }); }
        catch(const std::runtime_error & e) { canceled=std::string(e.what())=="test cancel"; }
        require(canceled && checkpoints==3 && cancel_before==stream.ctx.state() && stream.prefix==initial,
                "Cancel during forced summary changed live state");
        result["forced_summary_cancel_preserves_live_memory"]=true;
        const auto event=shadow.at_capacity();
        require(!event["ready_before_capacity"].get<bool>() && event["forced_steps"].get<int>()>0 && event["idle_steps"]==0,"No-idle fallback incorrect");
        result["forced_commit"]=event;
        auto before=stream.ctx.state();bool rejected=false;
        try{stream.replace_prefix_from(2,Tokens(513,1));}catch(const std::runtime_error&){rejected=true;}
        require(rejected && before==stream.ctx.state(),"Invalid prefix replacement mutated state");
        // Finish the previous eviction's lazy K rotation before comparing bytes.
        stream.query(question("Липа"),1);
        auto active=attention_range(stream.ctx,512,-1);
        auto replacement=model.tokens(message("system",base+"\nКод объекта Дуб: 5728."));
        stream.ctx.decode(replacement,0,2);
        stream.replace_prefix_from(2,replacement);
        require(active==attention_range(stream.ctx,512,-1),"Prefix replacement changed A KV before rebase");
        result["replace_preserves_active_kv"]=true;
    }
    {
        Stream stream(model,512,512,Shadow::extra_capacity(512,256),1024,5);stream.append_prefix(initial);
        Shadow shadow(stream,base,.7,64,1);
        stream.append(record_tokens(model,{{"text","Код объекта Кедр: 2247."}},shadow.threshold));
        shadow.start_if_due();auto before=stream.ctx.state();bool failed=false;
        try{while(shadow.phase!=Shadow::Phase::ready)shadow.step(false);}catch(const std::runtime_error&){failed=true;}
        require(failed && before==stream.ctx.state() && stream.prefix==initial,"Truncated summary destroyed memory");
        require(llama_memory_seq_pos_max(stream.ctx.memory(),4)<0,"Failed summary leaked branch");
        result["summary_failure_preserves_live_memory"]=true;
    }
    result["passed"]=true;
    std::cerr<<"Shadow state tests passed\n";return result;
}
}
