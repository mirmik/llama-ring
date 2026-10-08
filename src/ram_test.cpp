#include "shadow.hpp"
#include <algorithm>
#include <iostream>

namespace ring {
json ram_self_test(Model & model) {
    json result={{"architecture",model.architecture},{"comparisons",json::array()}};
    const std::string base="Запоминай факты. Отвечай кратко.";
    const auto initial=model.tokens(message("system",base+" Код Липы: 9980."));
    const auto filler=model.tokens(" Фоновая заметка: погода ясная.");
    auto fill=[&](int n){Tokens t; while(int(t.size())<n) t.push_back(filler[t.size()%filler.size()]); return t;};
    // Same allocation size eliminates context-capacity confounds; different
    // physical cell placement is expected, so test distributions as well as bytes.
    Stream ram(model,512,512,1536,1536,5,true);
    Stream ref(model,512,512,0,1536,5);
    for(auto * s:{&ram,&ref}) { s->append_prefix(initial); s->append(fill(360)); s->ctx.copy(0,4); }
    const auto main_before=ram.ctx.state();
    bool rejected=false;
    try { ram.evict(1); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected && main_before==ram.ctx.state(),"Borrowed-history mutation was not rejected");
    const auto shadow_before=ram.ctx.state(4);
    ram.ctx.park();
    require(main_before==ram.ctx.state(),"Parking changed main");
    require(ram.ctx.state(4)==shadow_before,"Shared snapshot roundtrip changed bytes");
    ram.ctx.park();
    const uint64_t snapshot_bytes=ram.ctx.parked[4].bytes.size();
    const auto compare=[&](const std::vector<float> & a,const std::vector<float> & b,const std::string & name){
        auto metric=compare_logits(a,b); metric["name"]=name; result["comparisons"].push_back(metric);
        std::cerr << "RAM numeric comparison: " << metric.dump() << std::endl;
        require(metric["kl"].get<double>()<1e-2,"RAM continuation differs: "+name+" "+metric.dump());
    };
    int pos=512+360;
    for(int i=0;i<6;++i) {
        auto token=fill(1);
        const auto live=ram.ctx.state();
        ram.ctx.decode(token,pos,4,true);auto logits=ram.ctx.logits();
        ref.ctx.decode(token,pos++,4,true);
        compare(logits,ref.ctx.logits(),"shadow-"+std::to_string(i));
        require(live==ram.ctx.state(),"Shadow changed main recurrence/KV");
        const auto frozen=ram.ctx.state(4);
        ram.ctx.park();
        require(frozen==ram.ctx.state(4),"Private KV/recurrent roundtrip changed bytes");
        ram.append(fill(8));ref.append(fill(8));
        require(frozen==ram.ctx.state(4),"Main append changed frozen branch");
        const auto question=model.tokens(message("user","Какой код Липы?")+model.generation_prompt());
        auto a=ram.query(question,1), b=ref.query(question,1);
        compare(a.first_logits,b.first_logits,"answer-"+std::to_string(i));
        require(frozen==ram.ctx.state(4),"Answer evicted/damaged shadow");
    }
    ram.ctx.remove(4);ref.ctx.remove(4);
    // Independent prefix state can itself be parked, restored, and extended.
    const auto prefix_more=model.tokens(" Еще один факт.");
    ram.append_prefix(prefix_more);ref.append_prefix(prefix_more);
    compare(ram.query(fill(1),1).first_logits,ref.query(fill(1),1).first_logits,"prefix-append");
    for(auto * s:{&ram,&ref}) {
        s->ctx.decode(initial,0,2);
        s->append(fill(512-s->active.size())); // park the prepared prefix
        s->replace_prefix_from(2,initial);
        s->evict(360);
        s->append(fill(8)); // performs lazy K rotation
    }
    compare(ram.query(fill(1),1).first_logits,ref.query(fill(1),1).first_logits,"replace-rebase");
    result["snapshot_private_bytes"]=snapshot_bytes;
    result["full_snapshot_bytes"]=shadow_before.size();
    require(snapshot_bytes<shadow_before.size(),"Shared KV was copied to RAM");
    result["rotation"]=ram.ctx.rotation_stats();
    // Exercise the actual scheduler: forced path, cancellation, ready commit,
    // failed/truncated summary, and destruction of an unfinished parked branch.
    for(int scenario=0;scenario<3;++scenario) {
        Stream stream(model,512,512,Shadow::extra_capacity(512,256),512,5,true);
        stream.append_prefix(initial);
        const auto before_prefix=stream.prefix;
        Shadow shadow(stream,base,.7,8,scenario==2?1:256);
        stream.append(record_tokens(model,{{"text","Код Кедра: 2247."}},shadow.threshold));
        shadow.start_if_due();stream.append(fill(512-stream.active.size()));
        const auto before=stream.ctx.state();
        if(scenario==2) {
            bool failed=false;
            try{while(shadow.phase!=Shadow::Phase::ready)shadow.step(false);}catch(const std::runtime_error&){failed=true;}
            require(failed && shadow.phase==Shadow::Phase::failed && before==stream.ctx.state() && before_prefix==stream.prefix,
                    "Failed summary changed main");
            require(stream.ctx.parked[4].bytes.empty() && stream.ctx.resident!=4,"Failed branch leaked");
        } else {
            bool canceled=false;int checks=0;
            try{shadow.at_capacity([&]{if(++checks==3)throw std::runtime_error("cancel");});}
            catch(const std::runtime_error& e){canceled=std::string(e.what())=="cancel";}
            require(canceled && before==stream.ctx.state(),"Cancel changed main");
            if(scenario==0) while(shadow.phase!=Shadow::Phase::ready)shadow.step(false);
            auto event=shadow.at_capacity();
            require(stream.active.size()==512-size_t(shadow.cut),"Wrong retained tail");
            stream.query(fill(1),1);
            result["commits"].push_back(event);
        }
    }
    result["passed"]=true;return result;
}
}
