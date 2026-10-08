#include "shadow.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>

using namespace ring;
int main(int argc, char ** argv) {
    try {
        std::map<std::string,std::string> args;
        for (int i=1; i<argc; ++i) {
            std::string key=argv[i];
            if (key=="--help") {
                std::cout << "llama-shadow --model MODEL --input fixture.json --output result.json\n"
                    "  [--prefix 4096] [--active 8192] [--trigger 0.7] [--quantum 64]\n"
                    "  [--max-summary 512] [--gpu-layers 99] [--cache-type f16|q8_0] [--self-test 1] [--ram-self-test 1] [--ram-rotation 0|1]\n"
                    "  [--summary-probe cases.json] (fixed-prompt diagnostic cases)\n"
                    "JSON blocks may specify delay_before_ms (idle before request arrival).\n";
                return 0;
            }
            require(i+1<argc && key.rfind("--",0)==0,"Expected --option value");
            args[key]=argv[++i];
        }
        auto get=[&](std::string k,std::string d){return args.count(k)?args.at(k):d;};
        require(args.count("--model") && args.count("--output"),"--model and --output required");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS","1",1);
        Model model(args.at("--model"),std::stoi(get("--gpu-layers","99")),get("--cache-type","f16"));
        if(args.count("--summary-probe")) {
            std::ifstream in(args.at("--summary-probe"));require(bool(in),"Cannot open summary cases");
            json cases=json::parse(in),results=json::array();
            Context ctx(model,16384);
            for(const auto & item:cases) {
                auto prompt=model.tokens(message("system",item.at("instruction"))+"<|im_start|>user\n");
                auto data=model.tokens(item.at("data"),false);
                prompt.insert(prompt.end(),data.begin(),data.end());
                auto end=model.tokens("<|im_end|>\n"+model.generation_prompt());
                prompt.insert(prompt.end(),end.begin(),end.end());
                auto answer=ctx.generate(prompt,0,512);
                results.push_back({{"name",item.at("name")},{"answer",answer.metrics()}});
                std::ofstream out(args.at("--output"));out<<results.dump(2)<<'\n';
                std::cerr<<"Summary case "<<item.at("name")<<": "<<answer.text<<'\n';
                ctx.remove(0);ctx.remove(1);
            }
            return 0;
        }
        if(args.count("--self-test") || args.count("--ram-self-test")) {
            auto result=args.count("--ram-self-test") ? ram_self_test(model) : shadow_self_test(model);
            std::ofstream out(args.at("--output"));
            require(bool(out),"Cannot write checks");out<<result.dump(2)<<'\n';
            return 0;
        }
        require(args.count("--input"),"--input required");
        std::ifstream in(args.at("--input"));require(bool(in),"Cannot open input");
        json data=json::parse(in);
        int p=std::stoi(get("--prefix","4096")), a=std::stoi(get("--active","8192"));
        int quantum=std::stoi(get("--quantum","64")), limit=std::stoi(get("--max-summary","512"));
        require(p>0 && p<=65536 && a>1 && a<=131072,"Invalid capacities");
        auto begin=Clock::now();
        Stream stream(model,p,a,Shadow::extra_capacity(p,limit),1024,5,get("--ram-rotation","0")=="1");
        std::string base=data.at("system");
        stream.append_prefix(model.tokens(message("system",base)));
        Shadow shadow(stream,base,std::stod(get("--trigger",".7")),quantum,limit);
        json result={{"model",args.at("--model")},{"mode","shadow"},{"architecture",model.architecture},
            {"prefix_capacity",p},{"active_capacity",a},{"trigger_tokens",shadow.threshold},
            {"quantum_tokens",quantum},{"max_summary_tokens",limit},{"blocks",json::array()},
            {"probes",json::array()},{"idle_windows",json::array()}};
        auto save=[&]{
            result["elapsed_s"]=seconds(begin);result["events"]=shadow.history;
            result["prefix_text"]=model.text(stream.prefix,true);
            result["prefix_tokens"]=stream.prefix.size();result["active_tokens"]=stream.active.size();
            result["pending_phase"]=shadow.status();
            if(shadow.pending()) result["pending_job"]=shadow.current; else result.erase("pending_job");
            std::ofstream out(args.at("--output")); require(bool(out),"Cannot write output");
            out<<result.dump(2)<<'\n';require(bool(out),"Output write failed");
        };
        try {
            int index=0;
            for(const auto & item:data.at("blocks")) {
                const double delay=item.value("delay_before_ms",0.0);
                require(std::isfinite(delay) && delay>=0 && delay<=3600000,"Invalid arrival delay");
                const auto arrival=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(delay));
                int idle_steps=0;
                auto idle_begin=Clock::now();
                while(Clock::now()<arrival) {
                    if(shadow.step(false)) ++idle_steps;
                    else std::this_thread::sleep_until(arrival);
                }
                const double dispatch_delay=std::max(0.0,std::chrono::duration<double>(Clock::now()-arrival).count());
                if(delay>0) result["idle_windows"].push_back({{"before_block",index},{"requested_ms",delay},
                    {"elapsed_s",seconds(idle_begin)},{"steps",idle_steps},{"arrival_delay_s",dispatch_delay}});
                auto foreground=Clock::now();
                // Requests are now pending: no background quantum until this record and its probes finish.
                require(!shadow.step(true),"Background ran with foreground pending");
                auto ids=record_tokens(model,item,data.value("block_tokens",0));
                double append_s=0;
                for(size_t offset=0;offset<ids.size();) {
                    if(stream.active.size()==size_t(a)) {
                        shadow.at_capacity(); save();
                    }
                    shadow.start_if_due();
                    size_t n=std::min(ids.size()-offset,a-stream.active.size());
                    if(!shadow.pending()) n=std::min(n,size_t(shadow.threshold)-stream.active.size());
                    require(n>0,"Dispatcher made no progress");
                    append_s+=stream.append(Tokens(ids.begin()+offset,ids.begin()+offset+n));
                    offset+=n;
                    shadow.start_if_due();
                }
                if(item.contains("questions")) for(const auto & question:item.at("questions")) {
                    auto prompt=model.tokens(message("user",question.at("text"))+model.generation_prompt());
                    auto answer=stream.query(prompt,32);
                    auto probe=answer.metrics();probe["block"]=index;probe["question"]=question;
                    if(question.contains("expected")) probe["correct"]=answer.text==question.at("expected").get<std::string>();
                    result["probes"].push_back(probe);
                }
                result["blocks"].push_back({{"index",index},{"tokens",ids.size()},
                    {"foreground_s",seconds(foreground)},{"append_s_including_pending_shift",append_s},
                    {"arrival_delay_s",dispatch_delay},{"active_tokens",stream.active.size()},
                    {"prefix_tokens",stream.prefix.size()},{"shadow_phase",shadow.status()}});
                ++index;save();
                std::cout<<"block="<<index<<" P="<<stream.prefix.size()<<" A="<<stream.active.size()
                         <<" shadow="<<shadow.status()<<" commits="<<shadow.history.size()<<std::endl;
            }
            result["completed"]=true;
            result["unused_pending_discarded_at_eof"]=shadow.pending();save();
        } catch(const std::exception & e) {result["failure"]=e.what();save();throw;}
        return 0;
    } catch(const std::exception & e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
