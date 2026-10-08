// JSONL transport; one thread exclusively owns every llama context and GPU dispatch.
#include "shadow.hpp"
#include "server_session.hpp"
#include "chat.h"
#include "sampling.h"
#include "mtp.hpp"
#include "vision.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

using namespace ring;
struct Request { json data; std::shared_ptr<std::atomic<bool>> canceled; };
static std::mutex mutex;
static std::mutex output_mutex;
static std::condition_variable wake;
static std::deque<Request> queue;
static std::map<std::string, std::shared_ptr<std::atomic<bool>>> pending;
static bool eof = false;
static void emit(const json & value) { std::lock_guard<std::mutex> lock(output_mutex); std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl; }
static bool begins(const Tokens & text, const Tokens & prefix) {
    return text.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), text.begin());
}
static std::string piece(const Model & model, llama_token token) {
    char small[256]; int n=llama_token_to_piece(model.vocab,token,small,sizeof(small),0,true);
    if(n>=0) return std::string(small,n);
    std::string result(-n,'\0'); llama_token_to_piece(model.vocab,token,result.data(),result.size(),0,true); return result;
}
int main(int argc,char **argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--help") { std::cout<<"llama-shadow-worker --model FILE [--prefix 8192 --active 8192 --max-output 2048 --max-sessions 2 --quantum 64 --max-summary 512 --trigger .7 --gpu-layers 99 --cache-type f16|q8_0 --ram-rotation 0|1 --mtp-draft 0..8 --mmproj FILE --image-max-tokens 1024]\nJSONL stdin/stdout; use scripts/serve.py for HTTP.\n"; return 0; }
            require(i+1<argc && key.rfind("--",0)==0,"Expected --option value"); args[key]=argv[++i];
        }
        auto get=[&](std::string k,std::string d){return args.count(k)?args.at(k):d;};
        require(args.count("--model"),"--model required");
        int p=std::stoi(get("--prefix","8192")),a=std::stoi(get("--active","8192")), limit=std::stoi(get("--max-output","2048"));
        int sessions_limit=std::stoi(get("--max-sessions","2")),quantum=std::stoi(get("--quantum","64")),summary=std::stoi(get("--max-summary","512"));
        double trigger=std::stod(get("--trigger",".7"));
        require(get("--ram-rotation","0")=="0" || get("--ram-rotation","0")=="1", "Invalid RAM rotation option");
        const bool rotate=get("--ram-rotation","0")=="1";
        require(get("--shared-context","0")=="0" || get("--shared-context","0")=="1","Invalid shared context option");
        const bool shared=get("--shared-context","0")=="1";
        require(p>=256 && p<=65536 && a>=128 && a<=131072 && limit>0 && limit<=32768 && sessions_limit>0 && sessions_limit<=16,"Invalid capacities");
        require(quantum>0 && quantum<=512 && summary>0 && summary<=p && summary<=65536 && std::isfinite(trigger) && trigger>0 && trigger<1 && std::ceil(trigger*a)<a,"Invalid shadow options");
        setenv("VCORTEX_LLAMA_ALLOW_NONCONTIGUOUS","1",1);
        const std::string save_path=get("--slot-save-path","");
        if(!save_path.empty()) std::filesystem::create_directories(save_path);
        const int mtp_draft=std::stoi(get("--mtp-draft","0"));
        json compatibility={{"version",4},{"mtp_draft",mtp_draft},{"shared",shared},{"llama_session_version",LLAMA_SESSION_VERSION},{"model",std::filesystem::canonical(args.at("--model")).string()},
            {"model_size",std::filesystem::file_size(args.at("--model"))},
            {"model_mtime",std::filesystem::last_write_time(args.at("--model")).time_since_epoch().count()},
            {"p",p},{"a",a},{"output",limit},{"summary",summary},{"quantum",quantum},{"trigger",trigger},
            {"rotation",rotate},{"cache",get("--cache-type","f16")}};
        Model model(args.at("--model"),std::stoi(get("--gpu-layers","99")),get("--cache-type","f16"),mtp_draft);
        const auto projector=get("--mmproj","");
        const int image_max=std::stoi(get("--image-max-tokens","1024"));
        Vision vision(model,projector,get("--mmproj-gpu","1")=="1" && std::stoi(get("--gpu-layers","99"))>0,image_max);
        compatibility["image_max_tokens"]=image_max;
        if(!projector.empty()) compatibility["mmproj"]={{"path",std::filesystem::canonical(projector).string()},
            {"size",std::filesystem::file_size(projector)},{"mtime",std::filesystem::last_write_time(projector).time_since_epoch().count()}};
        auto templates=common_chat_templates_init(model.ptr.get(),"");
        std::map<std::string,std::unique_ptr<Session>> sessions;
        uint64_t tick=0; int resets=0,evictions=0,cancellations=0;
        emit({{"event","ready"},{"vision",bool(vision.ctx)},{"architecture",model.architecture},{"prefix_capacity",p},{"active_capacity",a},
              {"cache_type",ggml_type_name(model.cache_type)},{"summary_sequences",1},
              {"main_ctx_requested",p+a+(rotate ? std::max(Shadow::extra_capacity(p,summary),limit+512) : Shadow::extra_capacity(p,summary)+limit+512)},
              {"summary_ctx_requested",0},{"summary_mode","cached_branch"},{"main_sequences",rotate?2:5},{"ram_rotation",rotate},{"mtp_draft",mtp_draft}});
        std::thread reader([] {
            std::string line;
            while(std::getline(std::cin,line)) {
                try {
                    auto data=json::parse(line); const auto id=data.at("id").get<std::string>();
                    std::lock_guard<std::mutex> lock(mutex);
                    if(data.value("op","chat")=="cancel") { if(pending.count(id)) pending[id]->store(true); }
                    else if(pending.count(id) || queue.size()>=32) emit({{"id",id},{"event","error"},{"error","Duplicate id or full worker queue"}});
                    else { auto cancel=std::make_shared<std::atomic<bool>>(false); pending[id]=cancel; queue.push_back({std::move(data),cancel}); }
                    wake.notify_one();
                } catch(const std::exception & e) { std::cerr<<"Invalid worker input: "<<e.what()<<'\n'; }
            }
            std::lock_guard<std::mutex> lock(mutex); eof=true;
            for(auto & item:pending) item.second->store(true);
            wake.notify_one();
        });
        // Always join the input reader, including fatal dispatch failures.
        try {
            while(true) {
                Request req;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    if(eof && queue.empty()) break;
                    if(queue.empty()) {
                        Session * idle=nullptr;
                        for(auto & item:sessions) if(item.second->shadow.pending() && item.second->shadow.phase!=Shadow::Phase::ready && item.second->shadow.phase!=Shadow::Phase::failed) {
                            if(!idle || item.second->used<idle->used) idle=item.second.get();
                        }
                        if(idle) {
                            lock.unlock();
                            try { idle->shadow.step(false); }
                            catch(const std::exception & e) { std::cerr<<"Background summary failed: "<<e.what()<<'\n'; }
                            continue;
                        }
                        wake.wait(lock,[&]{return eof || !queue.empty();}); continue;
                    }
                    req=std::move(queue.front());queue.pop_front();
                }
                const auto id=req.data.at("id").get<std::string>();
                const bool slot_request=req.data.contains("id_slot");
                const auto session_id=slot_request ? "slot:"+std::to_string(req.data.at("id_slot").get<int>()) : req.data.contains("session_id") ? "session:"+req.data.at("session_id").get<std::string>() : "ephemeral:"+id;
                auto send=[&](json event) {event["id"]=id;emit(event);};
                auto check=[&]{require(!req.canceled->load(),"Request canceled");};
                try {
                    check();
                    if(slot_request) require(req.data.at("id_slot").is_number_integer() && req.data.at("id_slot").get<int>()>=0 && req.data.at("id_slot").get<int>()<sessions_limit,"Invalid id_slot");
                    const auto op=req.data.value("op","chat");
                    if(op=="tokenize") {
                        send({{"event","done"},{"result",{{"tokens",model.tokens(req.data.at("content"),req.data.value("parse_special",false))}}}});
                    } else if(op=="status") {
                        json state=json::object();for(auto & item:sessions) state[item.first.rfind("session:",0)==0?item.first.substr(8):item.first]=item.second->stats();
                        send({{"event","done"},{"result",{{"sessions",state},{"resets",resets},{"evictions",evictions},{"cancellations",cancellations}}}});
                    } else if(op=="inspect") {
                        common_chat_templates_inputs inputs;
                        auto media=vision.input(req.data.at("messages"));
                        inputs.messages=common_chat_msgs_parse_oaicompat(media.messages);
                        const bool thinking=req.data.value("chat_template_kwargs",json::object()).value("enable_thinking",false);
                        inputs.enable_thinking=thinking; inputs.parallel_tool_calls=false;
                        inputs.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
                        inputs.chat_template_kwargs["enable_thinking"]=thinking?"true":"false";
                        inputs.chat_template_kwargs["preserve_reasoning"]="true";
                        if(req.data.contains("tools")) inputs.tools=common_chat_tools_parse_oaicompat(req.data.at("tools"));
                        inputs.tool_choice=common_chat_tool_choice_parse_oaicompat(req.data.value("tool_choice",std::string("auto")));
                        json system=json::array();
                        for(const auto & msg:req.data.at("messages")) {if(msg.at("role")!="system" && msg.at("role")!="developer") break;system.push_back(msg);}
                        inputs.add_generation_prompt=false;
                            auto prefix_inputs=inputs;prefix_inputs.messages=common_chat_msgs_parse_oaicompat(system);
                            // Qwen3.5 rejects a system-only/empty conversation. Render a sentinel user
                            // through the real template and keep everything before its ChatML boundary.
                            const std::string sentinel="vcortex_prefix_boundary_"+id;
                            common_chat_msg sentinel_msg;sentinel_msg.role="user";sentinel_msg.content=sentinel;
                            prefix_inputs.messages.push_back(sentinel_msg);
                            auto rendered_prefix=common_chat_templates_apply(templates.get(),prefix_inputs).prompt;
                            const auto boundary=rendered_prefix.rfind("<|im_start|>user\n"+sentinel);
                            require(boundary!=std::string::npos,"Unsupported model template user boundary");
                            std::string immutable=rendered_prefix.substr(0,boundary);
                            if(immutable.empty()) immutable=message("system","You are a helpful assistant.");
                        const auto count=model.tokens(immutable).size();
                        send({{"event","done"},{"result",{{"prefix_tokens",count},{"prefix_capacity",p},
                            {"summary_capacity",summary},{"prefix_fits",count+summary+128<=size_t(p)},
                            {"history_tokens",vision.tokenize(common_chat_templates_apply(templates.get(),inputs).prompt,media).tokens.size()}}}});
                    } else if(op=="save") {
                        require(sessions.count(session_id),"Slot is empty");
                        auto & s=*sessions.at(session_id);
                        require(s.shadow.phase!=Shadow::Phase::failed,"Cannot save failed shadow");
                        const auto path=checkpoint_path(save_path,req.data.at("filename"));
                        const auto meta=s.snapshot(compatibility);
                        const auto bytes=save_checkpoint(s.stream.ctx,path,meta);
                        send({{"event","done"},{"result",{{"id_slot",req.data.at("id_slot")},{"filename",req.data.at("filename")},{"n_saved",s.ledger.size()},{"n_written",bytes}}}});
                    } else if(op=="restore") {
                        const auto path=checkpoint_path(save_path,req.data.at("filename"));
                        Checkpoint saved(path);
                        require(saved.metadata.at("compatibility")==compatibility,"Incompatible model or context checkpoint");
                        // Validation happens before releasing the resident GPU allocation.
                        sessions.erase(session_id);
                        if(sessions.size()>=size_t(sessions_limit)) {
                            auto victim=std::min_element(sessions.begin(),sessions.end(),[](const auto & x,const auto & y){return x.second->used<y.second->used;});
                            sessions.erase(victim);++evictions;
                        }
                        auto fresh=std::make_unique<Session>(model,p,a,limit,quantum,summary,trigger,rotate,
                            saved.metadata.at("immutable"),saved.metadata.at("signature"),check,false,shared);
                        fresh->restore(saved);fresh->used=++tick;
                        const auto tokens=fresh->ledger.size();sessions[session_id]=std::move(fresh);
                        send({{"event","done"},{"result",{{"id_slot",req.data.at("id_slot")},{"filename",req.data.at("filename")},{"n_restored",tokens},{"n_read",saved.size}}}});
                    } else if(op=="reset") {
                        sessions.erase(session_id);send({{"event","done"},{"result",{{"reset",true}}}});
                    } else {
                        auto started=Clock::now();
                        common_chat_templates_inputs inputs;
                        auto media=vision.input(req.data.at("messages"));
                        inputs.messages=common_chat_msgs_parse_oaicompat(media.messages);
                        const bool thinking=req.data.value("chat_template_kwargs",json::object()).value("enable_thinking",false);
                        inputs.enable_thinking=thinking; inputs.parallel_tool_calls=false;
                        inputs.reasoning_format=COMMON_REASONING_FORMAT_DEEPSEEK;
                        inputs.chat_template_kwargs["enable_thinking"]=thinking?"true":"false";
                        inputs.chat_template_kwargs["preserve_reasoning"]="true";
                        if(req.data.contains("tools")) inputs.tools=common_chat_tools_parse_oaicompat(req.data.at("tools"));
                        inputs.tool_choice=common_chat_tool_choice_parse_oaicompat(req.data.value("tool_choice",std::string("auto")));
                        json system=json::array();
                        for(const auto & msg:req.data.at("messages")) {if(msg.at("role")!="system" && msg.at("role")!="developer") break;system.push_back(msg);}
                        const std::string signature=json({{"thinking",thinking},{"system",system},{"tools",req.data.value("tools",json::array())},{"tool_choice",req.data.value("tool_choice",std::string("auto"))}}).dump();
                        const bool had_session=sessions.count(session_id)>0;
                        if(sessions.count(session_id) && (sessions[session_id]->signature!=signature || sessions[session_id]->shadow.phase==Shadow::Phase::failed)) {sessions.erase(session_id);++resets;}
                        if(sessions.count(session_id)) inputs.now=sessions[session_id]->now;
                        inputs.add_generation_prompt=false;
                        auto history=common_chat_templates_apply(templates.get(),inputs);
                        auto multimodal=vision.tokenize(history.prompt,media);
                        const auto & full=multimodal.tokens;
                        bool reuse=sessions.count(session_id) && begins(full,sessions[session_id]->ledger);
                        if(reuse) {
                            const auto & saved=sessions[session_id]->image_ledger;
                            reuse=media.hashes.size()>=saved.size() && std::equal(saved.begin(),saved.end(),media.hashes.begin());
                        }
                        if(sessions.count(session_id) && !reuse) {sessions.erase(session_id);++resets;}
                        if(!sessions.count(session_id)) {
                            if(sessions.size()>=size_t(sessions_limit)) {
                                auto victim=std::min_element(sessions.begin(),sessions.end(),[](const auto & x,const auto & y){return x.second->used<y.second->used;});
                                sessions.erase(victim);++evictions;
                            }
                            // Protect the complete template-rendered system and tool definitions from lossy summarization.
                            auto prefix_inputs=inputs;prefix_inputs.messages=common_chat_msgs_parse_oaicompat(system);
                            // Qwen3.5 rejects a system-only/empty conversation. Render a sentinel user
                            // through the real template and keep everything before its ChatML boundary.
                            const std::string sentinel="vcortex_prefix_boundary_"+id;
                            common_chat_msg sentinel_msg;sentinel_msg.role="user";sentinel_msg.content=sentinel;
                            prefix_inputs.messages.push_back(sentinel_msg);
                            auto rendered_prefix=common_chat_templates_apply(templates.get(),prefix_inputs).prompt;
                            const auto boundary=rendered_prefix.rfind("<|im_start|>user\n"+sentinel);
                            require(boundary!=std::string::npos,"Unsupported model template user boundary");
                            std::string immutable=rendered_prefix.substr(0,boundary);
                            if(immutable.empty()) immutable=message("system","You are a helpful assistant.");
                            auto session=std::make_unique<Session>(model,p,a,limit,quantum,summary,trigger,rotate,immutable,signature,check,true,shared);
                            session->now=inputs.now;
                            auto prefix_ids=model.tokens(immutable);
                            if(begins(full,prefix_ids)) session->ledger=prefix_ids;
                            sessions[session_id]=std::move(session);
                        }
                        auto & s=*sessions.at(session_id);s.used=++tick;if(reuse) ++s.hits;
                        if(s.context_id.empty()) s.context_id=id;
                        const auto previous_revision=s.commits;
                        vision.append(s,multimodal,s.ledger.size(),check);s.ledger=full;s.image_ledger=media.hashes;
                        const auto context=s.context_state(!reuse?(had_session?"rebuilt":"created"):s.commits!=previous_revision?"compacted":"reused");
                        send({{"event","context"},{"context_state",context}});
                        inputs.add_generation_prompt=true;
                        auto params=common_chat_templates_apply(templates.get(),inputs);
                        auto prompt=model.tokens(params.prompt);
                        if(!media.hashes.empty()) {
                            require(params.prompt.compare(0,history.prompt.size(),history.prompt)==0,"Multimodal generation prompt changed history");
                            prompt=full;auto tail=model.tokens(params.prompt.substr(history.prompt.size()));prompt.insert(prompt.end(),tail.begin(),tail.end());
                        }
                        require(begins(prompt,full),"Model template cannot append a generation prompt");
                        Tokens suffix(prompt.begin()+full.size(),prompt.end());
                        require(!suffix.empty() && suffix.size()<=512,"Unsupported generation prompt length");
                        common_chat_parser_params parser(params);if(!params.parser.empty()) parser.parser.load(params.parser);
                        auto & ctx=s.stream.ctx;ctx.remove(1);ctx.copy(0,1);
                        int position=s.stream.next_position();
                        std::unique_ptr<MtpAnswer> mtp;
                        if(mtp_draft) {mtp=std::make_unique<MtpAnswer>(ctx,limit+1024);mtp->prefill(suffix,position);}
                        else ctx.decode(suffix,position,1,true);
                        position+=suffix.size();check();
                        common_params_sampling sampling;
                        sampling.penalty_last_n=64;
                        sampling.penalty_repeat=req.data.value("repeat_penalty",1.0f);
                        sampling.penalty_freq=req.data.value("frequency_penalty",0.0f);
                        sampling.penalty_present=req.data.value("presence_penalty",0.0f);
                        sampling.temp=req.data.value("temperature",0.7f);
                        sampling.top_k=req.data.value("top_k",40);
                        sampling.top_p=req.data.value("top_p",0.95f);
                        sampling.min_p=req.data.value("min_p",0.05f);
                        sampling.seed=req.data.value("seed",uint32_t(LLAMA_DEFAULT_SEED));
                        sampling.samplers={COMMON_SAMPLER_TYPE_PENALTIES,COMMON_SAMPLER_TYPE_TOP_K,
                            COMMON_SAMPLER_TYPE_TOP_P,COMMON_SAMPLER_TYPE_MIN_P,COMMON_SAMPLER_TYPE_TEMPERATURE};
                        if(!params.grammar.empty()) sampling.grammar={COMMON_GRAMMAR_TYPE_TOOL_CALLS,params.grammar};
                        sampling.grammar_lazy=params.grammar_lazy;
                        sampling.grammar_triggers=params.grammar_triggers;
                        std::unique_ptr<common_sampler,decltype(&common_sampler_free)> sampler(common_sampler_init(model.ptr.get(),sampling),common_sampler_free);
                        require(bool(sampler),"Failed to initialize chat sampler");
                        for(auto token:s.stream.prefix) common_sampler_accept(sampler.get(),token,false);
                        for(auto token:s.stream.active) common_sampler_accept(sampler.get(),token,false);
                        int maximum=req.data.value("max_tokens",limit);require(maximum>0 && maximum<=limit,"max_tokens exceeds configured --max-output");
                        std::string raw, emitted, emitted_reasoning;common_chat_msg parsed;bool eos=false;int count=0;
                        auto parse_raw = [&](bool partial) {
                            try { return common_chat_parse(raw,partial,parser); }
                            catch(const std::exception & e) {
                                if(const char * path=std::getenv("VCORTEX_PARSE_ERROR_TRACE")) {
                                    std::ofstream trace(path,std::ios::app);
                                    trace<<json({{"request_id",id},{"partial",partial},{"eos",eos},{"tokens",count},{"error",e.what()},{"raw",raw}}).dump(-1,' ',false,json::error_handler_t::replace)<<'\n';
                                }
                                throw;
                            }
                        };
                        send({{"event","delta"},{"delta",{{"role","assistant"}}}});
                        std::deque<llama_token> verified;
                        for(;count<maximum;) {
                            check(); llama_token token;
                            if(verified.empty()) {token=common_sampler_sample(sampler.get(),ctx.ptr.get(),-1);common_sampler_accept(sampler.get(),token,true);}
                            else {token=verified.front();verified.pop_front();}
                            ++count;
                            if(llama_vocab_is_eog(model.vocab,token)) {eos=true;break;}
                            raw+=piece(model,token);
                            // Do not emit partial UTF-8 code points. Tool calls are emitted atomically after final parsing.
                            bool valid=true;try { (void)json(raw).dump(); } catch(const json::type_error &) {valid=false;}
                            if(valid) {
                                parsed=parse_raw(true);
                                require(parsed.reasoning_content.compare(0,emitted_reasoning.size(),emitted_reasoning)==0,"Non-monotonic reasoning parser output");
                                if(parsed.reasoning_content.size()>emitted_reasoning.size()) {send({{"event","delta"},{"delta",{{"reasoning_content",parsed.reasoning_content.substr(emitted_reasoning.size())}}}});emitted_reasoning=parsed.reasoning_content;}
                                require(parsed.content.compare(0,emitted.size(),emitted)==0,"Non-monotonic chat parser output");
                                if(parsed.content.size()>emitted.size()) {send({{"event","delta"},{"delta",{{"content",parsed.content.substr(emitted.size())}}}});emitted=parsed.content;}
                            }
                            if(count<maximum) {
                                if(mtp) {
                                    if(verified.empty()) {
                                        const auto next=mtp->next(sampler.get(),token,position,maximum-count);
                                        verified.insert(verified.end(),next.begin(),next.end());
                                    }
                                } else ctx.decode({token},position++,1,true);
                            }
                        }
                        const auto mtp_stats=mtp?mtp->stats():json({{"enabled",false}});
                        mtp.reset();
                        ctx.remove(1);check();parsed=parse_raw(!eos);
                        require(parsed.reasoning_content.compare(0,emitted_reasoning.size(),emitted_reasoning)==0,"Non-monotonic final reasoning output");
                        if(parsed.reasoning_content.size()>emitted_reasoning.size()) send({{"event","delta"},{"delta",{{"reasoning_content",parsed.reasoning_content.substr(emitted_reasoning.size())}}}});
                        require(parsed.content.compare(0,emitted.size(),emitted)==0,"Non-monotonic final parser output");
                        if(parsed.content.size()>emitted.size()) send({{"event","delta"},{"delta",{{"content",parsed.content.substr(emitted.size())}}}});
                        json calls=json::array();int index=0;
                        require(inputs.tool_choice!=COMMON_CHAT_TOOL_CHOICE_NONE || parsed.tool_calls.empty(),"Model emitted a tool call despite tool_choice=none");
                        for(auto & call:parsed.tool_calls) {
                            require(!call.name.empty(),"Incomplete tool call");
                            // Never execute truncated or syntactically invalid tool arguments.
                            require(eos,"Tool call exceeded max_tokens");auto arguments=json::parse(call.arguments);
                            require(arguments.is_object(),"Tool arguments must be an object");
                            call.id="call_"+id+"_"+std::to_string(index);
                            calls.push_back({{"index",index++},{"id",call.id},{"type","function"},{"function",{{"name",call.name},{"arguments",call.arguments}}}});
                        }
                        if(!calls.empty()) send({{"event","delta"},{"delta",{{"tool_calls",calls}}}});
                        parsed.role="assistant";
                        auto response=json::parse(parsed.to_json_oaicompat().dump());
                        const auto finish=!calls.empty()?"tool_calls":eos?"stop":"length";
                        send({{"event","done"},{"message",response},{"finish_reason",finish},
                            {"usage",{{"prompt_tokens",prompt.size()},{"completion_tokens",count},{"total_tokens",prompt.size()+count}}},
                            {"shadow",s.stats()},{"mtp",mtp_stats},{"context_state",context},{"cache_reused",reuse},{"elapsed_s",seconds(started)}});
                        if(!slot_request && !req.data.contains("session_id")) sessions.erase(session_id);
                    }
                } catch(const std::exception & e) {
                    // Administrative failures must not evict a healthy resident conversation.
                    if(req.data.value("op","chat")=="chat") sessions.erase(session_id);
                    if(req.canceled->load()) ++cancellations;
                    send({{"event","error"},{"error",e.what()},{"canceled",req.canceled->load()}});
                }
                std::lock_guard<std::mutex> lock(mutex);pending.erase(id);
            }
        } catch(...) { reader.detach(); throw; }
        reader.join();return 0;
    } catch(const std::exception & e) {std::cerr<<"shadow-worker: "<<e.what()<<'\n';return 1;}
}
