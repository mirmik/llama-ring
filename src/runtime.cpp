#include "runtime.hpp"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace ring {
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
std::string trim(std::string text) {
    auto begin = text.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}
std::string message(const std::string & role, const std::string & text) {
    return "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
}
Model::Model(const std::string & path, int gpu_layers, const std::string & cache, int draft, const std::string & mtp_path) : mtp_draft(draft) {
    require(mtp_path.empty() || draft > 0, "--mtp-model requires --mtp-draft > 0");
    require(draft >= 0 && draft <= 8, "MTP draft count must be between 0 and 8");
    require(cache == "f16" || cache == "q8_0", "Cache type must be f16 or q8_0");
    cache_type = cache == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    // Vulkan's automatic MMVQ policy switches activation precision between
    // single-token decode and verifier batches. With Q8 KV this exceeds the
    // measured MTP error envelope. Choose the floating-input path before device
    // initialization; leave other modes and explicit backend overrides alone.
    const char * mtp_auto = std::getenv("VCORTEX_MTP_VULKAN_AUTO");
    if (draft > 0 && gpu_layers > 0 && cache == "q8_0" && !(mtp_auto && std::string(mtp_auto) == "1") &&
        !std::getenv("GGML_VK_FORCE_MMVQ") && !std::getenv("GGML_VK_DISABLE_MMVQ")) {
        require(setenv("GGML_VK_DISABLE_MMVQ", "1", 0) == 0, "Cannot configure Vulkan MTP precision");
        std::fprintf(stderr, "llama-ring: disabling Vulkan MMVQ for Q8 MTP batch consistency\n");
    }
    ggml_backend_load_all();
    if (gpu_layers > 0) {
        bool gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
            gpu |= type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        require(gpu, "No GPU visible; use --gpu-layers 0 for an explicit CPU run");
    }
    auto params = llama_model_default_params();
    params.n_gpu_layers = gpu_layers;
    params.load_mtp = draft > 0 && mtp_path.empty();
    // Keep architecture-marked lookup tables lazy even on integrated GPUs.
    params.lazy_mode = LLAMA_LAZY_MODE_ON;
    ptr.reset(llama_model_load_from_file(path.c_str(), params));
    require(bool(ptr), "Cannot load model: " + path);
    char arch[128] = {};
    llama_model_meta_val_str(ptr.get(), "general.architecture", arch, sizeof(arch));
    architecture = arch;
    char model_name[256] = {};
    llama_model_meta_val_str(ptr.get(), "general.name", model_name, sizeof(model_name));
    name = model_name;
    indexed = architecture == "qwen4exp";
    hybrid = architecture == "qwen35" || indexed;
    if (!mtp_path.empty()) {
        require(indexed, "Separate MTP models currently require qwen4exp");
        params.load_mtp = true;
        mtp_model.reset(llama_model_load_from_file(mtp_path.c_str(), params));
        require(bool(mtp_model), "Cannot load MTP model: " + mtp_path);
        char draft_arch[128] = {};
        llama_model_meta_val_str(mtp_model.get(), "general.architecture", draft_arch, sizeof(draft_arch));
        require(architecture == draft_arch && llama_model_n_embd(ptr.get()) == llama_model_n_embd(mtp_model.get()) &&
                llama_model_n_embd_out(ptr.get()) == llama_model_n_embd_out(mtp_model.get()),
                "MTP architecture/embedding width differs from target");
        const auto * target_vocab = llama_model_get_vocab(ptr.get());
        const auto * draft_vocab = llama_model_get_vocab(mtp_model.get());
        require(llama_vocab_n_tokens(target_vocab) == llama_vocab_n_tokens(draft_vocab), "MTP vocabulary size differs");
        for (int i = 0; i < llama_vocab_n_tokens(target_vocab); ++i) {
            require(std::string(llama_vocab_get_text(target_vocab, i)) == llama_vocab_get_text(draft_vocab, i) &&
                    llama_vocab_get_attr(target_vocab, i) == llama_vocab_get_attr(draft_vocab, i), "MTP vocabulary differs");
        }
    }
    require(!draft || (hybrid && llama_model_n_layer_nextn(mtp_model ? mtp_model.get() : ptr.get()) == 1),
            "MTP requires one NextN head; Flash-Next needs --mtp-model pointing to a compatible head");
    setenv("VCORTEX_QWEN35_TEXT_SHIFT", hybrid ? "1" : "0", 1);
    setenv("VCORTEX_QWEN4_RING", indexed ? "1" : "0", 1);
    require(architecture == "qwen3" || hybrid, "Supported architectures: qwen3 / qwen35 / qwen4exp");
    vocab = llama_model_get_vocab(ptr.get());
}
std::string Model::generation_prompt() const {
    return "<|im_start|>assistant\n" + std::string(hybrid ? "<think>\n\n</think>\n\n" : "");
}
Tokens Model::tokens(const std::string & text, bool special) const {
    int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, special);
    Tokens result(std::abs(n));
    n = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, special);
    require(n >= 0, "Tokenization failed");
    result.resize(n);
    return result;
}
std::string Model::text(const Tokens & ids, bool special) const {
    std::string result;
    std::vector<char> buf(128);
    for (auto token : ids) {
        int n = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, special);
        if (n < 0) {
            buf.resize(-n);
            n = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, special);
        }
        require(n >= 0, "Token decoding failed");
        result.append(buf.data(), n);
    }
    // A generation limit or an evicted token boundary can split a UTF-8
    // character. Preserve the raw token IDs, use replacement in readable text.
    return json::parse(json(result).dump(-1, ' ', false, json::error_handler_t::replace)).get<std::string>();
}
json Generation::metrics() const {
    return {{"text", text}, {"tokens", tokens}, {"eos", eos},
            {"prefill_tokens", prefill_tokens}, {"prefill_s", prefill_s}, {"decode_s", decode_s}};
}
Context::Context(Model & model, int capacity, int batch_size, int sequences, bool rotate) : model(model), batch_size(batch_size), ram_rotation(rotate) {
    require(capacity > 0 && batch_size > 0 && sequences > 0 && sequences <= 5, "Invalid context allocation");
    auto params = llama_context_default_params();
    params.n_ctx = capacity;
    params.n_batch = batch_size;
    params.n_ubatch = batch_size;
    params.n_seq_max = rotate ? 2 : sequences;
    // Ordinary decode needs the final row; MTP verifies up to draft+1 rows.
    // Avoid reserving n_batch full-vocabulary projections.
    params.n_outputs_max = std::max(params.n_seq_max, uint32_t(model.mtp_draft + 1));
    params.n_rs_seq = model.mtp_draft;
    params.kv_unified = true;
    params.n_threads = 8;
    params.n_threads_batch = 8;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = params.type_v = model.cache_type;
    params.no_perf = false;
    ptr.reset(llama_init_from_model(model.ptr.get(), params));
    require(bool(ptr), "Cannot create context");
    require(llama_memory_can_shift(memory()), "Model cache cannot shift");
}
llama_memory_t Context::memory() const { return llama_get_memory(ptr.get()); }
llama_memory_t Context::attention_memory() const {
    return model.hybrid ? llama_memory_hybrid_attention(memory()) : memory();
}
llama_memory_t Context::recurrent_memory() const { return llama_memory_hybrid_recurrent(memory()); }
json Context::rotation_stats() const {
    uint64_t bytes = 0;
    for (const auto & branch : parked) bytes += branch.bytes.size();
    return {{"enabled",ram_rotation},{"resident_branch",resident},{"ram_bytes",bytes},
        {"peak_ram_bytes",peak_ram_bytes},{"exports",exports},{"imports",imports},
        {"export_bytes",export_bytes},{"import_bytes",import_bytes},
        {"export_s",export_s},{"import_s",import_s}};
}
void Context::park() {
    if (!ram_rotation || resident < 0) return;
    auto start = Clock::now();
    llama_synchronize(ptr.get());
    auto & branch = parked.at(resident);
    // Remove only this branch's membership of shared cells. Main retains them.
    // Serialization then transfers private KV and the recurrent tail only.
    if (branch.shared_end > 0)
        require(llama_memory_attention_seq_rm(memory(),1,0,branch.shared_end), "Cannot detach shared KV");
    try {
        std::vector<uint8_t> bytes(llama_state_seq_get_size(ptr.get(),1));
        require(llama_state_seq_get_data(ptr.get(),bytes.data(),bytes.size(),1)==bytes.size(), "RAM export failed");
        require(llama_memory_seq_rm(memory(),1,-1,-1), "Cannot release RAM branch GPU cells");
        branch.bytes = std::move(bytes);
    } catch (...) {
        // Export allocation/serialization failure leaves the resident state usable.
        if (branch.shared_end > 0) llama_memory_attention_seq_cp(memory(),0,1,0,branch.shared_end);
        throw;
    }
    ++exports; export_bytes += branch.bytes.size(); export_s += seconds(start);
    resident = -1;
    uint64_t bytes = 0;
    for (const auto & item : parked) bytes += item.bytes.size();
    peak_ram_bytes = std::max(peak_ram_bytes,bytes);
}
int Context::physical(int seq) {
    require(seq >= 0 && seq < 5, "Invalid logical sequence");
    if (!ram_rotation || seq == 0) return seq;
    if (resident == seq) return 1;
    park();
    auto & branch = parked.at(seq);
    auto start = Clock::now();
    if (!branch.bytes.empty()) {
        try {
            require(llama_state_seq_set_data(ptr.get(),branch.bytes.data(),branch.bytes.size(),1)==branch.bytes.size(),
                    "RAM import failed; host copy retained");
        } catch (...) {
            llama_memory_seq_rm(memory(),1,-1,-1);
            throw;
        }
        ++imports; import_bytes += branch.bytes.size(); import_s += seconds(start);
        std::vector<uint8_t>().swap(branch.bytes);
    }
    if (branch.shared_end > 0) llama_memory_attention_seq_cp(memory(),0,1,0,branch.shared_end);
    resident = seq;
    return 1;
}
void Context::share_prefix(int seq, int end) {
    if (ram_rotation) { require(seq > 0 && seq < 5 && resident == seq, "Inactive shared branch"); parked.at(seq).shared_end = end; }
}
void Context::rename_branch(int src, int dst) {
    require(ram_rotation && src > 0 && dst > 0 && src != dst, "Invalid branch rename");
    remove(dst);
    parked.at(dst) = std::move(parked.at(src)); parked.at(src) = {};
    if (resident == src) resident = dst;
}
void Context::remove(int seq, int from, int to) {
    if (ram_rotation && seq != 0 && from == -1 && to == -1) {
        require(seq > 0 && seq < 5, "Invalid logical sequence");
        if (resident == seq) {
            require(llama_memory_seq_rm(memory(),1,-1,-1), "KV removal failed");
            resident = -1;
        }
        parked.at(seq) = {};
        return;
    }
    require(llama_memory_seq_rm(memory(),physical(seq),from,to), "KV removal failed");
}
void Context::copy(int src, int dst, int from, int to) {
    if (ram_rotation) {
        require(src == 0 && dst > 0 && dst < 5 && from <= 0, "RAM branches must fork main from zero");
        remove(dst);
        int target = physical(dst);
        if (model.indexed) {
            llama_memory_attention_seq_cp(memory(),0,target,from,to);
            llama_memory_seq_cp(recurrent_memory(),0,target,-1,-1);
        } else llama_memory_seq_cp(memory(),0,target,from,to);
        share_prefix(dst,to < 0 ? llama_memory_seq_pos_max(attention_memory(),0)+1 : to);
    } else if (model.indexed) {
        llama_memory_attention_seq_cp(memory(),src,dst,from,to);
        llama_memory_seq_cp(recurrent_memory(),src,dst,-1,-1);
    } else llama_memory_seq_cp(memory(),src,dst,from,to);
}
void Context::decode(const Tokens & ids, int position, int seq, bool want_logits) {
    require(!ids.empty(), "Cannot decode empty input");
    if (ram_rotation && seq == 0) park();
    seq = physical(seq);
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    try {
        for (size_t start = 0; start < ids.size(); start += batch_size) {
            batch.n_tokens = std::min<size_t>(batch_size, ids.size() - start);
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = ids[start + i];
                batch.pos[i] = position + start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = want_logits && start + i + 1 == ids.size();
            }
            int code = llama_decode(ptr.get(), batch);
            require(code == 0, "llama_decode failed: " + std::to_string(code));
        }
        llama_synchronize(ptr.get());
    } catch (...) { llama_batch_free(batch); throw; }
    llama_batch_free(batch);
}
std::vector<float> Context::logits() {
    const float * data = llama_get_logits_ith(ptr.get(), -1);
    require(data != nullptr, "No logits");
    std::vector<float> result(data, data + llama_vocab_n_tokens(model.vocab));
    require(std::all_of(result.begin(), result.end(), [](float f) { return std::isfinite(f); }), "Nonfinite logits");
    return result;
}
Generation Context::generate(const Tokens & prompt, int position, int limit, int source, int prefix_end) {
    require(source != 1 && limit > 0, "Invalid generation branch");
    require(!model.hybrid || prefix_end < 0, "Hybrid generation must branch from a complete saved state");
    remove(1);
    copy(source, 1, -1, prefix_end);
    Generation result;
    try {
        auto start = Clock::now();
        decode(prompt, position, 1, true);
        result.first_logits = logits();
        result.prefill_s = seconds(start);
        result.prefill_tokens = prompt.size();
        position += prompt.size();
        start = Clock::now();
        auto current = result.first_logits;
        for (int i = 0; i < limit; ++i) {
            auto token = llama_token(std::max_element(current.begin(), current.end()) - current.begin());
            result.tokens.push_back(token);
            if (llama_vocab_is_eog(model.vocab, token)) { result.eos = true; break; }
            if (i + 1 < limit) { decode({token}, position++, 1, true); current = logits(); }
        }
        result.decode_s = seconds(start);
        result.text = trim(model.text(result.tokens));
    } catch (...) { remove(1); throw; }
    remove(1);
    return result;
}
std::vector<uint8_t> Context::state(int seq) {
    seq = physical(seq);
    llama_synchronize(ptr.get());
    std::vector<uint8_t> result(llama_state_seq_get_size(ptr.get(), seq));
    require(llama_state_seq_get_data(ptr.get(), result.data(), result.size(), seq) == result.size(), "State serialization failed");
    return result;
}
std::vector<uint8_t> Context::recurrent_state(int seq) {
    require(model.hybrid, "Recurrent state requires a hybrid model");
    seq = physical(seq);
    llama_synchronize(ptr.get());
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> result(llama_state_seq_get_size_ext(ptr.get(), seq, flags));
    require(llama_state_seq_get_data_ext(ptr.get(), result.data(), result.size(), seq, flags) == result.size(), "Recurrent serialization failed");
    return result;
}
std::vector<uint8_t> Context::range_state(int from, int to) {
    require(!ram_rotation, "range_state is a resident-only diagnostic");
    remove(2); copy(0, 2, from, to);
    // Hybrid seq_cp copies the latest recurrent tail regardless of range.
    // Remove it so this helper compares only the chosen attention interval.
    if (model.hybrid) require(llama_memory_seq_rm(recurrent_memory(), 2, -1, -1), "Cannot remove test recurrent branch");
    auto result = state(2);
    remove(2);
    return result;
}
void Context::restore(const std::vector<uint8_t> & state, int seq) {
    if (ram_rotation) remove(seq);
    seq = physical(seq);
    require(llama_state_seq_set_data(ptr.get(), state.data(), state.size(), seq) == state.size(), "State restoration failed");
}
Stream::Stream(Model & model, int p, int a, int extra, int generation, int sequences, bool rotate)
    : ctx(model, p + a + (rotate ? std::max(generation,extra) : generation + extra), 512, sequences, rotate), prefix_capacity(p), active_capacity(a), prefix_limit(p), total_capacity(p+a) {
    require(p > 0 && a > 0 && extra >= 0 && generation > 0, "Invalid stream capacities");
}
void Stream::enable_flexible() {
    require(active.empty(), "Initial flexible layout requires empty history");
    flexible=true;prefix_capacity=prefix.size();active_capacity=total_capacity-prefix_capacity;
}
double Stream::append_prefix(const Tokens & ids) {
    require(!ctx.ram_rotation || (ctx.parked[1].shared_end == 0 && ctx.parked[4].shared_end == 0),
            "Cannot edit a prefix borrowed by a RAM branch");
    require(!flexible, "Use prepared prefix replacement for flexible context");
    require(prefix.size() + ids.size() <= size_t(prefix_capacity), "Prefix full (old memory is immutable)");
    if (ids.empty()) return 0;
    auto start = Clock::now();
    if (ctx.model.hybrid) {
        // Sequence 3 preserves the independent prefix recurrence. Decoding a
        // prefix addition on sequence 0 would consume/overwrite A's latest state.
        ctx.decode(ids, prefix.size(), 3);
        llama_memory_attention_seq_cp(ctx.memory(), ctx.physical(3), 0, prefix.size(), prefix.size() + ids.size());
        ctx.share_prefix(3, prefix.size() + ids.size());
        if (!started) {
            // Initial P: seed the stream recurrence. Later P additions must
            // never replace the running stream's recurrent state.
            llama_memory_seq_cp(ctx.recurrent_memory(), ctx.physical(3), 0, -1, -1);
        }
    } else {
        ctx.decode(ids, prefix.size());
    }
    prefix.insert(prefix.end(), ids.begin(), ids.end());
    return seconds(start);
}
int Stream::position_at(int offset) const {
    require(offset >= 0 && size_t(offset) <= active.size(), "Invalid token boundary");
    int result = offset;
    for (const auto & image : images) {
        require(offset <= image.begin || offset >= image.begin + image.count, "Cannot split an image chunk");
        if (offset >= image.begin + image.count) result += image.positions - image.count;
    }
    return result;
}
void Stream::erase_active(int count) {
    (void) position_at(count);
    std::vector<ImageSpan> retained;
    for (auto image : images) if (image.begin >= count) { image.begin -= count; retained.push_back(image); }
    images = std::move(retained);
    active.erase(active.begin(), active.begin() + count);
}
double Stream::append(const Tokens & ids) {
    require(active.size() + ids.size() <= size_t(active_capacity), "Evict before appending");
    if (ids.empty()) return 0;
    auto start = Clock::now();
    ctx.decode(ids, next_position());
    started = true;
    active.insert(active.end(), ids.begin(), ids.end());
    return seconds(start);
}
Tokens Stream::discarded(int count) const {
    require(count > 0 && size_t(count) <= active.size(), "Invalid eviction count");
    return Tokens(active.begin(), active.begin() + count);
}
double Stream::evict(int count) {
    const int distance = position_at(count);
    require(!ctx.ram_rotation || (ctx.parked[1].shared_end == 0 && ctx.parked[4].shared_end == 0),
            "Cannot rebase history borrowed by a RAM branch");
    require(count > 0 && size_t(count) <= active.size(), "Invalid eviction count");
    auto start = Clock::now();
    require(llama_memory_attention_seq_rm(ctx.memory(), 0, prefix_capacity, prefix_capacity + distance), "Attention eviction failed");
    llama_memory_attention_seq_add(ctx.memory(), 0, prefix_capacity + distance, -1, -distance);
    if (ctx.model.hybrid) {
        // The recurrent tail must move even when *all* active attention cells
        // were evicted. Its tensor history is retained, its position is rebased.
        const int shift = size_t(count)==active.size()
            ? prefix_capacity-1-llama_memory_seq_pos_max(ctx.recurrent_memory(),0) : -distance;
        llama_memory_seq_add(ctx.recurrent_memory(), 0, prefix_capacity, -1, shift);
    }
    // Shifts are lazy: the following decode pays for GPU K rotation.
    erase_active(count);
    return seconds(start);
}
Generation Stream::query(const Tokens & prompt, int limit) { return ctx.generate(prompt, next_position(), limit); }
double Stream::replace_prefix_from(int sequence, const Tokens & ids, int evict_count) {
    require(!ctx.ram_rotation || (ctx.parked[1].shared_end == 0 && ctx.parked[4].shared_end == 0),
            "Cannot replace a prefix borrowed by a RAM branch");
    require(sequence == 2 && !ids.empty() && ids.size() <= size_t(prefix_limit), "Invalid prepared prefix");
    if(flexible) require(evict_count>0 && size_t(evict_count)<=active.size() &&
        ids.size()+active.size()-evict_count<=size_t(total_capacity), "Prepared prefix does not fit shared context");
    else require(evict_count==0, "Fixed layout evicts separately");
    const int distance = position_at(evict_count);
    const int native = ctx.physical(sequence);
    require(llama_memory_seq_pos_min(ctx.attention_memory(), native) == 0 &&
            llama_memory_seq_pos_max(ctx.attention_memory(), native) == int(ids.size()) - 1, "Prepared prefix positions mismatch");
    require(!ctx.model.hybrid || llama_memory_seq_pos_max(ctx.recurrent_memory(), native) == int(ids.size())-1,
            "Prepared prefix recurrent tail mismatch");
    auto start = Clock::now();
    llama_synchronize(ctx.ptr.get());
    require(llama_memory_attention_seq_rm(ctx.memory(), 0, 0, prefix_capacity), "Old prefix removal failed");
    if(flexible) {
        // Remove old P and the cut before moving the tail, so a growing or
        // shrinking prefix cannot alias either removal range.
        require(llama_memory_attention_seq_rm(ctx.memory(),0,prefix_capacity,prefix_capacity+distance),"Attention eviction failed");
        const int delta=int(ids.size())-prefix_capacity-distance;
        llama_memory_attention_seq_add(ctx.memory(),0,prefix_capacity+distance,-1,delta);
        if(ctx.model.hybrid) {
            const int shift = size_t(evict_count)==active.size()
                ? int(ids.size())-1-llama_memory_seq_pos_max(ctx.recurrent_memory(),0) : delta;
            llama_memory_seq_add(ctx.recurrent_memory(),0,prefix_capacity,-1,shift);
        }
        erase_active(evict_count);
        prefix_capacity=ids.size();active_capacity=total_capacity-prefix_capacity;
    }
    if (ctx.ram_rotation) {
        // Candidate becomes the independently appendable prefix. Its attention
        // is now shared with main; its recurrence stays separate in host memory.
        llama_memory_attention_seq_cp(ctx.memory(),native,0,0,ids.size());
        ctx.rename_branch(sequence,3);
        ctx.share_prefix(3,ids.size());
    } else if (ctx.model.hybrid) {
        ctx.remove(3);
        ctx.copy(sequence, 3);
        llama_memory_attention_seq_cp(ctx.memory(), sequence, 0, 0, ids.size());
    } else {
        ctx.copy(sequence, 0, 0, ids.size());
    }
    ctx.remove(sequence);
    prefix = ids;
    return seconds(start);
}
static const std::string summary_instruction =
    "Ты дополняешь память потоковой модели. На входе текущая память и удаляемый фрагмент. "
    "Выдай ТОЛЬКО новые существенные факты из удаляемого фрагмента, которых ещё нет в памяти. "
    "Старую память не переписывай и не пересказывай. Сохраняй имена, числа, коды и связи точно. "
    "Не придумывай сведения. Фоновые повторяющиеся заметки о погоде не сохраняй. "
    "Пиши краткими самостоятельными строками без вводных слов. Если дополнять нечего, "
    "выдай ровно NONE. Максимум 256 токенов.";
Summarizer::Summarizer(Model & model, const std::string & initial, int capacity, bool cached)
    : ctx(model, capacity), cached(cached) {
    prefix = model.tokens(message("system", summary_instruction) + "<|im_start|>user\n[Текущая память]\n" + initial);
    if (cached) ctx.decode(prefix, 0);
}
Generation Summarizer::summarize(const Tokens & discarded, int limit) {
    auto suffix = ctx.model.tokens("\n[Удаляемый фрагмент]\n" + ctx.model.text(discarded) +
                                 "\n[Дополнение]<|im_end|>\n" + ctx.model.generation_prompt());
    if (cached) return ctx.generate(suffix, prefix.size(), limit);
    Tokens all = prefix;
    all.insert(all.end(), suffix.begin(), suffix.end());
    return ctx.generate(all, 0, limit);
}
double Summarizer::append_memory(const std::string & addition) {
    if (addition.empty()) return 0;
    const auto ids = ctx.model.tokens("\n\nДополнение памяти:\n" + addition);
    auto start = Clock::now();
    if (cached) ctx.decode(ids, prefix.size());
    prefix.insert(prefix.end(), ids.begin(), ids.end());
    return seconds(start);
}
json compare_logits(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "Incompatible logits");
    double max_delta = 0, za = 0, zb = 0;
    double ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
    for (size_t i = 0; i < a.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(double(a[i]) - b[i]));
        za += std::exp(a[i] - ma); zb += std::exp(b[i] - mb);
    }
    double kl = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
        kl += std::exp(la) * (la - lb);
    }
    return {{"max_abs", max_delta}, {"kl", kl},
            {"top1_equal", std::max_element(a.begin(), a.end()) - a.begin() == std::max_element(b.begin(), b.end()) - b.begin()}};
}
} // namespace ring
