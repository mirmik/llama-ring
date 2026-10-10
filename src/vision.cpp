#include "vision.hpp"
#include "mtmd-helper.h"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace ring {
Vision::Vision(Model & m, const std::string & projector, bool gpu, int max_tokens) : model(m) {
    if(projector.empty()) return;
    require(model.architecture == "qwen35" || model.architecture == "qwen4exp","Vision requires qwen35 or qwen4exp");
    require(max_tokens==-1 || (max_tokens>=64 && max_tokens<=32768),"image-max-tokens must be -1 (projector default) or 64..32768");
    auto params=mtmd_context_params_default();
    params.use_gpu=gpu;params.n_threads=8;params.warmup=false;
    // Preserve the projector's resolution limits unless explicitly overridden.
    if(max_tokens>0) {params.image_min_tokens=64;params.image_max_tokens=max_tokens;}
    ctx.reset(mtmd_init_from_file(projector.c_str(),model.ptr.get(),params));
    require(bool(ctx) && mtmd_support_vision(ctx.get()) && mtmd_decode_use_mrope(ctx.get()),"Incompatible vision projector");
}
VisionInput Vision::input(const json & messages) {
    VisionInput out;out.messages=messages;
    for(auto & msg:out.messages) {
        if(!msg.contains("content") || !msg["content"].is_array()) continue;
        for(auto & part:msg["content"]) {
            const auto type=part.value("type",std::string());
            if(type!="image_url" && type!="input_video") continue;
            require(bool(ctx),"Vision disabled: configure --mmproj");
            require(msg.at("role")=="user","Images are supported in user messages only");
            const auto url=type=="image_url"?part.at("image_url").at("url").get<std::string>():part.at("input_video").at("data").get<std::string>();
            const auto comma=url.find(',');
            require(type=="input_video" || (comma!=std::string::npos && url.rfind("data:image/",0)==0 && url.substr(0,comma).find(";base64")!=std::string::npos),"Use inline base64 media");
            const std::string encoded=comma==std::string::npos?url:url.substr(comma+1);
            require(!encoded.empty() && encoded.size()%4==0 && encoded.size()<=12*1024*1024,"Invalid image size/base64");
            std::vector<unsigned char> bytes(encoded.size()/4*3);
            int count=EVP_DecodeBlock(bytes.data(),reinterpret_cast<const unsigned char *>(encoded.data()),encoded.size());
            require(count>0,"Invalid image base64");
            if(encoded.back()=='=') --count;
            if(encoded.size()>1 && encoded[encoded.size()-2]=='=') --count;
            bytes.resize(count);
            unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(bytes.data(),bytes.size(),digest);
            std::ostringstream hash;for(auto c:digest) hash<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
            out.hashes.push_back(hash.str());
            auto bitmap=mtmd_helper_bitmap_init_from_buf(ctx.get(),bytes.data(),bytes.size(),false,mtmd_helper_init_opt_default());
            if(bitmap.video_ctx) out.videos.emplace_back(bitmap.video_ctx);
            require(bitmap.bitmap,"Cannot decode image/video");
            out.bitmaps.emplace_back(bitmap.bitmap,mtmd_bitmap_free);
            require(type=="input_video" || !bitmap.video_ctx,"Expected an image, received a video");
            require(!mtmd_bitmap_is_audio(bitmap.bitmap),"Audio is unsupported");
            part={{"type","media_marker"},{"text",mtmd_get_marker(ctx.get())}};
        }
    }
    return out;
}
VisionPrompt Vision::tokenize(const std::string & text, const VisionInput & input) {
    VisionPrompt out;
    if(input.bitmaps.empty()) {out.tokens=model.tokens(text);return out;}
    out.chunks.reset(mtmd_input_chunks_init());
    std::vector<const mtmd_bitmap *> bitmaps;for(auto & b:input.bitmaps) bitmaps.push_back(b.get());
    mtmd_input_text prompt{text.c_str(),text.size(),false,true};
    require(mtmd_tokenize(ctx.get(),out.chunks.get(),&prompt,bitmaps.data(),bitmaps.size())==0,"Image prompt tokenization failed");
    const auto pad=model.tokens("<|image_pad|>");
    require(pad.size()==1,"Missing image placeholder token");
    for(size_t i=0;i<mtmd_input_chunks_size(out.chunks.get());++i) {
        auto chunk=mtmd_input_chunks_get(out.chunks.get(),i);
        if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t count=0;auto ids=mtmd_input_chunk_get_tokens_text(chunk,&count);
            out.tokens.insert(out.tokens.end(),ids,ids+count);
        } else {
            require(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE,"Only images are supported");
            out.tokens.insert(out.tokens.end(),mtmd_input_chunk_get_n_tokens(chunk),pad.front());
        }
    }
    return out;
}
void Vision::append(Session & s, const VisionPrompt & prompt, size_t skip, const std::function<void()> & check) {
    if(!prompt.chunks) {s.append(Tokens(prompt.tokens.begin()+skip,prompt.tokens.end()),check);return;}
    size_t offset=0;
    for(size_t i=0;i<mtmd_input_chunks_size(prompt.chunks.get());++i) {
        auto chunk=mtmd_input_chunks_get(prompt.chunks.get(),i);
        const auto count=mtmd_input_chunk_get_n_tokens(chunk);
        if(offset+count<=skip) {offset+=count;continue;}
        if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_TEXT) {
            s.append(Tokens(prompt.tokens.begin()+std::max(offset,skip),prompt.tokens.begin()+offset+count),check);
        } else {
            require(skip<=offset,"Cannot resume inside an image");
            auto & stream=s.stream;
            require(count>0 && count+stream.prefix.size()<=size_t(stream.total_capacity),"Image exceeds working context");
            // An image is atomic: finish the old shadow before making room, never cut its patch grid.
            while(stream.active.size()+count>size_t(stream.active_capacity)) {
                require(!stream.active.empty(),"Image does not fit after compaction");
                s.shadow.start_if_due(true);s.commit(check,count);
            }
            check();
            require(mtmd_encode_chunk(ctx.get(),chunk)==0,"Image encoding failed");
            check();
            if(stream.ctx.ram_rotation) stream.ctx.park();
            const int position=stream.next_position();llama_pos end=position;
            auto checkpoint=[](const mtmd_helper_embd_batch *,void * opaque)->int32_t {
                (*static_cast<const std::function<void()> *>(opaque))();return 0;
            };
            require(mtmd_helper_decode_image_chunk(ctx.get(),stream.ctx.ptr.get(),chunk,mtmd_get_output_embd(ctx.get()),
                position,0,stream.ctx.batch_size,&end,checkpoint,const_cast<std::function<void()> *>(&check))==0,"Image decode failed");
            llama_synchronize(stream.ctx.ptr.get());
            require(end-position>0 && size_t(end-position)<=count,"Unsupported image position extent");
            stream.images.push_back({int(stream.active.size()),int(count),end-position});
            stream.active.insert(stream.active.end(),prompt.tokens.begin()+offset,prompt.tokens.begin()+offset+count);
            stream.started=true;s.shadow.start_if_due();
        }
        offset+=count;
    }
}
}
