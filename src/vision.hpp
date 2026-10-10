#pragma once
#include "server_session.hpp"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace ring {
struct VisionInput {
    json messages;
    std::vector<mtmd_helper::video_ptr> videos;
    std::vector<std::unique_ptr<mtmd_bitmap,decltype(&mtmd_bitmap_free)>> bitmaps;
    std::vector<std::string> hashes;
};
struct VisionPrompt {
    std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)> chunks{nullptr,mtmd_input_chunks_free};
    Tokens tokens;
};
class Vision {
public:
    Model & model;
    std::unique_ptr<mtmd_context,decltype(&mtmd_free)> ctx{nullptr,mtmd_free};
    Vision(Model & model, const std::string & projector, bool gpu, int max_tokens);
    VisionInput input(const json & messages);
    VisionPrompt tokenize(const std::string & text, const VisionInput & input);
    void append(Session & session, const VisionPrompt & prompt, size_t skip, const std::function<void()> & check);
};
}
