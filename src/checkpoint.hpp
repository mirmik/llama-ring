#pragma once
#include "runtime.hpp"
#include <filesystem>
#include <functional>

namespace ring {
// Single atomic file: bounded CBOR metadata plus native complete llama state.
// mmap keeps large KV snapshots out of an additional heap-sized buffer.
uint64_t save_checkpoint(Context & ctx, const std::filesystem::path & path, const json & metadata);
struct Checkpoint {
    int fd = -1;
    const uint8_t * data = nullptr;
    size_t size = 0, offset = 0, state_size = 0;
    json metadata;
    explicit Checkpoint(const std::filesystem::path & path);
    ~Checkpoint();
    Checkpoint(const Checkpoint &) = delete;
    void restore(Context & ctx) const;
};
std::filesystem::path checkpoint_path(const std::string & directory, const std::string & filename);
}
