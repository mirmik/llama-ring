#include "checkpoint.hpp"
#include <array>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ring {
namespace {
constexpr size_t header_size = 40;
constexpr uint64_t magic = 0x3154504b43534356ULL; // VCSCKPT1, native little endian
// Accidental-corruption checksum, not an authentication mechanism. Files are local/private.
uint64_t checksum(const uint8_t * p, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for(size_t i=0; i<n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
uint64_t read64(const uint8_t * p) { uint64_t v; std::memcpy(&v,p,8); return v; }
void write64(uint8_t * p, uint64_t v) { std::memcpy(p,&v,8); }
}
std::filesystem::path checkpoint_path(const std::string & directory, const std::string & name) {
    require(!directory.empty(), "Slot persistence disabled: set --slot-save-path");
    require(!name.empty() && name.size() <= 160 && name != "." && name != ".." && name.front() != '.', "Invalid checkpoint filename");
    for(unsigned char c:name) require((c>='a'&&c<='z') || (c>='A'&&c<='Z') || (c>='0'&&c<='9') || c=='_' || c=='-' || c=='.', "Invalid checkpoint filename");
    return std::filesystem::path(directory)/name;
}
uint64_t save_checkpoint(Context & ctx, const std::filesystem::path & path, const json & metadata) {
    auto meta = json::to_cbor(metadata);
    llama_synchronize(ctx.ptr.get());
    const size_t state_size = llama_state_get_size(ctx.ptr.get());
    const size_t size = header_size + meta.size() + state_size;
    std::string pattern = path.string()+".tmp.XXXXXX";
    std::vector<char> tmp(pattern.begin(),pattern.end());tmp.push_back(0);
    const int fd=mkstemp(tmp.data());require(fd>=0,"Cannot create checkpoint temporary file");
    uint8_t * mapped=nullptr;
    try {
        require(posix_fallocate(fd,0,size)==0,"Cannot reserve checkpoint disk space");
        void * address=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        require(address!=MAP_FAILED,"Cannot map checkpoint file");mapped=static_cast<uint8_t *>(address);
        write64(mapped,magic);write64(mapped+8,meta.size());write64(mapped+16,state_size);
        std::memcpy(mapped+header_size,meta.data(),meta.size());
        require(llama_state_get_data(ctx.ptr.get(),mapped+header_size+meta.size(),state_size)==state_size,"Cannot export full context");
        write64(mapped+24,checksum(mapped+header_size,size-header_size));write64(mapped+32,0);
        require(msync(mapped,size,MS_SYNC)==0 && fsync(fd)==0,"Cannot flush checkpoint");
        munmap(mapped,size);mapped=nullptr;
        require(rename(tmp.data(),path.c_str())==0,"Cannot commit checkpoint");
        const int parent=open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
        require(parent>=0,"Cannot open checkpoint directory");
        const int flushed=fsync(parent);close(parent);require(flushed==0,"Cannot flush checkpoint directory");
        close(fd);return size;
    } catch(...) { if(mapped) munmap(mapped,size);close(fd);unlink(tmp.data());throw; }
}
Checkpoint::Checkpoint(const std::filesystem::path & path) {
    try {
        fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW);require(fd>=0,"Checkpoint not found or unreadable");
        struct stat st{};require(fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_size>=int64_t(header_size),"Invalid checkpoint file");
        size=st.st_size;
        void * address=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);require(address!=MAP_FAILED,"Cannot map checkpoint");
        data=static_cast<const uint8_t *>(address);
        require(read64(data)==magic && read64(data+32)==0,"Incompatible checkpoint format");
        uint64_t meta_size=read64(data+8);state_size=read64(data+16);
        require(meta_size<=1024ULL*1024*1024 && meta_size<=size-header_size && state_size==size-header_size-meta_size,"Invalid checkpoint sizes");
        require(read64(data+24)==checksum(data+header_size,size-header_size),"Checkpoint checksum mismatch");
        offset=header_size+meta_size;
        metadata=json::from_cbor(data+header_size,data+offset);
    } catch(...) { if(data) munmap(const_cast<uint8_t *>(data),size);if(fd>=0) close(fd);throw; }
}
Checkpoint::~Checkpoint() { if(data) munmap(const_cast<uint8_t *>(data),size);if(fd>=0) close(fd); }
void Checkpoint::restore(Context & ctx) const {
    require(llama_state_set_data(ctx.ptr.get(),data+offset,state_size)==state_size,"Cannot restore full context");
}
}
