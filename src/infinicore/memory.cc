#include "infinicore/memory.hpp"

namespace infinicore {

Memory::Memory(std::byte *data,
               size_t size,
               Device device,
               Memory::Deleter deleter,
               bool pin_memory)
    : data_{data}, size_{size}, device_{device}, deleter_{deleter}, is_pinned_(pin_memory) {}

Memory::~Memory() {
    if (!deleter_) {
        return;
    }
    try {
        deleter_(data_);
    } catch (...) {
        // Memory can be released during interpreter/static shutdown after
        // allocator metadata has already been torn down. Destructors must not
        // let allocator cleanup errors escape and terminate the process.
    }
}

std::byte *Memory::data() {
    return data_;
}

Device Memory::device() const {
    return device_;
}

size_t Memory::size() const {
    return size_;
}

bool Memory::is_pinned() const {
    return is_pinned_;
}
} // namespace infinicore
