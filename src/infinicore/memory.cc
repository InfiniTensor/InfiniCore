#include "infinicore/memory.hpp"

namespace infinicore {

Memory::Memory(std::byte *data,
               size_t size,
               Device device,
               Memory::Deleter deleter)
    : data_{data}, size_{size}, device_{device_}, deleter_{deleter} {}

std::byte *Memory::data() {
    return data_;
}

Device Memory::device() const {
    return device_;
}

size_t Memory::size() const {
    return size_;
}

} // namespace infinicore
