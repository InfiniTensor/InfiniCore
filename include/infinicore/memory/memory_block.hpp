#pragma once

#include "../device.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace infinicore {

class MemoryBlock {
public:
    using Deleter = std::function<void(std::byte *)>;

    MemoryBlock(std::byte *data, size_t size, Device device, Deleter deleter, bool pin_memory = false);
    ~MemoryBlock();

    // Copy constructor and copy assignment with reference counting
    MemoryBlock(const MemoryBlock& other);
    MemoryBlock& operator=(const MemoryBlock& other);

    // Move constructor and move assignment
    MemoryBlock(MemoryBlock&& other) noexcept;
    MemoryBlock& operator=(MemoryBlock&& other) noexcept;

    std::byte *data() const;
    Device device() const;
    size_t size() const;
    bool is_pinned() const;

private:
    std::byte *data_;
    size_t size_;
    Device device_;
    Deleter deleter_;
    bool is_pinned_;
};

} // namespace infinicore
