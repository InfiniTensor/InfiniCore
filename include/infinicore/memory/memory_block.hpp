#pragma once

#include "../device.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace infinicore {

class Memory {
public:
    using Deleter = std::function<void(std::byte *)>;

    Memory(std::byte *data, size_t size, Device device, Deleter deleter, bool pin_memory = false);
    ~Memory();

    // Copy constructor and copy assignment with reference counting
    Memory(const Memory& other);
    Memory& operator=(const Memory& other);

    // Move constructor and move assignment
    Memory(Memory&& other) noexcept;
    Memory& operator=(Memory&& other) noexcept;

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
