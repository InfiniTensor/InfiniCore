#pragma once

#include "memory_allocator.hpp"

#include "../context_impl.hpp"

namespace infinicore {
class DeviceCachingAllocator : public MemoryAllocator {
public:
    DeviceCachingAllocator() = default;
    ~DeviceCachingAllocator() = default;

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;
};

} // namespace infinicore
