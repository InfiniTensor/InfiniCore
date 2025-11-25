#pragma once

#include "memory_allocator.hpp"

#include "../context_impl.hpp"

#include <mutex>
#include <queue>

namespace infinicore {
class DevicePinnedHostAllocator : public MemoryAllocator {
public:
    explicit DevicePinnedHostAllocator(Device device);
    ~DevicePinnedHostAllocator();

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

    void gc();

private:
    Device owner_;

    // Thread-safe queue for deferred deallocation
    std::queue<std::byte *> gc_queue_;
    // Mutex to protect concurrent allocate/deallocate/gc operations
    mutable std::mutex allocator_mutex_;
};

} // namespace infinicore
