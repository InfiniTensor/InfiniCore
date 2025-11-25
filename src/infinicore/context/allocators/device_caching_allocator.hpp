#pragma once

#include "memory_allocator.hpp"

#include "../context_impl.hpp"

#include <list>
#include <mutex>
#include <unordered_map>

namespace infinicore {

// Memory block information
struct MemoryBlock {
    void *ptr;
    size_t size;
    infinirtStream_t stream; // Stream this block was allocated on

    // Default constructor (required for std::unordered_map::operator[])
    MemoryBlock() : ptr(nullptr), size(0), stream(nullptr) {}

    MemoryBlock(void *p, size_t s, infinirtStream_t st)
        : ptr(p), size(s), stream(st) {}
};

class DeviceCachingAllocator : public MemoryAllocator {
public:
    explicit DeviceCachingAllocator(Device device);
    ~DeviceCachingAllocator();

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

private:
    Device device_;
    // Mutex to protect concurrent allocate/deallocate operations
    mutable std::mutex allocator_mutex_;

    // Memory pool/cache: size -> list of free blocks
    // Using list for O(1) removal from middle
    std::unordered_map<size_t, std::list<MemoryBlock>> free_blocks_;

    // Active blocks: pointer -> block info (for tracking)
    std::unordered_map<void *, MemoryBlock> active_blocks_;

    // Total cached memory size (for cleanup threshold)
    size_t cached_memory_size_;

    // Maximum cache size before cleanup (e.g., 512MB)
    static constexpr size_t MAX_CACHE_SIZE = 512 * 1024 * 1024;

    // Helper methods
    void cleanupCache();
    void returnToCache(void *ptr, size_t size, infinirtStream_t stream);
    bool tryGetFromCache(size_t size, MemoryBlock &out_block);
};

} // namespace infinicore
