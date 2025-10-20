#pragma once

#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <functional>

namespace infinicore {

struct MemoryInfo {
    std::byte* ptr;
    size_t size;
    std::atomic<int> ref_count;
    bool is_freed;

    MemoryInfo(std::byte* p, size_t s)
        : ptr(p), size(s), ref_count(1), is_freed(false) {}
};

class MemoryPool {
public:
    static MemoryPool& instance();

    // Register a memory allocation
    void registerMemory(std::byte* ptr, size_t size);

    // Increment reference count
    void addRef(std::byte* ptr);

    // Decrement reference count and potentially free memory
    void releaseMemory(std::byte* ptr, std::function<void(std::byte*)> actual_deleter);

    // Get reference count
    int getRefCount(std::byte* ptr) const;

    // Check if memory is registered
    bool isRegistered(std::byte* ptr) const;

    // Check if memory is already freed
    bool isFreed(std::byte* ptr) const;

private:
    MemoryPool() = default;
    ~MemoryPool() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::byte*, std::shared_ptr<MemoryInfo>> memory_map_;
};

} // namespace infinicore
