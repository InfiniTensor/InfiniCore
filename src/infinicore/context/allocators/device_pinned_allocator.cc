#include "device_pinned_allocator.hpp"

#include <infinirt.h>
#include <spdlog/spdlog.h>

#include "../../utils.hpp"

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator(Device device) : MemoryAllocator(), owner_(device) {}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    SPDLOG_DEBUG("[ALLOCATOR] ~DevicePinnedHostAllocator: START, owner device type={}, index={}",
                 static_cast<int>(owner_.getType()), owner_.getIndex());

    SPDLOG_DEBUG("[ALLOCATOR] ~DevicePinnedHostAllocator: Calling gc()");
    try {
        gc();
        SPDLOG_DEBUG("[ALLOCATOR] ~DevicePinnedHostAllocator: gc() completed successfully");
    } catch (...) {
        SPDLOG_WARN("[ALLOCATOR] ~DevicePinnedHostAllocator: WARNING - gc() threw exception");
    }

    SPDLOG_DEBUG("[ALLOCATOR] ~DevicePinnedHostAllocator: Complete");
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    void *ptr;
    INFINICORE_CHECK_ERROR(infinirtMallocHost(&ptr, size));
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return; // Nothing to free
    }
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    if (owner_ == context::getDevice()) {
        INFINICORE_CHECK_ERROR(infinirtFreeHost(ptr));
        // Process gc_queue_ inline while we have the lock
        while (gc_queue_.empty() == false) {
            std::byte *p = gc_queue_.front();
            INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
            gc_queue_.pop();
        }
    } else {
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    SPDLOG_DEBUG("[ALLOCATOR] DevicePinnedHostAllocator::gc(): START, queue size={}", gc_queue_.size());

    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        SPDLOG_DEBUG("[ALLOCATOR] DevicePinnedHostAllocator::gc(): Freeing pointer {}", static_cast<void *>(p));
        INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
        gc_queue_.pop();
    }

    SPDLOG_DEBUG("[ALLOCATOR] DevicePinnedHostAllocator::gc(): Complete");
}

} // namespace infinicore
