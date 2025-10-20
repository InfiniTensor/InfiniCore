#include "device_pinned_allocator.hpp"

#include <infinirt.h>

#include "../../utils.hpp"

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator(Device device) : MemoryAllocator(), owner_(device) {}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    gc();
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    void *ptr;

    // Update statistics before allocation
    stats_.allocation[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.requested_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);

    INFINICORE_CHECK_ERROR(infinirtMallocHost(&ptr, size));

    // Update statistics after successful allocation
    stats_.segment[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.active[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.num_device_alloc++;

    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    // Update statistics before deallocation
    stats_.active[static_cast<size_t>(StatType::AGGREGATE)].decrease(1);
    // Note: We don't know the exact size being deallocated here, so we can't update
    // active_bytes, allocated_bytes, etc. This is a limitation of the current design.

    if (owner_ == context::getDevice()) {
        INFINICORE_CHECK_ERROR(infinirtFreeHost(ptr));
        stats_.num_device_free++;
        gc();
    } else {
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
        stats_.num_device_free++;
        gc_queue_.pop();
    }
}

} // namespace infinicore
