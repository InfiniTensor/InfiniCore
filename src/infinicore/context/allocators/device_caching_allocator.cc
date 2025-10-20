#include "device_caching_allocator.hpp"

#include <infinirt.h>

#include "../../utils.hpp"

namespace infinicore {
DeviceCachingAllocator::DeviceCachingAllocator(Device device) : MemoryAllocator(), device_(device) {}

std::byte *DeviceCachingAllocator::allocate(size_t size) {
    void *ptr = nullptr;
    spdlog::debug("DeviceCachingAllocator::allocate() called for size={}", size);

    // Update statistics before allocation
    stats_.allocation[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.requested_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);

    INFINICORE_CHECK_ERROR(infinirtMallocAsync(&ptr, size, context::getStream()));

    // Update statistics after successful allocation
    stats_.segment[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.active[static_cast<size_t>(StatType::AGGREGATE)].increase(1);
    stats_.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].increase(size);
    stats_.num_device_alloc++;

    spdlog::debug("DeviceCachingAllocator::allocate() returned memory={}", static_cast<void *>(ptr));
    return (std::byte *)ptr;
}

void DeviceCachingAllocator::deallocate(std::byte *ptr) {
    spdlog::debug("DeviceCachingAllocator::deallocate() called for memory={}", static_cast<void *>(ptr));

    // Update statistics before deallocation
    stats_.active[static_cast<size_t>(StatType::AGGREGATE)].decrease(1);
    // Note: We don't know the exact size being deallocated here, so we can't update
    // active_bytes, allocated_bytes, etc. This is a limitation of the current design.
    // In a more sophisticated implementation, we would track the size of each allocation.

    INFINICORE_CHECK_ERROR(infinirtFreeAsync(ptr, context::getStream()));

    // Update statistics after successful deallocation
    stats_.num_device_free++;

    spdlog::debug("DeviceCachingAllocator::deallocate() returned");
}
} // namespace infinicore
