#include "device_caching_allocator.hpp"

#include <infinirt.h>

#include "infinicore/common/utils.hpp"

namespace infinicore {
DeviceCachingAllocator::DeviceCachingAllocator(Device device) : MemoryAllocator(), device_(device) {}

std::byte *DeviceCachingAllocator::allocate(size_t size) {
    void *ptr = nullptr;
    INFINICORE_CHECK_ERROR(infinirtMallocAsync(&ptr, size, context::getStream()));
    return (std::byte *)ptr;
}

void DeviceCachingAllocator::deallocate(std::byte *ptr) {
    INFINICORE_CHECK_ERROR(infinirtFreeAsync(ptr, context::getStream()));
}
} // namespace infinicore
