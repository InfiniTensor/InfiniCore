#include "device_caching_allocator.hpp"

#include <infinirt.h>

namespace infinicore {
std::byte *DeviceCachingAllocator::allocate(size_t size) {
    void *ptr = nullptr;
    infinirtMallocAsync(&ptr, size, context::getStream());
    return (std::byte *)ptr;
}

void DeviceCachingAllocator::deallocate(std::byte *ptr) {
    infinirtFreeAsync(ptr, context::getStream());
}
} // namespace infinicore
