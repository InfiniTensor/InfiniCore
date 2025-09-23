#include "device_pinned_allocator.hpp"

#include <infinirt.h>

#include "infinicore/common/utils.hpp"

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator() : MemoryAllocator() {
    owner_ = context::getDevice();
}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    gc();
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    void *ptr;
    INFINICORE_CHECK_ERROR(infinirtMallocHost(&ptr, size));
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (owner_ == context::getDevice()) {
        INFINICORE_CHECK_ERROR(infinirtFreeHost(ptr));
        gc();
    } else {
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
        gc_queue_.pop();
    }
}

} // namespace infinicore
