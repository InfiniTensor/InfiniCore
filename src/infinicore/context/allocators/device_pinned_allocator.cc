#include "device_pinned_allocator.hpp"

#include <infinirt.h>

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator() : MemoryAllocator() {
    owner_ = context::getDevice();
}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    gc();
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    void *ptr;
    infinirtMallocHost(&ptr, size);
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (owner_ == context::getDevice()) {
        infinirtFreeHost(ptr);
        gc();
    } else {
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        infinirtFreeHost(p);
        gc_queue_.pop();
    }
}

} // namespace infinicore
