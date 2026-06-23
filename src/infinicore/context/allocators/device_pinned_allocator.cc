#include "device_pinned_allocator.hpp"

#include <infinirt.h>

#include "../../utils.hpp"

#include <cstdint>

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator(Device device) : MemoryAllocator(), owner_(device) {}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    gc();
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    const Device active = context::getDevice();
    if (owner_ != active) {
        context::setDevice(owner_);
    }
    void *ptr;
    INFINICORE_CHECK_ERROR(infinirtMallocHost(&ptr, size));
    if (owner_ != active) {
        context::setDevice(active);
    }
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    const Device active = context::getDevice();
    if (owner_ != active) {
        context::setDevice(owner_);
    }
    INFINICORE_CHECK_ERROR(infinirtFreeHost(ptr));
    gc();
    if (owner_ != active) {
        context::setDevice(active);
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
