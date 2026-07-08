#include "stream_ordered_allocator.hpp"

#include "../../../bridge/infini/rt.hpp"

#include "../../utils.hpp"

namespace infinicore {
StreamOrderedAllocator::StreamOrderedAllocator(Device device) : MemoryAllocator(), device_(device) {}

std::byte *StreamOrderedAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (device_.getType() != Device::Type::CPU) {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::MallocAsync(
            &ptr,
            size,
            bridge::infini::rt::to_rt_stream(context::getStream()))));
    } else {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Malloc(&ptr, size)));
    }
    return (std::byte *)ptr;
}

void StreamOrderedAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (device_.getType() != Device::Type::CPU) {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::FreeAsync(
            ptr,
            bridge::infini::rt::to_rt_stream(context::getStream()))));
    } else {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Free(ptr)));
    }
}
} // namespace infinicore
