#include "runtime.hpp"

#include "../../utils.hpp"

#include "../allocators/device_caching_allocator.hpp"
#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"

namespace infinicore {
Runtime::Runtime(Device device) : device_(device) {
    activate();
    INFINICORE_CHECK_ERROR(infinirtStreamCreate(&stream_));
    INFINICORE_CHECK_ERROR(infiniopCreateHandle(&infiniop_handle_));
    if (device_.getType() == Device::Type::CPU) {
        device_memory_allocator_ = std::make_unique<HostAllocator>();
    } else {
        device_memory_allocator_ = std::make_unique<DeviceCachingAllocator>(device);
        pinned_host_memory_allocator_ = std::make_unique<DevicePinnedHostAllocator>(device);
    }
}
Runtime::~Runtime() {
    activate();
    if (pinned_host_memory_allocator_) {
        pinned_host_memory_allocator_.reset();
    }
    device_memory_allocator_.reset();
    infiniopDestroyHandle(infiniop_handle_);
    infinirtStreamDestroy(stream_);
}

Runtime *Runtime::activate() {
    spdlog::debug("Runtime::activate() called for device={}", device_.toString());
    INFINICORE_CHECK_ERROR(infinirtSetDevice((infiniDevice_t)device_.getType(), (int)device_.getIndex()));
    return this;
}

Device Runtime::device() const {
    return device_;
}

infinirtStream_t Runtime::stream() const {
    return stream_;
}

infiniopHandle_t Runtime::infiniopHandle() const {
    return infiniop_handle_;
}

void Runtime::syncStream() {
    INFINICORE_CHECK_ERROR(infinirtStreamSynchronize(stream_));
}

void Runtime::syncDevice() {
    INFINICORE_CHECK_ERROR(infinirtDeviceSynchronize());
}

std::shared_ptr<Memory> Runtime::allocateMemory(size_t size) {
    std::byte *data_ptr = device_memory_allocator_->allocate(size);
    return std::make_shared<Memory>(
        data_ptr, size, device_,
        [alloc = device_memory_allocator_.get()](std::byte *p) {
            alloc->deallocate(p);
        });
}

std::shared_ptr<Memory> Runtime::allocatePinnedHostMemory(size_t size) {
    if (!pinned_host_memory_allocator_) {
        spdlog::warn("For CPU devices, pinned memory is not supported, falling back to regular host memory");
        return allocateMemory(size);
    }
    std::byte *data_ptr = pinned_host_memory_allocator_->allocate(size);
    return std::make_shared<Memory>(
        data_ptr, size, device_,
        [alloc = pinned_host_memory_allocator_.get()](std::byte *p) {
            alloc->deallocate(p);
        },
        true);
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size) {
    INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_H2D, stream_));
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    INFINICORE_CHECK_ERROR(infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2H));
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size) {
    INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_D2D, stream_));
}

MemoryAllocator *Runtime::getDeviceMemoryAllocator() const {
    return device_memory_allocator_.get();
}

MemoryAllocator *Runtime::getPinnedHostMemoryAllocator() const {
    return pinned_host_memory_allocator_.get();
}

std::string Runtime::toString() const {
    return fmt::format("Runtime({})", device_.toString());
}

} // namespace infinicore
