#include "runtime.hpp"
#include "../allocators/device_caching_allocator.hpp"
#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"

namespace infinicore {
Runtime::Runtime(Device device) : device_(device) {
    activate();
    infinirtStreamCreate(&stream_);
    infiniopCreateHandle(&infiniop_handle_);
    if (device_.getType() == Device::Type::CPU) {
        device_memory_allocator_ = std::make_unique<HostAllocator>();
    } else {
        device_memory_allocator_ = std::make_unique<DeviceCachingAllocator>();
        pinned_host_memory_allocator_ = std::make_unique<DevicePinnedHostAllocator>();
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
    infinirtSetDevice((infiniDevice_t)device_.getType(), (int)device_.getIndex());
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
    infinirtStreamSynchronize(stream_);
}

void Runtime::syncDevice() {
    infinirtDeviceSynchronize();
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
    std::byte *data_ptr = pinned_host_memory_allocator_->allocate(size);
    return std::make_shared<Memory>(
        data_ptr, size, device_,
        [alloc = pinned_host_memory_allocator_.get()](std::byte *p) {
            alloc->deallocate(p);
        });
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size) {
    infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_H2D);
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2H);
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size) {
    infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2D);
}

} // namespace infinicore
