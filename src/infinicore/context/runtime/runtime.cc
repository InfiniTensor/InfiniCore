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

// Timing method implementations
infinirtEvent_t Runtime::createEvent() {
    infinirtEvent_t event;
    INFINICORE_CHECK_ERROR(infinirtEventCreate(&event));
    return event;
}

infinirtEvent_t Runtime::createEventWithFlags(uint32_t flags) {
    infinirtEvent_t event;
    INFINICORE_CHECK_ERROR(infinirtEventCreateWithFlags(&event, flags));
    return event;
}

void Runtime::recordEvent(infinirtEvent_t event, infinirtStream_t stream) {
    if (stream == nullptr) {
        stream = stream_;
    }
    INFINICORE_CHECK_ERROR(infinirtEventRecord(event, stream));
}

bool Runtime::queryEvent(infinirtEvent_t event) {
    infinirtEventStatus_t status;
    INFINICORE_CHECK_ERROR(infinirtEventQuery(event, &status));
    return status == INFINIRT_EVENT_COMPLETE;
}

void Runtime::synchronizeEvent(infinirtEvent_t event) {
    INFINICORE_CHECK_ERROR(infinirtEventSynchronize(event));
}

void Runtime::destroyEvent(infinirtEvent_t event) {
    INFINICORE_CHECK_ERROR(infinirtEventDestroy(event));
}

float Runtime::elapsedTime(infinirtEvent_t start, infinirtEvent_t end) {
    float ms;
    INFINICORE_CHECK_ERROR(infinirtEventElapsedTime(&ms, start, end));
    return ms;
}

void Runtime::streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    // Use current stream if no specific stream is provided
    if (stream == nullptr) {
        stream = stream_;
    }
    INFINICORE_CHECK_ERROR(infinirtStreamWaitEvent(stream, event));
}

std::string Runtime::toString() const {
    return fmt::format("Runtime({})", device_.toString());
}

} // namespace infinicore
