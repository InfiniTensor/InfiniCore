#include "runtime.hpp"

#include "../../utils.hpp"

#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"
#include "../allocators/pinnable_block_allocator.hpp"
#include "../allocators/stream_ordered_allocator.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <infinirt.h>

namespace infinicore {
Runtime::Runtime(Device device) : device_(device), graph_manager_(std::make_unique<graph::GraphManager>()) {
    activate();
    INFINICORE_CHECK_ERROR(infinirtStreamCreate(&stream_));
    INFINICORE_CHECK_ERROR(infiniopCreateHandle(&infiniop_handle_));
    if (device_.getType() == Device::Type::CPU) {
        device_memory_allocator_ = std::make_unique<PinnableBlockAllocator>(device);
    } else {
        device_memory_allocator_ = std::make_unique<PinnableBlockAllocator>(device);
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

void Runtime::trimDeviceMemory() {
    syncDevice();
    if (isGraphRecording()) {
        return;
    }
    if (device_memory_allocator_) {
        device_memory_allocator_->trim();
    }
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

std::shared_ptr<Memory> Runtime::reinstantiateBlob(std::shared_ptr<Memory> blob) {
    const auto &device = blob->device();
    if (device_memory_allocator_) {
        try {
            device_memory_allocator_.get()->mark_in_use_(blob->data(), true);
        } catch (const std::runtime_error &) {
            // External memory (e.g. from_torch): alias without allocator tracking.
            return std::make_shared<Memory>(blob->data(), blob->size(), device, nullptr);
        }
    }
    // Non-owning view: the source blob keeps the deleter; graph replay must not
    // double-free the same pinned allocation (see PinnableBlockAllocator).
    return std::make_shared<Memory>(blob->data(), blob->size(), device, nullptr);
}

void Runtime::flushDeferredPinnedHostFrees() {
    if (pinned_host_memory_allocator_) {
        static_cast<DevicePinnedHostAllocator *>(pinned_host_memory_allocator_.get())->gc();
    }
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    std::shared_ptr<Memory> staging;
    const void *h2d_src = src;
    // HPCC 3.1: host pages touched by H2D become pinned on the active GPU node.
    // Stage arbitrary host pointers through rank-local pinned memory so TP ranks
    // never DMA from pages registered on another node. Sync before releasing staging.
    if (size > 0 && pinned_host_memory_allocator_) {
        staging = allocatePinnedHostMemory(size);
        std::memcpy(staging->data(), src, size);
        h2d_src = staging->data();
    }
    if (async) {
        INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(dst, h2d_src, size, INFINIRT_MEMCPY_H2D, stream_));
        if (staging) {
            syncStream();
        }
    } else {
        INFINICORE_CHECK_ERROR(infinirtMemcpy(dst, h2d_src, size, INFINIRT_MEMCPY_H2D));
    }
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    INFINICORE_CHECK_ERROR(infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2H));
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    if (async) {
        INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_D2D, stream_));
    } else {
        INFINICORE_CHECK_ERROR(infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2D));
    }
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

bool Runtime::isGraphRecording() const {
    return graph_manager_->is_recording();
}

void Runtime::startGraphRecording() {
    device_memory_allocator_->set_pin_mode(true);
    return graph_manager_->start_recording();
}

void Runtime::addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    return graph_manager_->add_operator(op);
}

std::shared_ptr<graph::Graph> Runtime::stopGraphRecording() {
    // End pin_mode before instantiate warmup/replay; plan() already captured GraphTensor blobs.
    // capture_device_segment_ re-enables pin via CaptureArena for segment temps.
    device_memory_allocator_->set_pin_mode(false);
    auto graph = graph_manager_->stop_recording();
    return graph;
}

void Runtime::setDeviceAllocatorPinMode(bool pinned) {
    if (device_memory_allocator_) {
        device_memory_allocator_->set_pin_mode(pinned);
    }
}

std::string Runtime::toString() const {
    return fmt::format("Runtime({})", device_.toString());
}

} // namespace infinicore
