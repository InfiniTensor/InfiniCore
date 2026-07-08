#include "runtime.hpp"
#include "../../../bridge/infini/rt.hpp"

#include "../../utils.hpp"

#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"
#include "../allocators/pinnable_block_allocator.hpp"
#include "../allocators/stream_ordered_allocator.hpp"

namespace infinicore {
Runtime::Runtime(Device device) : device_(device), graph_manager_(std::make_unique<graph::GraphManager>()) {
    activate();
    infini::rt::runtime::Stream stream = nullptr;
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::StreamCreate(&stream)));
    stream_ = bridge::infini::rt::to_core_stream(stream);
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
    (void)infini::rt::runtime::StreamDestroy(bridge::infini::rt::to_rt_stream(stream_));
}

Runtime *Runtime::activate() {
    auto rt_device = bridge::infini::rt::translate(static_cast<infiniDevice_t>(device_.getType()));
    INFINICORE_ASSERT(rt_device != infini::rt::Device::Type::kCount);
    infini::rt::set_runtime_device_type(rt_device);
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::SetDevice(static_cast<int>(device_.getIndex()))));
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
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::StreamSynchronize(bridge::infini::rt::to_rt_stream(stream_))));
}

void Runtime::syncDevice() {
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::DeviceSynchronize()));
}

void Runtime::trimMemory() {
    device_memory_allocator_->trim();
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
    std::lock_guard<std::mutex> lock(reinstantiated_blob_mutex_);

    auto ptr = blob->data();
    auto it = reinstantiated_blobs_.find(ptr);
    if (it != reinstantiated_blobs_.end()) {
        if (auto memory = it->second.lock()) {
            return memory;
        }
    }

    device_memory_allocator_.get()->mark_in_use_(ptr, true);
    auto memory = std::make_shared<Memory>(
        blob->data(), blob->size(), device_,
        [alloc = device_memory_allocator_.get()](std::byte *p) {
            alloc->deallocate(p);
        });
    reinstantiated_blobs_[ptr] = memory;
    return memory;
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    if (async && device_.getType() != Device::Type::CPU) {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice, bridge::infini::rt::to_rt_stream(stream_))));
    } else {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice)));
    }
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToHost)));
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    if (async && device_.getType() != Device::Type::CPU) {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice, bridge::infini::rt::to_rt_stream(stream_))));
    } else {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice)));
    }
}

void Runtime::setDeviceMemory(void *ptr, int value, size_t count) {
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Memset(ptr, value, count)));
}

void Runtime::setDeviceMemoryAsync(void *ptr, int value, size_t count, infinirtStream_t stream) {
    if (device_.getType() != Device::Type::CPU) {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::MemsetAsync(ptr, value, count, bridge::infini::rt::to_rt_stream(stream))));
    } else {
        INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::Memset(ptr, value, count)));
    }
}

// Timing method implementations
infinirtEvent_t Runtime::createEvent() {
    infinirtEvent_t event;
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventCreate(&event)));
    return event;
}

infinirtEvent_t Runtime::createEventWithFlags(uint32_t flags) {
    infinirtEvent_t event;
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventCreateWithFlags(&event, flags)));
    return event;
}

void Runtime::recordEvent(infinirtEvent_t event, infinirtStream_t stream) {
    if (stream == nullptr) {
        stream = stream_;
    }
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventRecord(event, bridge::infini::rt::to_rt_stream(stream))));
}

bool Runtime::queryEvent(infinirtEvent_t event) {
    return bridge::infini::rt::translate(infini::rt::runtime::EventQuery(event)) == INFINI_STATUS_SUCCESS;
}

void Runtime::synchronizeEvent(infinirtEvent_t event) {
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventSynchronize(event)));
}

void Runtime::destroyEvent(infinirtEvent_t event) {
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventDestroy(event)));
}

float Runtime::elapsedTime(infinirtEvent_t start, infinirtEvent_t end) {
    float ms;
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::EventElapsedTime(&ms, start, end)));
    return ms;
}

void Runtime::streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    // Use current stream if no specific stream is provided
    if (stream == nullptr) {
        stream = stream_;
    }
    INFINICORE_CHECK_ERROR(bridge::infini::rt::translate(infini::rt::runtime::StreamWaitEvent(bridge::infini::rt::to_rt_stream(stream), event, 0)));
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
    auto graph = graph_manager_->stop_recording();
    device_memory_allocator_->set_pin_mode(false);
    return graph;
}

std::string Runtime::toString() const {
    return fmt::format("Runtime({})", device_.toString());
}

} // namespace infinicore
