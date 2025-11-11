#include "context_impl.hpp"

#include "../utils.hpp"

namespace infinicore {

thread_local Runtime *ContextImpl::current_runtime_ = nullptr;

Runtime *ContextImpl::getCurrentRuntime() {
    if (current_runtime_ == nullptr) {
        spdlog::debug("current_runtime_ is null, performing lazy initialization");
        // Lazy initialization: use the first available runtime
        // Try to find the first non-CPU device, fallback to CPU
        for (int i = int(Device::Type::COUNT) - 1; i > 0; i--) {
            if (!runtime_table_[i].empty() && runtime_table_[i][0] != nullptr) {
                current_runtime_ = runtime_table_[i][0].get();
                spdlog::debug("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
                return current_runtime_;
            }
        }
        // Fallback to CPU runtime
        if (!runtime_table_[0].empty() && runtime_table_[0][0] != nullptr) {
            current_runtime_ = runtime_table_[0][0].get();
            spdlog::debug("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
        }
    } else {
        spdlog::debug("getCurrentRuntime() returning {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
    }
    return current_runtime_;
}

Runtime *ContextImpl::getCpuRuntime() {
    return runtime_table_[int(Device::Type::CPU)][0].get();
}

void ContextImpl::setDevice(Device device) {
    if (device == getCurrentRuntime()->device()) {
        // Do nothing if the device is already set.
        return;
    }

    if (runtime_table_[int(device.getType())][device.getIndex()] == nullptr) {
        // Lazy initialization of runtime if never set before.
        runtime_table_[int(device.getType())][device.getIndex()] = std::unique_ptr<Runtime>(new Runtime(device));
        current_runtime_ = runtime_table_[int(device.getType())][device.getIndex()].get();
    } else {
        current_runtime_ = runtime_table_[int(device.getType())][device.getIndex()].get()->activate();
    }
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    return runtime_table_[int(type)].size();
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

ContextImpl::ContextImpl() {
    std::vector<int> device_counter(static_cast<size_t>(Device::Type::COUNT));
    INFINICORE_CHECK_ERROR(infinirtGetAllDeviceCount(device_counter.data()));

    // Reserve runtime slot for all devices.
    runtime_table_[0].resize(device_counter[0]);
    runtime_table_[0][0] = std::unique_ptr<Runtime>(new Runtime(Device(Device::Type::CPU, 0)));

    // Context will try to use the first non-cpu available device as the default runtime.
    for (int i = int(Device::Type::COUNT) - 1; i > 0; i--) {
        if (device_counter[i] > 0) {
            runtime_table_[i].resize(device_counter[i]);
            if (current_runtime_ == nullptr) {
                runtime_table_[i][0] = std::unique_ptr<Runtime>(new Runtime(Device(Device::Type(i), 0)));
                current_runtime_ = runtime_table_[i][0].get();
            }
        }
    }

    if (current_runtime_ == nullptr) {
        current_runtime_ = runtime_table_[0][0].get();
    }
}

namespace context {

void setDevice(Device device) {
    ContextImpl::singleton().setDevice(device);
}

Device getDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->device();
}

size_t getDeviceCount(Device::Type type) {
    return ContextImpl::singleton().getDeviceCount(type);
}

infinirtStream_t getStream() {
    return ContextImpl::singleton().getCurrentRuntime()->stream();
}

infiniopHandle_t getInfiniopHandle(Device device) {
    if (device.getType() == Device::Type::CPU) {
        return ContextImpl::singleton().getCpuRuntime()->infiniopHandle();
    }
    if (device != getDevice()) {
        throw std::runtime_error("Requested device doesn't match current runtime.");
    }
    return ContextImpl::singleton().getCurrentRuntime()->infiniopHandle();
}

void syncStream() {
    return ContextImpl::singleton().getCurrentRuntime()->syncStream();
}

void syncDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->syncDevice();
}

std::shared_ptr<Memory> allocateMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocateMemory(size);
}

std::shared_ptr<Memory> allocateHostMemory(size_t size) {
    return ContextImpl::singleton().getCpuRuntime()->allocateMemory(size);
}

std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocatePinnedHostMemory(size);
}

void memcpyH2D(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyH2D(dst, src, size);
}

void memcpyD2H(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2H(dst, src, size);
}

void memcpyD2D(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2D(dst, src, size);
}

void memcpyH2H(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCpuRuntime()->memcpyD2D(dst, src, size);
}

// Timing API implementations
infinirtEvent_t createEvent() {
    return ContextImpl::singleton().getCurrentRuntime()->createEvent();
}

infinirtEvent_t createEventWithFlags(uint32_t flags) {
    return ContextImpl::singleton().getCurrentRuntime()->createEventWithFlags(flags);
}

void recordEvent(infinirtEvent_t event, infinirtStream_t stream) {
    ContextImpl::singleton().getCurrentRuntime()->recordEvent(event, stream);
}

bool queryEvent(infinirtEvent_t event) {
    return ContextImpl::singleton().getCurrentRuntime()->queryEvent(event);
}

void synchronizeEvent(infinirtEvent_t event) {
    ContextImpl::singleton().getCurrentRuntime()->synchronizeEvent(event);
}

void destroyEvent(infinirtEvent_t event) {
    ContextImpl::singleton().getCurrentRuntime()->destroyEvent(event);
}

float elapsedTime(infinirtEvent_t start, infinirtEvent_t end) {
    return ContextImpl::singleton().getCurrentRuntime()->elapsedTime(start, end);
}

void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    ContextImpl::singleton().getCurrentRuntime()->streamWaitEvent(stream, event);
}

} // namespace context

} // namespace infinicore
