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
    if (runtime_table_[int(Device::Type::CPU)].empty() || runtime_table_[int(Device::Type::CPU)][0] == nullptr) {
        throw std::runtime_error("CPU runtime not initialized");
    }
    return runtime_table_[int(Device::Type::CPU)][0].get();
}

Runtime *ContextImpl::getRuntime(Device device) {
    int device_type = int(device.getType());
    size_t device_index = device.getIndex();

    if (device_type >= 0 && device_type < int(Device::Type::COUNT) && device_index < runtime_table_[device_type].size() && runtime_table_[device_type][device_index] != nullptr) {
        return runtime_table_[device_type][device_index].get();
    }

    throw std::runtime_error("Runtime for device " + device.toString() + " is not available");
}

void ContextImpl::setDevice(Device device) {
    Runtime *current = getCurrentRuntime();
    if (current != nullptr && device == current->device()) {
        // Do nothing if the device is already set.
        spdlog::debug("Device {} is already set, no change needed", device.toString());
        return;
    }

    int device_type = int(device.getType());
    size_t device_index = device.getIndex();

    spdlog::debug("Attempting to set device to {} (type={}, index={})",
                  device.toString(), device_type, device_index);

    // Check if the device type is valid and the runtime table has been initialized for this device type
    if (device_type >= 0 && device_type < int(Device::Type::COUNT) && device_index < runtime_table_[device_type].size()) {

        // Use mutex to prevent race conditions when creating new runtimes
        std::lock_guard<std::mutex> lock(runtime_table_mutex_);

        if (runtime_table_[device_type][device_index] == nullptr) {
            // Create the runtime outside of the table first to avoid race conditions
            spdlog::debug("Initializing new runtime for device {}", device.toString());
            auto new_runtime = std::unique_ptr<Runtime>(new Runtime(device));
            auto *runtime_ptr = new_runtime.get();

            // Atomically set the runtime in the table
            runtime_table_[device_type][device_index] = std::move(new_runtime);
            current_runtime_ = runtime_ptr;
            spdlog::debug("Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
        } else {
            spdlog::debug("Activating existing runtime for device {}", device.toString());
            current_runtime_ = runtime_table_[device_type][device_index].get()->activate();
            spdlog::debug("Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
        }
    } else {
        spdlog::error("Failed to set device: {} is not available or has invalid index", device.toString());
        throw std::runtime_error("Device " + device.toString() + " is not available or has invalid index");
    }

    spdlog::info("Successfully set device to {}", device.toString());
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    return runtime_table_[int(type)].size();
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

ContextImpl::ContextImpl() {
    std::vector<int> device_counter(size_t(Device::Type::COUNT));
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

infiniopHandle_t getInfiniopHandle() {
    return ContextImpl::singleton().getCurrentRuntime()->infiniopHandle();
}

void syncStream() {
    return ContextImpl::singleton().getCurrentRuntime()->syncStream();
}

void syncDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->syncDevice();
}

std::shared_ptr<MemoryBlock> allocateMemory(size_t size) {
    spdlog::debug("context::allocateMemory() called for size={}", size);
    auto runtime = ContextImpl::singleton().getCurrentRuntime();
    spdlog::debug("Current runtime device: {}", runtime->device().toString());
    auto memory = runtime->allocateMemory(size);
    spdlog::debug("context::allocateMemory() returned memory={}", static_cast<void *>(memory.get()));
    return memory;
}

std::shared_ptr<MemoryBlock> allocateHostMemory(size_t size) {
    spdlog::debug("context::allocateHostMemory() called for size={}", size);
    auto memory = ContextImpl::singleton().getCpuRuntime()->allocateMemory(size);
    spdlog::debug("context::allocateHostMemory() returned memory={}", static_cast<void *>(memory.get()));
    return memory;
}

std::shared_ptr<MemoryBlock> allocatePinnedHostMemory(size_t size) {
    spdlog::debug("context::allocatePinnedHostMemory() called for size={}", size);
    auto memory = ContextImpl::singleton().getCurrentRuntime()->allocatePinnedHostMemory(size);
    spdlog::debug("context::allocatePinnedHostMemory() returned memory={}", static_cast<void *>(memory.get()));
    return memory;
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

} // namespace context

} // namespace infinicore
