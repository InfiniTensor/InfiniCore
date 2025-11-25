#include "context_impl.hpp"

#include "../utils.hpp"
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>

namespace infinicore {

// Static guard to detect when static destruction starts
struct StaticDestructionGuard {
    ~StaticDestructionGuard() {
        SPDLOG_DEBUG("[STATIC] StaticDestructionGuard: Static destruction starting");
    }
};
static StaticDestructionGuard static_guard;

thread_local Runtime *ContextImpl::current_runtime_ = nullptr;

Runtime *ContextImpl::getCurrentRuntime() {
    if (current_runtime_ == nullptr) {
        SPDLOG_DEBUG("current_runtime_ is null, performing lazy initialization");
        // Lazy initialization: use the first available runtime
        // Protect runtime_table_ access with mutex for thread safety
        std::lock_guard<std::mutex> lock(runtime_table_mutex_);

        // Double-check after acquiring lock (another thread may have set it)
        if (current_runtime_ == nullptr) {
            // Try to find the first non-CPU device, fallback to CPU
            for (int i = int(Device::Type::COUNT) - 1; i > 0; i--) {
                if (!runtime_table_[i].empty() && runtime_table_[i][0] != nullptr) {
                    current_runtime_ = runtime_table_[i][0].get();
                    SPDLOG_DEBUG("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
                    return current_runtime_;
                }
            }
            // Fallback to CPU runtime
            if (!runtime_table_[0].empty() && runtime_table_[0][0] != nullptr) {
                current_runtime_ = runtime_table_[0][0].get();
                SPDLOG_DEBUG("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
            }
        }
    } else {
        // SPDLOG_DEBUG("getCurrentRuntime() returning {} (ptr={})", current_runtime_->device().toString(), static_cast<void *>(current_runtime_));
    }
    return current_runtime_;
}

Runtime *ContextImpl::getCpuRuntime() {
    return runtime_table_[int(Device::Type::CPU)][0].get();
}

void ContextImpl::setDevice(Device device) {
    Device current_device = getCurrentRuntime()->device();
    SPDLOG_DEBUG("[CONTEXT] setDevice: ENTERED - target={}, current={}",
                 device.toString(), current_device.toString());

    if (device == current_device) {
        // Do nothing if the device is already set.
        SPDLOG_DEBUG("[CONTEXT] setDevice: Device already set, returning early");
        return;
    }

    // Fast path: check without lock (double-checked locking pattern)
    int device_type = int(device.getType());
    int device_index = device.getIndex();

    SPDLOG_DEBUG("[CONTEXT] setDevice: Switching from {} to {}",
                 current_device.toString(), device.toString());

    if (runtime_table_[device_type][device_index] == nullptr) {
        // Slow path: acquire lock and create Runtime if needed
        std::lock_guard<std::mutex> lock(runtime_table_mutex_);
        // Double-check after acquiring lock (another thread may have created it)
        if (runtime_table_[device_type][device_index] == nullptr) {
            // Lazy initialization of runtime if never set before.
            runtime_table_[device_type][device_index] = std::unique_ptr<Runtime>(new Runtime(device));
        }
        // Access Runtime within the same lock to prevent use-after-free
        Runtime *runtime_ptr = runtime_table_[device_type][device_index].get();
        if (runtime_ptr != nullptr) {
            SPDLOG_DEBUG("[CONTEXT] setDevice: Activating runtime for {}", device.toString());
            current_runtime_ = runtime_ptr->activate();
            SPDLOG_DEBUG("[CONTEXT] setDevice: Runtime activated, current_runtime_ now points to {}",
                         current_runtime_->device().toString());
        }
    } else {
        // Runtime already exists, protect access to prevent use-after-free
        std::lock_guard<std::mutex> lock(runtime_table_mutex_);
        Runtime *runtime_ptr = runtime_table_[device_type][device_index].get();
        if (runtime_ptr != nullptr) {
            SPDLOG_DEBUG("[CONTEXT] setDevice: Activating existing runtime for {}", device.toString());
            current_runtime_ = runtime_ptr->activate();
            SPDLOG_DEBUG("[CONTEXT] setDevice: Runtime activated, current_runtime_ now points to {}",
                         current_runtime_->device().toString());
        }
    }
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    return runtime_table_[int(type)].size();
}

// Guard struct implementation - can access protected ContextImpl constructor/destructor
// because it's a nested struct
ContextImpl::SingletonGuard::SingletonGuard() {
    // ContextImpl constructor is protected, but we're a nested struct so we can create it
    instance = new ContextImpl();
}

ContextImpl::SingletonGuard::~SingletonGuard() {
    SPDLOG_DEBUG("[CONTEXT] SingletonGuard destructor called - singleton destruction starting");
    // Explicitly delete, which will call ContextImpl destructor
    // Since we're a nested struct, we can access the protected destructor
    if (instance != nullptr) {
        delete instance;
        instance = nullptr;
    }
    SPDLOG_DEBUG("[CONTEXT] SingletonGuard: ContextImpl destroyed");
}

ContextImpl &ContextImpl::singleton() {
    static SingletonGuard guard;
    return *guard.instance;
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

ContextImpl::~ContextImpl() {
    // Wrap entire destructor in try-catch to catch any exceptions
    try {
        SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: START");

        // Clear current_runtime_ pointer first to avoid accessing destroyed objects
        SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Clearing current_runtime_ pointer");
        current_runtime_ = nullptr;

        // Destroy runtimes in reverse order (non-CPU first, then CPU)
        SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Starting to destroy runtime_table_");

        // Count total runtimes
        size_t total_runtimes = 0;
        for (size_t i = 0; i < runtime_table_.size(); ++i) {
            total_runtimes += runtime_table_[i].size();
        }
        SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Total runtimes to destroy: {}", total_runtimes);

        // Destroy runtimes in reverse order
        for (int i = int(Device::Type::COUNT) - 1; i >= 0; --i) {
            SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Destroying runtimes for device type {}", i);

            for (size_t j = runtime_table_[i].size(); j > 0; --j) {
                size_t idx = j - 1;
                if (runtime_table_[i][idx] != nullptr) {
                    SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: About to destroy runtime at type={}, index={}", i, idx);
                    // The unique_ptr will automatically call Runtime destructor
                    runtime_table_[i][idx].reset();
                    SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Successfully destroyed runtime at type={}, index={}", i, idx);
                }
            }

            SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Clearing runtime vector for device type {}", i);
            runtime_table_[i].clear();
            SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Cleared runtime vector for device type {}", i);
        }

        SPDLOG_DEBUG("[CONTEXT] ~ContextImpl: Complete");
    } catch (const std::exception &e) {
        SPDLOG_ERROR("[CONTEXT] ~ContextImpl: EXCEPTION caught: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("[CONTEXT] ~ContextImpl: UNKNOWN EXCEPTION caught");
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
