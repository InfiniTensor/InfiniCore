#include "runtime.hpp"

#include "../../utils.hpp"

#include "../allocators/device_caching_allocator.hpp"
#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"

#include <iostream>

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
    // Wrap entire destructor in try-catch to prevent exceptions from causing segfaults
    try {
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: START, device type={}, index={}",
                     static_cast<int>(device_.getType()), device_.getIndex());

        // Step 1: Activate device
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 1 - Activating device");
        infiniStatus_t status = infinirtSetDevice((infiniDevice_t)device_.getType(), (int)device_.getIndex());
        if (status != INFINI_STATUS_SUCCESS) {
            SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - Failed to activate device (status={}), continuing cleanup", static_cast<int>(status));
        } else {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 1 - Device activated successfully");
        }

        // Step 2: Sync stream FIRST before destroying allocators
        // This is critical for CUDA devices where allocators use async operations
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 2 - Syncing stream");
        try {
            {
                std::lock_guard<std::mutex> lock(stream_mutex_);
                infiniStatus_t sync_status = infinirtStreamSynchronize(stream_);
                if (sync_status != INFINI_STATUS_SUCCESS) {
                    SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - Stream sync failed (status={})", static_cast<int>(sync_status));
                } else {
                    SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 2 - Stream synced successfully");
                }
            }
        } catch (...) {
            SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - Exception during stream sync");
        }

        // Step 3: Sync device to ensure all operations complete
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 3 - Syncing device");
        try {
            infiniStatus_t device_sync_status = infinirtDeviceSynchronize();
            if (device_sync_status != INFINI_STATUS_SUCCESS) {
                SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - Device sync failed (status={})", static_cast<int>(device_sync_status));
            } else {
                SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 3 - Device synced successfully");
            }
        } catch (...) {
            SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - Exception during device sync");
        }

        // Step 4: NOW reset allocators (safe after stream sync)
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 4 - Resetting pinned_host_memory_allocator");
        if (pinned_host_memory_allocator_) {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 4 - pinned_host_memory_allocator exists, resetting...");
            pinned_host_memory_allocator_.reset();
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 4 - pinned_host_memory_allocator reset complete");
        } else {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 4 - pinned_host_memory_allocator is null, skipping");
        }

        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 5 - Resetting device_memory_allocator");
        if (device_memory_allocator_) {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 5 - device_memory_allocator exists, resetting...");
            device_memory_allocator_.reset();
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 5 - device_memory_allocator reset complete");
        } else {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 5 - device_memory_allocator is null, skipping");
        }

        // Step 6: Destroy infiniop handle
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 6 - Destroying infiniop handle");
        infiniStatus_t handle_status = infiniopDestroyHandle(infiniop_handle_);
        if (handle_status != INFINI_STATUS_SUCCESS) {
            SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - infiniopDestroyHandle failed (status={})", static_cast<int>(handle_status));
        } else {
            SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 6 - infiniop handle destroyed successfully");
        }

        // Step 7: Destroy stream (last, after all operations complete)
        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 7 - Destroying stream");
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            infiniStatus_t stream_status = infinirtStreamDestroy(stream_);
            if (stream_status != INFINI_STATUS_SUCCESS) {
                SPDLOG_WARN("[RUNTIME] ~Runtime: WARNING - infinirtStreamDestroy failed (status={})", static_cast<int>(stream_status));
            } else {
                SPDLOG_DEBUG("[RUNTIME] ~Runtime: Step 7 - stream destroyed successfully");
            }
        }

        SPDLOG_DEBUG("[RUNTIME] ~Runtime: Complete, device type={}, index={}",
                     static_cast<int>(device_.getType()), device_.getIndex());
    } catch (const std::exception &e) {
        SPDLOG_ERROR("[RUNTIME] ~Runtime: EXCEPTION caught: {}", e.what());
    } catch (...) {
        SPDLOG_ERROR("[RUNTIME] ~Runtime: UNKNOWN EXCEPTION caught");
    }
}

Runtime *Runtime::activate() {
    SPDLOG_DEBUG("[RUNTIME] activate: device type={}, index={}",
                 static_cast<int>(device_.getType()), device_.getIndex());
    INFINICORE_CHECK_ERROR(infinirtSetDevice((infiniDevice_t)device_.getType(), (int)device_.getIndex()));
    return this;
}

Device Runtime::device() const {
    return device_;
}

infinirtStream_t Runtime::stream() const {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    return stream_;
}

infiniopHandle_t Runtime::infiniopHandle() const {
    return infiniop_handle_;
}

void Runtime::syncStream() {
    std::lock_guard<std::mutex> lock(stream_mutex_);
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
        SPDLOG_WARN("For CPU devices, pinned memory is not supported, falling back to regular host memory");
        return allocateMemory(size);
    }
    std::byte *data_ptr = pinned_host_memory_allocator_->allocate(size);
    // Pinned host memory is always CPU memory, regardless of which runtime allocates it
    Device cpu_device(Device::Type::CPU, 0);
    return std::make_shared<Memory>(
        data_ptr, size, cpu_device,
        [alloc = pinned_host_memory_allocator_.get()](std::byte *p) {
            alloc->deallocate(p);
        },
        true);
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size) {
    // H2D operations require device runtime, not CPU runtime
    if (device_.getType() == Device::Type::CPU) {
        throw std::runtime_error(
            "Cannot perform H2D memcpy with CPU runtime. "
            "Host-to-device operations require the destination device's runtime to be active. "
            "Please call context::setDevice(destination_device) before performing H2D copy.");
    }
    std::lock_guard<std::mutex> lock(stream_mutex_);
    INFINICORE_CHECK_ERROR(infinirtMemcpyAsync(dst, src, size, INFINIRT_MEMCPY_H2D, stream_));
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Called with runtime device: {}", device_.toString());

    // CRITICAL: Check device type FIRST before any other operations
    // This must be the very first check to prevent segfault
    // Note: The context layer should handle CPU runtime cases by switching to device runtime,
    // but we keep this check as a safety measure

    // Check FIRST, before any logging that might not flush
    if (device_.getType() == Device::Type::CPU) {
        SPDLOG_ERROR("[RUNTIME] memcpyD2H: CRITICAL ERROR - Attempted D2H with CPU runtime!");
        SPDLOG_ERROR("[RUNTIME] memcpyD2H: Current device: type={}, index={}",
                     static_cast<int>(device_.getType()), device_.getIndex());

        // The context layer should have handled this, but if we reach here, it's a programming error
        throw std::runtime_error(
            "Cannot perform D2H memcpy with CPU runtime. "
            "Device-to-host operations require the source device's runtime to be active. "
            "The context layer should automatically switch to a device runtime, but this check indicates it failed.");
    }

    // CRITICAL: Verify infinirt device state matches before calling infinirtMemcpy
    // This ensures CURRENT_DEVICE_TYPE is correctly set
    infiniDevice_t infinirt_device_type;
    int infinirt_device_id;
    infiniStatus_t get_device_status = infinirtGetDevice(&infinirt_device_type, &infinirt_device_id);

    if (get_device_status == INFINI_STATUS_SUCCESS) {
        Device::Type expected_type = device_.getType();
        infiniDevice_t expected_infinirt_type = static_cast<infiniDevice_t>(expected_type);

        SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Verifying infinirt device state before memcpy - "
                     "runtime device={}, infinirt device_type={}, device_id={}",
                     device_.toString(), (int)infinirt_device_type, infinirt_device_id);

        if (infinirt_device_type != expected_infinirt_type || infinirt_device_id != static_cast<int>(device_.getIndex())) {
            SPDLOG_WARN("[RUNTIME] memcpyD2H: Device state mismatch - runtime={}, infinirt device_type={}, device_id={}",
                        device_.toString(), (int)infinirt_device_type, infinirt_device_id);

            // Force re-activation to synchronize state
            activate();

            // Verify again after re-activation
            get_device_status = infinirtGetDevice(&infinirt_device_type, &infinirt_device_id);
            if (get_device_status == INFINI_STATUS_SUCCESS && (infinirt_device_type != expected_infinirt_type || infinirt_device_id != static_cast<int>(device_.getIndex()))) {
                SPDLOG_ERROR("[RUNTIME] memcpyD2H: CRITICAL - Device state still mismatched after re-activation!");
                throw std::runtime_error(
                    "Failed to synchronize infinirt device state before D2H memcpy. "
                    "Runtime device is "
                    + device_.toString() + " but infinirt reports device_type=" + std::to_string((int)infinirt_device_type) + ", device_id=" + std::to_string(infinirt_device_id) + ". This will cause incorrect routing in infinirtMemcpy.");
            } else {
                SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Device state synchronized after re-activation");
            }
        } else {
            SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Device state verified - infinirt matches runtime device");
        }
    } else {
        SPDLOG_WARN("[RUNTIME] memcpyD2H: Could not verify infinirt device state (status={}), proceeding anyway",
                    static_cast<int>(get_device_status));
    }

    // CRITICAL: Synchronize device before D2H memcpy (PyTorch pattern)
    // This ensures all pending operations complete before accessing device memory
    // Prevents race conditions where memory is still being written
    SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Synchronizing device before D2H memcpy");
    syncDevice(); // Ensure all operations complete before memcpy

    SPDLOG_DEBUG("[RUNTIME] memcpyD2H: device type={}, index={}",
                 static_cast<int>(device_.getType()), device_.getIndex());
    SPDLOG_DEBUG("[RUNTIME] memcpyD2H: Calling infinirtMemcpy with device runtime");
    INFINICORE_CHECK_ERROR(infinirtMemcpy(dst, src, size, INFINIRT_MEMCPY_D2H));
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size) {
    // D2D operations require device runtime, not CPU runtime
    if (device_.getType() == Device::Type::CPU) {
        throw std::runtime_error(
            "Cannot perform D2D memcpy with CPU runtime. "
            "Device-to-device operations require a device runtime to be active. "
            "Please call context::setDevice(device) before performing D2D copy.");
    }
    std::lock_guard<std::mutex> lock(stream_mutex_);
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
