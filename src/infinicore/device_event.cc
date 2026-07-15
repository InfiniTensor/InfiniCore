#include "infinicore.hpp"

namespace infinicore {
namespace {

void destroy_event_on_device(
    infinirtEvent_t event, Device device) noexcept {
    if (event == nullptr) {
        return;
    }
    try {
        const auto previous_device = context::getDevice();
        const bool changed_device = previous_device != device;
        try {
            context::setDevice(device);
            context::destroyEvent(event);
        } catch (...) {
        }
        try {
            if (changed_device) {
                context::setDevice(previous_device);
            }
        } catch (...) {
        }
    } catch (...) {
        // Destructors and move assignment cannot surface context errors.
    }
}

} // namespace

DeviceEvent::DeviceEvent()
    : device_(context::getDevice()), is_recorded_(false) {
    event_ = context::createEvent();
}

DeviceEvent::DeviceEvent(uint32_t flags)
    : device_(context::getDevice()), is_recorded_(false) {
    event_ = context::createEventWithFlags(flags);
}

DeviceEvent::DeviceEvent(Device device)
    : device_(device), is_recorded_(false) {
    // Switch to target device for event creation
    Device current_device = context::getDevice();
    context::setDevice(device_);
    event_ = context::createEvent();
    // Restore original device
    context::setDevice(current_device);
}

DeviceEvent::DeviceEvent(Device device, uint32_t flags)
    : device_(device), is_recorded_(false) {
    // Switch to target device for event creation
    Device current_device = context::getDevice();
    context::setDevice(device_);
    event_ = context::createEventWithFlags(flags);
    // Restore original device
    context::setDevice(current_device);
}

DeviceEvent::DeviceEvent(DeviceEvent &&other) noexcept
    : event_(other.event_), device_(other.device_), is_recorded_(other.is_recorded_) {
    other.event_ = nullptr;
    other.is_recorded_ = false;
}

DeviceEvent &DeviceEvent::operator=(DeviceEvent &&other) noexcept {
    if (this != &other) {
        // Clean up current resources
        destroy_event_on_device(event_, device_);

        // Transfer ownership
        event_ = other.event_;
        device_ = other.device_;
        is_recorded_ = other.is_recorded_;

        // Reset source
        other.event_ = nullptr;
        other.is_recorded_ = false;
    }
    return *this;
}

DeviceEvent::~DeviceEvent() {
    destroy_event_on_device(event_, device_);
}

void DeviceEvent::record() {
    Device current_device = context::getDevice();

    // Ensure we're on the correct device
    if (current_device != device_) {
        context::setDevice(device_);
    }

    context::recordEvent(event_);
    is_recorded_ = true;

    // Restore original device if we changed it
    if (current_device != device_) {
        context::setDevice(current_device);
    }
}

void DeviceEvent::record(infinirtStream_t stream) {
    Device current_device = context::getDevice();

    // Ensure we're on the correct device
    if (current_device != device_) {
        context::setDevice(device_);
    }

    context::recordEvent(event_, stream);
    is_recorded_ = true;

    // Restore original device if we changed it
    if (current_device != device_) {
        context::setDevice(current_device);
    }
}

void DeviceEvent::synchronize() {
    Device current_device = context::getDevice();

    // Ensure we're on the correct device
    if (current_device != device_) {
        context::setDevice(device_);
    }

    context::synchronizeEvent(event_);

    // Restore original device if we changed it
    if (current_device != device_) {
        context::setDevice(current_device);
    }
}

bool DeviceEvent::query() const {
    Device current_device = context::getDevice();
    bool result = false;

    // Ensure we're on the correct device
    if (current_device != device_) {
        context::setDevice(device_);
    }

    result = context::queryEvent(event_);

    // Restore original device if we changed it
    if (current_device != device_) {
        context::setDevice(current_device);
    }

    return result;
}

float DeviceEvent::elapsed_time(const DeviceEvent &other) const {
    // Both events must be on the same device
    if (device_ != other.device_) {
        throw std::runtime_error("Cannot measure elapsed time between events on different devices");
    }

    // Both events must be recorded
    if (!is_recorded_ || !other.is_recorded_) {
        throw std::runtime_error("Both events must be recorded before measuring elapsed time");
    }

    Device current_device = context::getDevice();

    // Switch to the device where events reside
    if (current_device != device_) {
        context::setDevice(device_);
    }

    float elapsed_ms = context::elapsedTime(event_, other.event_);

    // Restore original device if we changed it
    if (current_device != device_) {
        context::setDevice(current_device);
    }

    return elapsed_ms;
}

void DeviceEvent::wait(infinirtStream_t stream) const {
    wait_on(device_, stream);
}

void DeviceEvent::wait_on(Device stream_device,
                          infinirtStream_t stream) const {
    if (!is_recorded_) {
        throw std::runtime_error("Cannot wait for an event before it is recorded");
    }

    Device current_device = context::getDevice();

    if (current_device != stream_device) {
        context::setDevice(stream_device);
    }

    try {
        context::streamWaitEvent(stream, event_);
        if (current_device != stream_device) {
            context::setDevice(current_device);
        }
    } catch (...) {
        try {
            if (current_device != stream_device) {
                context::setDevice(current_device);
            }
        } catch (...) {
        }
        throw;
    }
}

} // namespace infinicore
