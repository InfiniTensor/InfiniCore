#pragma once

#include "infinicore.h"
#include "infinirt.h"

#include <infini/rt.h>

namespace infinicore::bridge::infini::rt {

inline infiniStatus_t translate(::infini::rt::runtime::Error error) {
    switch (error) {
    case ::infini::rt::runtime::kSuccess:
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

inline ::infini::rt::Device::Type translate(infiniDevice_t device) {
    switch (device) {
    case INFINI_DEVICE_CPU:
        return ::infini::rt::Device::Type::kCpu;
    case INFINI_DEVICE_NVIDIA:
        return ::infini::rt::Device::Type::kNvidia;
    default:
        return ::infini::rt::Device::Type::kCount;
    }
}

inline infiniDevice_t translate(::infini::rt::Device::Type device) {
    switch (device) {
    case ::infini::rt::Device::Type::kCpu:
        return INFINI_DEVICE_CPU;
    case ::infini::rt::Device::Type::kNvidia:
        return INFINI_DEVICE_NVIDIA;
    default:
        return INFINI_DEVICE_TYPE_COUNT;
    }
}

inline ::infini::rt::runtime::Stream translate(infinirtStream_t stream) {
    return reinterpret_cast<::infini::rt::runtime::Stream>(stream);
}

inline ::infini::rt::runtime::Stream *translate(infinirtStream_t *stream) {
    return reinterpret_cast<::infini::rt::runtime::Stream *>(stream);
}

} // namespace infinicore::bridge::infini::rt
