#include "standalone_infinirt_graph_bridge.hpp"

#ifdef USE_STANDALONE_INFINIRT_GRAPH

#include <cstdlib>
#include <string>

#include <infini/rt.h>

namespace infinicore::graph::standalone_infinirt {
namespace {

using StandaloneDevice = infini::rt::Device;

bool truthy_env(const char *name) {
    auto value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text{value};
    return text == "1" || text == "ON" || text == "on" || text == "true" || text == "TRUE";
}

template <StandaloneDevice::Type device_type>
constexpr bool standalone_device_enabled() {
    return infini::rt::DeviceEnabled<device_type>::value;
}

bool supports_device(Device::Type type) {
    switch (type) {
    case Device::Type::NVIDIA:
        return standalone_device_enabled<StandaloneDevice::Type::kNvidia>();
    case Device::Type::ASCEND:
        return standalone_device_enabled<StandaloneDevice::Type::kAscend>();
    default:
        return false;
    }
}

thread_local Device::Type current_device_type = Device::Type::CPU;

template <typename Runtime, typename Status>
infiniStatus_t to_core_status(Status status) {
    return status == Runtime::kSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

template <StandaloneDevice::Type device_type>
infiniStatus_t set_device_impl(int index) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::SetDevice(index));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t stream_begin_capture_impl(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        auto standalone_mode = Runtime::kStreamCaptureModeRelaxed;
        switch (mode) {
        case INFINIRT_STREAM_CAPTURE_MODE_GLOBAL:
            standalone_mode = Runtime::kStreamCaptureModeGlobal;
            break;
        case INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL:
            standalone_mode = Runtime::kStreamCaptureModeThreadLocal;
            break;
        case INFINIRT_STREAM_CAPTURE_MODE_RELAXED:
            standalone_mode = Runtime::kStreamCaptureModeRelaxed;
            break;
        }
        return to_core_status<Runtime>(Runtime::StreamBeginCapture(
            reinterpret_cast<typename Runtime::Stream>(stream),
            standalone_mode));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t stream_end_capture_impl(infinirtStream_t stream, infinirtGraph_t *graph) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::StreamEndCapture(
            reinterpret_cast<typename Runtime::Stream>(stream),
            reinterpret_cast<typename Runtime::Graph *>(graph)));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t graph_destroy_impl(infinirtGraph_t graph) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::GraphDestroy(
            reinterpret_cast<typename Runtime::Graph>(graph)));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t graph_instantiate_impl(infinirtGraphExec_t *graph_exec, infinirtGraph_t graph) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::GraphInstantiate(
            reinterpret_cast<typename Runtime::GraphExec *>(graph_exec),
            reinterpret_cast<typename Runtime::Graph>(graph)));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t graph_exec_destroy_impl(infinirtGraphExec_t graph_exec) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::GraphExecDestroy(
            reinterpret_cast<typename Runtime::GraphExec>(graph_exec)));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <StandaloneDevice::Type device_type>
infiniStatus_t graph_launch_impl(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    if constexpr (standalone_device_enabled<device_type>()) {
        using Runtime = infini::rt::runtime::Runtime<device_type>;
        return to_core_status<Runtime>(Runtime::GraphLaunch(
            reinterpret_cast<typename Runtime::GraphExec>(graph_exec),
            reinterpret_cast<typename Runtime::Stream>(stream)));
    } else {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

template <typename... Args>
infiniStatus_t dispatch_current(
    infiniStatus_t (*nvidia_fn)(Args...),
    infiniStatus_t (*ascend_fn)(Args...),
    Args... args) {
    switch (current_device_type) {
    case Device::Type::NVIDIA:
        return nvidia_fn(args...);
    case Device::Type::ASCEND:
        return ascend_fn(args...);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

} // namespace

bool enabled() {
    return truthy_env("INFINICORE_USE_STANDALONE_INFINIRT_GRAPH");
}

bool available(const Device &device) {
    return enabled() && supports_device(device.getType());
}

infiniStatus_t set_device(const Device &device) {
    if (!supports_device(device.getType())) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    auto status = INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    switch (device.getType()) {
    case Device::Type::NVIDIA:
        status = set_device_impl<StandaloneDevice::Type::kNvidia>(static_cast<int>(device.getIndex()));
        break;
    case Device::Type::ASCEND:
        status = set_device_impl<StandaloneDevice::Type::kAscend>(static_cast<int>(device.getIndex()));
        break;
    default:
        break;
    }
    if (status == INFINI_STATUS_SUCCESS) {
        current_device_type = device.getType();
    }
    return status;
}

infiniStatus_t stream_begin_capture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    if (stream == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        stream_begin_capture_impl<StandaloneDevice::Type::kNvidia>,
        stream_begin_capture_impl<StandaloneDevice::Type::kAscend>,
        stream,
        mode);
}

infiniStatus_t stream_end_capture(infinirtStream_t stream, infinirtGraph_t *graph) {
    if (stream == nullptr || graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        stream_end_capture_impl<StandaloneDevice::Type::kNvidia>,
        stream_end_capture_impl<StandaloneDevice::Type::kAscend>,
        stream,
        graph);
}

infiniStatus_t graph_destroy(infinirtGraph_t graph) {
    if (graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        graph_destroy_impl<StandaloneDevice::Type::kNvidia>,
        graph_destroy_impl<StandaloneDevice::Type::kAscend>,
        graph);
}

infiniStatus_t graph_instantiate(infinirtGraphExec_t *graph_exec, infinirtGraph_t graph) {
    if (graph_exec == nullptr || graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        graph_instantiate_impl<StandaloneDevice::Type::kNvidia>,
        graph_instantiate_impl<StandaloneDevice::Type::kAscend>,
        graph_exec,
        graph);
}

infiniStatus_t graph_exec_destroy(infinirtGraphExec_t graph_exec) {
    if (graph_exec == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        graph_exec_destroy_impl<StandaloneDevice::Type::kNvidia>,
        graph_exec_destroy_impl<StandaloneDevice::Type::kAscend>,
        graph_exec);
}

infiniStatus_t graph_launch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    if (graph_exec == nullptr || stream == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return dispatch_current(
        graph_launch_impl<StandaloneDevice::Type::kNvidia>,
        graph_launch_impl<StandaloneDevice::Type::kAscend>,
        graph_exec,
        stream);
}

} // namespace infinicore::graph::standalone_infinirt

#else

namespace infinicore::graph::standalone_infinirt {

bool enabled() {
    return false;
}

bool available(const Device &) {
    return false;
}

infiniStatus_t set_device(const Device &) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t stream_begin_capture(infinirtStream_t, infinirtStreamCaptureMode_t) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t stream_end_capture(infinirtStream_t, infinirtGraph_t *) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t graph_destroy(infinirtGraph_t) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t graph_instantiate(infinirtGraphExec_t *, infinirtGraph_t) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t graph_exec_destroy(infinirtGraphExec_t) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t graph_launch(infinirtGraphExec_t, infinirtStream_t) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

} // namespace infinicore::graph::standalone_infinirt

#endif
