#include "standalone_infinirt_graph_bridge.hpp"

#ifdef USE_STANDALONE_INFINIRT_GRAPH

#include <cstdlib>
#include <string>

#include <infini/rt.h>

namespace infinicore::graph::standalone_infinirt {
namespace {

using StandaloneDevice = infini::rt::Device;
using StandaloneRuntime = infini::rt::runtime::Runtime<StandaloneDevice::Type::kNvidia>;

bool truthy_env(const char *name) {
    auto value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text{value};
    return text == "1" || text == "ON" || text == "on" || text == "true" || text == "TRUE";
}

bool supports_device(Device::Type type) {
    return type == Device::Type::NVIDIA;
}

template <typename Status>
infiniStatus_t to_core_status(Status status) {
    return status == StandaloneRuntime::kSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

StandaloneRuntime::Stream to_standalone_stream(infinirtStream_t stream) {
    return reinterpret_cast<StandaloneRuntime::Stream>(stream);
}

StandaloneRuntime::Graph to_standalone_graph(infinirtGraph_t graph) {
    return reinterpret_cast<StandaloneRuntime::Graph>(graph);
}

StandaloneRuntime::GraphExec to_standalone_graph_exec(infinirtGraphExec_t graph_exec) {
    return reinterpret_cast<StandaloneRuntime::GraphExec>(graph_exec);
}

decltype(StandaloneRuntime::kStreamCaptureModeRelaxed)
to_standalone_capture_mode(infinirtStreamCaptureMode_t mode) {
    switch (mode) {
    case INFINIRT_STREAM_CAPTURE_MODE_GLOBAL:
        return StandaloneRuntime::kStreamCaptureModeGlobal;
    case INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL:
        return StandaloneRuntime::kStreamCaptureModeThreadLocal;
    case INFINIRT_STREAM_CAPTURE_MODE_RELAXED:
        return StandaloneRuntime::kStreamCaptureModeRelaxed;
    }
    return StandaloneRuntime::kStreamCaptureModeRelaxed;
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
    return to_core_status(StandaloneRuntime::SetDevice(static_cast<int>(device.getIndex())));
}

infiniStatus_t stream_begin_capture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    if (stream == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::StreamBeginCapture(
        to_standalone_stream(stream),
        to_standalone_capture_mode(mode)));
}

infiniStatus_t stream_end_capture(infinirtStream_t stream, infinirtGraph_t *graph) {
    if (stream == nullptr || graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::StreamEndCapture(
        to_standalone_stream(stream),
        reinterpret_cast<StandaloneRuntime::Graph *>(graph)));
}

infiniStatus_t graph_destroy(infinirtGraph_t graph) {
    if (graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::GraphDestroy(to_standalone_graph(graph)));
}

infiniStatus_t graph_instantiate(infinirtGraphExec_t *graph_exec, infinirtGraph_t graph) {
    if (graph_exec == nullptr || graph == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::GraphInstantiate(
        reinterpret_cast<StandaloneRuntime::GraphExec *>(graph_exec),
        to_standalone_graph(graph)));
}

infiniStatus_t graph_exec_destroy(infinirtGraphExec_t graph_exec) {
    if (graph_exec == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::GraphExecDestroy(
        to_standalone_graph_exec(graph_exec)));
}

infiniStatus_t graph_launch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    if (graph_exec == nullptr || stream == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return to_core_status(StandaloneRuntime::GraphLaunch(
        to_standalone_graph_exec(graph_exec),
        to_standalone_stream(stream)));
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
