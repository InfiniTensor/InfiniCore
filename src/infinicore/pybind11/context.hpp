#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore.hpp"
#include "infinicore/graph/capture_arena.hpp"

#ifdef ENABLE_ATEN
#include <torch/extension.h>
#endif

namespace py = pybind11;

namespace infinicore::context {

inline void bind(py::module &m) {
    // Device management
    m.def("get_device", &getDevice, "Get the current active device");
    m.def("get_device_count", &getDeviceCount,
          "Get the number of available devices of a specific type",
          py::arg("device_type"));
    m.def("set_device", &setDevice,
          "Set the current active device",
          py::arg("device"));

    // Stream and handle management
    m.def("get_stream", &getStream, "Get the current stream");

    // Synchronization
    m.def("sync_stream", &syncStream, "Synchronize the current stream");
    m.def("sync_device", &syncDevice, "Synchronize the current device");

    // Graph
    m.def("is_graph_recording", &isGraphRecording, "Check if graph recording is turned on");
    m.def("is_device_stream_capturing",
          &isDeviceStreamCapturing,
          "True while Graph::instantiate is inside hcStreamBeginCapture");
    m.def("moe_triton_capture_allowed",
          &moeTritonCaptureAllowed,
          "Phase-adaptive MoE under stream capture (Decode + non-eager; "
          "FORCE_CAPTURE / deprecated TRITON_CAPTURE under non-eager Decode; "
          "MetaX also needs INFINI_MOE_METAX_CAPTURE_UNSAFE (MoE-in-graph garbles).",
          "INFINI_MOE_FORCE_HOST_BREAK forces false)");
    m.def(
        "set_inference_phase",
        [](const std::string &phase) {
            if (phase == "decode" || phase == "Decode") {
                setInferencePhase(InferencePhase::Decode);
            } else if (phase == "prefill" || phase == "Prefill") {
                setInferencePhase(InferencePhase::Prefill);
            } else {
                setInferencePhase(InferencePhase::Unknown);
            }
        },
        py::arg("phase"),
        "Set TLS InferencePhase: decode | prefill | unknown");
    m.def(
        "get_inference_phase",
        []() -> std::string {
            switch (getInferencePhase()) {
            case InferencePhase::Decode:
                return "decode";
            case InferencePhase::Prefill:
                return "prefill";
            default:
                return "unknown";
            }
        },
        "Current TLS InferencePhase string");
    m.def("start_graph_recording", &startGraphRecording, "Start graph recording");
    m.def("stop_graph_recording", &stopGraphRecording, "Stop graph recording and return the graph");

    // Capture arena (Phase 3 unified MM — no c10 MemPool)
    m.def("capture_arena_active",
          &infinicore::graph::capture_arena_active,
          "True while InfiniCore CaptureArena is active on this thread");
    m.def("capture_used_torch_mempool",
          &infinicore::graph::capture_used_torch_mempool,
          "Always false after Phase 3 (MemPool removed from capture path)");
#ifdef ENABLE_ATEN
    m.def(
        "capture_empty_like",
        [](const torch::Tensor &prototype, const std::vector<int64_t> &sizes) -> torch::Tensor {
            auto *arena = infinicore::graph::current_capture_arena();
            if (arena == nullptr) {
                throw std::runtime_error(
                    "capture_empty_like requires an active InfiniCore CaptureArena "
                    "(inside hcStreamBeginCapture)");
            }
            return arena->empty_aten(sizes, prototype.options().requires_grad(false));
        },
        py::arg("prototype"),
        py::arg("sizes"),
        "Allocate IC-backed torch tensor under CaptureArena (dtype/device from prototype)");
    m.def(
        "capture_retain",
        [](const torch::Tensor &t) {
            auto *arena = infinicore::graph::current_capture_arena();
            if (arena == nullptr) {
                return;
            }
            arena->retain(t);
        },
        py::arg("tensor"),
        "Retain a torch tensor on the active CaptureArena for segment lifetime");
#endif
}

} // namespace infinicore::context
