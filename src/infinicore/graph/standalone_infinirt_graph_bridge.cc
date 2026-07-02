#include "standalone_infinirt_graph_bridge.hpp"

#ifdef USE_STANDALONE_INFINIRT_GRAPH

#include "../utils.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <string>

#include <infini/rt/c_api.h>

namespace infinicore::graph::standalone_infinirt {
namespace {

using StreamWrapFn = infiniRtStatus_t (*)(infiniRtDevice_t, void *, infiniRtStream_t *);
using StreamDestroyFn = infiniRtStatus_t (*)(infiniRtStream_t);
using StreamBeginCaptureFn = infiniRtStatus_t (*)(infiniRtStream_t, infiniRtStreamCaptureMode_t);
using StreamEndCaptureFn = infiniRtStatus_t (*)(infiniRtStream_t, infiniRtGraph_t *);
using GraphDestroyFn = infiniRtStatus_t (*)(infiniRtGraph_t);
using GraphInstantiateFn = infiniRtStatus_t (*)(infiniRtGraphExec_t *, infiniRtGraph_t);
using GraphExecDestroyFn = infiniRtStatus_t (*)(infiniRtGraphExec_t);
using GraphLaunchFn = infiniRtStatus_t (*)(infiniRtGraphExec_t, infiniRtStream_t);

struct Api {
    void *handle = nullptr;
    StreamWrapFn stream_wrap = nullptr;
    StreamDestroyFn stream_destroy = nullptr;
    StreamBeginCaptureFn stream_begin_capture = nullptr;
    StreamEndCaptureFn stream_end_capture = nullptr;
    GraphDestroyFn graph_destroy = nullptr;
    GraphInstantiateFn graph_instantiate = nullptr;
    GraphExecDestroyFn graph_exec_destroy = nullptr;
    GraphLaunchFn graph_launch = nullptr;
    bool loaded = false;
};

bool truthy_env(const char *name) {
    auto value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    std::string text{value};
    return text == "1" || text == "ON" || text == "on" || text == "true" || text == "TRUE";
}

std::string standalone_library_path() {
    if (auto explicit_path = std::getenv("INFINIRT_GRAPH_LIBRARY")) {
        return explicit_path;
    }
    if (auto root = std::getenv("INFINI_RT_ROOT")) {
        auto lib = std::string(root) + "/lib/libinfinirt.so";
        void *handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            dlclose(handle);
            return lib;
        }
        return std::string(root) + "/lib64/libinfinirt.so";
    }
    return "libinfinirt.so";
}

template <typename T>
bool load_symbol(void *handle, const char *name, T *symbol) {
    *symbol = reinterpret_cast<T>(dlsym(handle, name));
    return *symbol != nullptr;
}

Api &api() {
    static Api api_ = [] {
        Api loaded{};
        const auto path = standalone_library_path();
        loaded.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (loaded.handle == nullptr) {
            spdlog::warn("Standalone InfiniRT graph bridge disabled: failed to load {}: {}", path, dlerror());
            return loaded;
        }

        loaded.loaded =
            load_symbol(loaded.handle, "infiniRtStreamWrap", &loaded.stream_wrap)
            && load_symbol(loaded.handle, "infiniRtStreamDestroy", &loaded.stream_destroy)
            && load_symbol(loaded.handle, "infiniRtStreamBeginCapture", &loaded.stream_begin_capture)
            && load_symbol(loaded.handle, "infiniRtStreamEndCapture", &loaded.stream_end_capture)
            && load_symbol(loaded.handle, "infiniRtGraphDestroy", &loaded.graph_destroy)
            && load_symbol(loaded.handle, "infiniRtGraphInstantiate", &loaded.graph_instantiate)
            && load_symbol(loaded.handle, "infiniRtGraphExecDestroy", &loaded.graph_exec_destroy)
            && load_symbol(loaded.handle, "infiniRtGraphLaunch", &loaded.graph_launch);

        if (loaded.loaded) {
            spdlog::info("Standalone InfiniRT graph bridge loaded: {}", path);
        } else {
            spdlog::warn("Standalone InfiniRT graph bridge disabled: required graph symbols missing in {}", path);
        }
        return loaded;
    }();
    return api_;
}

infiniRtDeviceType_t to_standalone_device_type(Device::Type type) {
    switch (type) {
    case Device::Type::CPU:
        return INFINI_RT_DEVICE_CPU;
    case Device::Type::NVIDIA:
        return INFINI_RT_DEVICE_NVIDIA;
    case Device::Type::CAMBRICON:
        return INFINI_RT_DEVICE_CAMBRICON;
    case Device::Type::ASCEND:
        return INFINI_RT_DEVICE_ASCEND;
    case Device::Type::METAX:
        return INFINI_RT_DEVICE_METAX;
    case Device::Type::MOORE:
        return INFINI_RT_DEVICE_MOORE;
    case Device::Type::ILUVATAR:
        return INFINI_RT_DEVICE_ILUVATAR;
    case Device::Type::KUNLUN:
        return INFINI_RT_DEVICE_KUNLUN;
    case Device::Type::HYGON:
        return INFINI_RT_DEVICE_HYGON;
    case Device::Type::QY:
        return INFINI_RT_DEVICE_QY;
    default:
        return INFINI_RT_DEVICE_CPU;
    }
}

infiniStatus_t to_core_status(infiniRtStatus_t status) {
    switch (status) {
    case INFINI_RT_STATUS_SUCCESS:
        return INFINI_STATUS_SUCCESS;
    case INFINI_RT_STATUS_INVALID_ARGUMENT:
        return INFINI_STATUS_BAD_PARAM;
    case INFINI_RT_STATUS_UNSUPPORTED_DEVICE:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    case INFINI_RT_STATUS_RUNTIME_ERROR:
    default:
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

infiniRtStreamCaptureMode_t to_standalone_capture_mode(infinirtStreamCaptureMode_t mode) {
    switch (mode) {
    case INFINIRT_STREAM_CAPTURE_MODE_GLOBAL:
        return INFINI_RT_STREAM_CAPTURE_MODE_GLOBAL;
    case INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL:
        return INFINI_RT_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    case INFINIRT_STREAM_CAPTURE_MODE_RELAXED:
        return INFINI_RT_STREAM_CAPTURE_MODE_RELAXED;
    }
    return INFINI_RT_STREAM_CAPTURE_MODE_RELAXED;
}

infiniRtDevice_t to_standalone_device(const Device &device) {
    return infiniRtDevice_t{
        to_standalone_device_type(device.getType()),
        static_cast<int>(device.getIndex()),
    };
}

} // namespace

bool enabled() {
    return truthy_env("INFINICORE_USE_STANDALONE_INFINIRT_GRAPH");
}

bool available() {
    return enabled() && api().loaded;
}

infinirtStream_t wrap_stream(const Device &device, infinirtStream_t stream) {
    auto &rt = api();
    if (!rt.loaded || stream == nullptr) {
        return nullptr;
    }

    auto standalone_device = to_standalone_device(device);
    infiniRtStream_t wrapped = nullptr;
    if (rt.stream_wrap(standalone_device, stream, &wrapped) != INFINI_RT_STATUS_SUCCESS) {
        return nullptr;
    }
    return reinterpret_cast<infinirtStream_t>(wrapped);
}

void destroy_wrapped_stream(infinirtStream_t stream) {
    if (stream == nullptr || !api().loaded) {
        return;
    }
    api().stream_destroy(reinterpret_cast<infiniRtStream_t>(stream));
}

infiniStatus_t stream_begin_capture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    if (!api().loaded || stream == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().stream_begin_capture(
        reinterpret_cast<infiniRtStream_t>(stream),
        to_standalone_capture_mode(mode)));
}

infiniStatus_t stream_end_capture(infinirtStream_t stream, infinirtGraph_t *graph) {
    if (!api().loaded || stream == nullptr || graph == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().stream_end_capture(
        reinterpret_cast<infiniRtStream_t>(stream),
        reinterpret_cast<infiniRtGraph_t *>(graph)));
}

infiniStatus_t graph_destroy(infinirtGraph_t graph) {
    if (!api().loaded || graph == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().graph_destroy(reinterpret_cast<infiniRtGraph_t>(graph)));
}

infiniStatus_t graph_instantiate(infinirtGraphExec_t *graph_exec, infinirtGraph_t graph) {
    if (!api().loaded || graph_exec == nullptr || graph == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().graph_instantiate(
        reinterpret_cast<infiniRtGraphExec_t *>(graph_exec),
        reinterpret_cast<infiniRtGraph_t>(graph)));
}

infiniStatus_t graph_exec_destroy(infinirtGraphExec_t graph_exec) {
    if (!api().loaded || graph_exec == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().graph_exec_destroy(reinterpret_cast<infiniRtGraphExec_t>(graph_exec)));
}

infiniStatus_t graph_launch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    if (!api().loaded || graph_exec == nullptr || stream == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return to_core_status(api().graph_launch(
        reinterpret_cast<infiniRtGraphExec_t>(graph_exec),
        reinterpret_cast<infiniRtStream_t>(stream)));
}

} // namespace infinicore::graph::standalone_infinirt

#else

namespace infinicore::graph::standalone_infinirt {

bool enabled() {
    return false;
}

bool available() {
    return false;
}

infinirtStream_t wrap_stream(const Device &, infinirtStream_t) {
    return nullptr;
}

void destroy_wrapped_stream(infinirtStream_t) {
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
