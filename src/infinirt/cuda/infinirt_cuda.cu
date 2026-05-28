#include "../../utils.h"
#include "infinirt_cuda.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <unordered_map>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#define CHECK_CUDART(RT_API) CHECK_INTERNAL(RT_API, cudaSuccess)

#define RUN_CUDART(RT_API)                           \
    do {                                             \
        auto api_result_ = (RT_API);                 \
        if (api_result_ != (cudaSuccess)) {          \
            { return INFINI_STATUS_INTERNAL_ERROR; } \
        }                                            \
    } while (0)

// ============================================================
// Communication sampling — shared by all CUDA-like backends
// (NVIDIA, Iluvatar, QY, Hygon, Ali, …)
// ============================================================
namespace {

struct PendingCommunicationSample {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    uint64_t bytes = 0;
};

struct CompletedCommunicationSample {
    std::chrono::steady_clock::time_point completed_at;
    double duration_ms = 0.0;
    uint64_t bytes = 0;
};

struct DeviceCommunicationState {
    std::deque<PendingCommunicationSample> pending;
    std::deque<CompletedCommunicationSample> recent;
};

struct CommunicationStatsStore {
    std::mutex mutex;
    std::unordered_map<int, DeviceCommunicationState> per_device;
};

CommunicationStatsStore &communicationStatsStore() {
    static CommunicationStatsStore store;
    return store;
}

constexpr auto kCommunicationWindow = std::chrono::seconds(1);

void destroyCommunicationSample(const PendingCommunicationSample &sample) {
    if (sample.start_event != nullptr) {
        cudaEventDestroy(sample.start_event);
    }
    if (sample.end_event != nullptr) {
        cudaEventDestroy(sample.end_event);
    }
}

template <typename DeviceFn>
cudaError_t withDeviceGuard(int device_id, DeviceFn &&fn) {
    int previous_device = 0;
    auto status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) {
        return status;
    }

    if (previous_device != device_id) {
        status = cudaSetDevice(device_id);
        if (status != cudaSuccess) {
            return status;
        }
    }

    auto fn_status = fn();

    if (previous_device != device_id) {
        auto restore_status = cudaSetDevice(previous_device);
        if (fn_status == cudaSuccess && restore_status != cudaSuccess) {
            fn_status = restore_status;
        }
    }
    return fn_status;
}

void pruneCommunicationWindow(DeviceCommunicationState &state, std::chrono::steady_clock::time_point now) {
    while (!state.recent.empty() && now - state.recent.front().completed_at > kCommunicationWindow) {
        state.recent.pop_front();
    }
}

void flushCompletedCommunicationSamples(int device_id, DeviceCommunicationState &state) {
    std::deque<PendingCommunicationSample> remaining;

    auto status = withDeviceGuard(device_id, [&]() {
        for (auto &sample : state.pending) {
            auto query_status = cudaEventQuery(sample.end_event);
            if (query_status == cudaSuccess) {
                float elapsed_ms = 0.0f;
                auto elapsed_status = cudaEventElapsedTime(&elapsed_ms, sample.start_event, sample.end_event);
                if (elapsed_status == cudaSuccess) {
                    state.recent.push_back({std::chrono::steady_clock::now(),
                                            static_cast<double>(elapsed_ms),
                                            sample.bytes});
                }
                destroyCommunicationSample(sample);
            } else if (query_status == cudaErrorNotReady) {
                remaining.push_back(sample);
            } else {
                destroyCommunicationSample(sample);
            }
        }
        return cudaSuccess;
    });

    if (status == cudaSuccess) {
        state.pending.swap(remaining);
    }
}

void populateCommunicationSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &store = communicationStatsStore();
    std::lock_guard<std::mutex> lock(store.mutex);
    auto &state = store.per_device[device_id];

    flushCompletedCommunicationSamples(device_id, state);
    auto now = std::chrono::steady_clock::now();
    pruneCommunicationWindow(state, now);

    double total_comm_ms = 0.0;
    uint64_t total_comm_bytes = 0;
    for (auto const &sample : state.recent) {
        total_comm_ms += sample.duration_ms;
        total_comm_bytes += sample.bytes;
    }

    double window_ms = std::chrono::duration<double, std::milli>(kCommunicationWindow).count();

    snapshot->communication_bytes = total_comm_bytes;
    // Avoid std::min — Corex SDK cuda_wrappers/algorithm conflicts with
    // the standard library on float-vs-double template deduction.
    double ratio = total_comm_ms / window_ms;
    snapshot->communication_time_ratio = static_cast<float>(ratio > 1.0 ? 1.0 : ratio);
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_COMMUNICATION;
}

} // namespace

// ============================================================
// GPU utilization via management library (NVML / IXML)
//
// NVIDIA: dlopen libnvidia-ml.so  → nvml* symbols
// Iluvatar: dlopen libixml.so     → same nvml* symbols (IXML is
//           an NVML-compatible management library)
// ============================================================
#if !defined(_WIN32) && (defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API))
namespace {

typedef struct nvmlDevice_st *nvmlDevice_t;
typedef int nvmlReturn_t;

struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};

constexpr nvmlReturn_t NVML_SUCCESS = 0;

using NvmlInitV2Fn = nvmlReturn_t (*)();
using NvmlShutdownFn = nvmlReturn_t (*)();
using NvmlDeviceGetHandleByIndexV2Fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using NvmlDeviceGetUtilizationRatesFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t *);

struct NvmlApi {
    void *handle = nullptr;
    NvmlInitV2Fn init_v2 = nullptr;
    NvmlShutdownFn shutdown = nullptr;
    NvmlDeviceGetHandleByIndexV2Fn get_handle_by_index_v2 = nullptr;
    NvmlDeviceGetUtilizationRatesFn get_utilization_rates = nullptr;
    bool available = false;
    bool initialized = false;
};

NvmlApi &nvmlApi() {
    static NvmlApi api = []() {
        NvmlApi loaded;
        const char *candidates[] = {
#if defined(ENABLE_NVIDIA_API)
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "libnvidia-ml.dylib",
#elif defined(ENABLE_ILUVATAR_API)
            "libixml.so",
#endif
        };

        for (auto candidate : candidates) {
            loaded.handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
            if (loaded.handle != nullptr) {
                break;
            }
        }

        if (loaded.handle == nullptr) {
            return loaded;
        }

        loaded.init_v2 = reinterpret_cast<NvmlInitV2Fn>(dlsym(loaded.handle, "nvmlInit_v2"));
        loaded.shutdown = reinterpret_cast<NvmlShutdownFn>(dlsym(loaded.handle, "nvmlShutdown"));
        loaded.get_handle_by_index_v2 = reinterpret_cast<NvmlDeviceGetHandleByIndexV2Fn>(dlsym(loaded.handle, "nvmlDeviceGetHandleByIndex_v2"));
        loaded.get_utilization_rates = reinterpret_cast<NvmlDeviceGetUtilizationRatesFn>(dlsym(loaded.handle, "nvmlDeviceGetUtilizationRates"));

        loaded.available = loaded.init_v2 != nullptr
                        && loaded.shutdown != nullptr
                        && loaded.get_handle_by_index_v2 != nullptr
                        && loaded.get_utilization_rates != nullptr;
        return loaded;
    }();
    return api;
}

bool tryPopulateNvmlUtilization(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    auto &api = nvmlApi();
    if (!api.available) {
        return false;
    }

    if (!api.initialized) {
        if (api.init_v2() != NVML_SUCCESS) {
            return false;
        }
        api.initialized = true;
    }

    nvmlDevice_t device = nullptr;
    if (api.get_handle_by_index_v2(static_cast<unsigned int>(device_id), &device) != NVML_SUCCESS) {
        return false;
    }

    nvmlUtilization_t util{};
    if (api.get_utilization_rates(device, &util) != NVML_SUCCESS) {
        return false;
    }

    snapshot->compute_utilization = static_cast<float>(util.gpu) / 100.0f;
    snapshot->memory_bandwidth_utilization = static_cast<float>(util.memory) / 100.0f;
    snapshot->kernel_time_ratio = snapshot->compute_utilization;
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_COMPUTE_UTILIZATION
                            | INFINIRT_RESOURCE_FIELD_MEMORY_BANDWIDTH_UTILIZATION
                            | INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO;
    snapshot->estimated_fields |= INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO;
    return true;
}

} // namespace
#endif

// 根据宏定义选择命名空间并实现
#if defined(ENABLE_NVIDIA_API)
namespace infinirt::cuda {
#elif defined(ENABLE_ILUVATAR_API)
namespace infinirt::iluvatar {
#elif defined(ENABLE_QY_API)
namespace infinirt::qy {
#elif defined(ENABLE_HYGON_API)
namespace infinirt::hygon {
#elif defined(ENABLE_ALI_API)
namespace infinirt::ali {
#else
namespace infinirt::cuda { // 默认回退
#endif

infiniStatus_t getDeviceCount(int *count) {
    CHECK_CUDART(cudaGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_CUDART(cudaSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getMemInfo(int device_id, size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    int previous_device = 0;
    CHECK_CUDART(cudaGetDevice(&previous_device));

    if (previous_device != device_id) {
        CHECK_CUDART(cudaSetDevice(device_id));
    }

    auto query_status = cudaMemGetInfo(free_bytes, total_bytes);

    if (previous_device != device_id) {
        auto restore_status = cudaSetDevice(previous_device);
        if (query_status == cudaSuccess && restore_status != cudaSuccess) {
            query_status = restore_status;
        }
    }

    CHECK_CUDART(query_status);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getDeviceResourceSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    if (snapshot == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    std::memset(snapshot, 0, sizeof(*snapshot));
    snapshot->device_id = device_id;
#if defined(ENABLE_NVIDIA_API)
    snapshot->device_type = INFINI_DEVICE_NVIDIA;
#elif defined(ENABLE_ILUVATAR_API)
    snapshot->device_type = INFINI_DEVICE_ILUVATAR;
#elif defined(ENABLE_QY_API)
    snapshot->device_type = INFINI_DEVICE_QY;
#elif defined(ENABLE_HYGON_API)
    snapshot->device_type = INFINI_DEVICE_HYGON;
#elif defined(ENABLE_ALI_API)
    snapshot->device_type = INFINI_DEVICE_ALI;
#else
    snapshot->device_type = INFINI_DEVICE_NVIDIA;
#endif

    auto status = getMemInfo(device_id, &snapshot->free_bytes, &snapshot->total_bytes);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    if (snapshot->total_bytes >= snapshot->free_bytes) {
        snapshot->used_bytes = snapshot->total_bytes - snapshot->free_bytes;
    }
    snapshot->valid_fields |= INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY;

#if !defined(_WIN32) && (defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API))
    (void)tryPopulateNvmlUtilization(device_id, snapshot);
#endif
    populateCommunicationSnapshot(device_id, snapshot);

    return INFINI_STATUS_SUCCESS;
}

void recordCommunicationSample(int device_id, infinirtEvent_t start_event, infinirtEvent_t end_event, uint64_t bytes) {
    if (start_event == nullptr || end_event == nullptr || bytes == 0) {
        return;
    }

    auto &store = communicationStatsStore();
    std::lock_guard<std::mutex> lock(store.mutex);
    store.per_device[device_id].pending.push_back(
        PendingCommunicationSample{
            static_cast<cudaEvent_t>(start_event),
            static_cast<cudaEvent_t>(end_event),
            bytes});
}

infiniStatus_t deviceSynchronize() {
    CHECK_CUDART(cudaDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    cudaStream_t stream;
    CHECK_CUDART(cudaStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    RUN_CUDART(cudaStreamDestroy((cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_CUDART(cudaStreamSynchronize((cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
#ifdef ENABLE_ILUVATAR_API
    return INFINI_STATUS_NOT_IMPLEMENTED;
#else
    CHECK_CUDART(cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
#endif
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    cudaEvent_t event;
    CHECK_CUDART(cudaEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    cudaEvent_t event;
    unsigned int cuda_flags = cudaEventDefault;

    // Convert infinirt flags to CUDA flags
    if (flags & INFINIRT_EVENT_DISABLE_TIMING) {
        cuda_flags |= cudaEventDisableTiming;
    }
    if (flags & INFINIRT_EVENT_BLOCKING_SYNC) {
        cuda_flags |= cudaEventBlockingSync;
    }

    CHECK_CUDART(cudaEventCreateWithFlags(&event, cuda_flags));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_CUDART(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    auto status = cudaEventQuery((cudaEvent_t)event);
    if (status == cudaSuccess) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else if (status == cudaErrorNotReady) {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    } else {
        CHECK_CUDART(status);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_CUDART(cudaEventSynchronize((cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    RUN_CUDART(cudaEventDestroy((cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    CHECK_CUDART(cudaEventElapsedTime(ms_ptr, (cudaEvent_t)start, (cudaEvent_t)end));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_CUDART(cudaMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_CUDART(cudaMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    RUN_CUDART(cudaFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    RUN_CUDART(cudaFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

cudaMemcpyKind toCudaMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    default:
        return cudaMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_CUDART(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_CUDART(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_CUDART(cudaMallocAsync(p_ptr, size, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    RUN_CUDART(cudaFreeAsync(ptr, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memsetDevice(void *ptr, int value, size_t count) {
    RUN_CUDART(cudaMemset(ptr, value, count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memsetDeviceAsync(void *ptr, int value, size_t count, infinirtStream_t stream) {
    RUN_CUDART(cudaMemsetAsync(ptr, value, count, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    cudaStreamCaptureMode graph_mode;
    if (mode == INFINIRT_STREAM_CAPTURE_MODE_GLOBAL) {
        graph_mode = cudaStreamCaptureModeGlobal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL) {
        graph_mode = cudaStreamCaptureModeThreadLocal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_RELAXED) {
        graph_mode = cudaStreamCaptureModeRelaxed;
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    CHECK_CUDART(cudaStreamBeginCapture((cudaStream_t)stream, graph_mode));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    cudaGraph_t graph;
    CHECK_CUDART(cudaStreamEndCapture((cudaStream_t)stream, &graph));
    *graph_ptr = graph;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    RUN_CUDART(cudaGraphDestroy((cudaGraph_t)graph));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    CHECK_CUDART(cudaGraphInstantiate((cudaGraphExec_t *)graph_exec_ptr, (cudaGraph_t)graph, (cudaGraphNode_t *)node_ptr, log_buffer, buffer_size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    RUN_CUDART(cudaGraphExecDestroy((cudaGraphExec_t)graph_exec));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    CHECK_CUDART(cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
}
