#include "infinirt_kunlun.h"
#include "../../utils.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>

typedef XPUStream kunlunStream_t;
typedef XPUEvent kunlunEvent_t;

#define CHECK_KUNLUNRT(RT_API) CHECK_INTERNAL(RT_API, XPU_SUCCESS)

namespace infinirt::kunlun {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_KUNLUNRT(xpu_device_count(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_KUNLUNRT(xpu_set_device(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    // TODO: kunlun xpu has no device synchronization API
    // xpu_wait() is waiting for default stream
    CHECK_KUNLUNRT(xpu_wait());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    kunlunStream_t stream;
    CHECK_KUNLUNRT(xpu_stream_create(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_stream_destroy((kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_wait((kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_stream_wait_event((kunlunStream_t)stream, (kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    kunlunEvent_t event;
    CHECK_KUNLUNRT(xpu_event_create(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_event_record((kunlunEvent_t)event, (kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    // no event query in kunlun2
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_event_wait((kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_event_destroy((kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_KUNLUNRT(xpu_malloc(p_ptr, static_cast<uint64_t>(size)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_KUNLUNRT(xpu_host_alloc(p_ptr, static_cast<uint64_t>(size), 0));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_KUNLUNRT(xpu_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

namespace {
// The Kunlun XRE runtime exposes no memset API, so emulate it with H2D
// copies from a host staging buffer filled with the requested byte value.
// The copy is chunked so that large fills (e.g. a multi-GB KV cache) only
// need a bounded amount of host memory.
constexpr size_t MEMSET_STAGING_BYTES = 16ULL * 1024 * 1024;

std::mutex memset_staging_mutex;
std::vector<uint8_t> memset_staging_buffer;

infiniStatus_t memsetViaCopy(void *ptr, int value, size_t count, kunlunStream_t stream) {
    if (count == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(memset_staging_mutex);

    size_t chunk = std::min(count, MEMSET_STAGING_BYTES);
    if (memset_staging_buffer.size() < chunk) {
        memset_staging_buffer.resize(chunk);
    }
    std::memset(memset_staging_buffer.data(), value, chunk);

    auto dst = static_cast<uint8_t *>(ptr);
    for (size_t done = 0; done < count;) {
        size_t n = std::min(chunk, count - done);
        if (stream) {
            // Enqueue on the caller's stream so the fill stays ordered with
            // respect to work already queued on it (a plain xpu_memcpy would
            // bypass the stream and could race kernels writing the buffer).
            CHECK_KUNLUNRT(xpu_memcpy_async(dst + done, memset_staging_buffer.data(),
                                            static_cast<uint64_t>(n), XPUMemcpyKind::XPU_HOST_TO_DEVICE, stream));
        } else {
            CHECK_KUNLUNRT(xpu_memcpy(dst + done, memset_staging_buffer.data(),
                                      static_cast<uint64_t>(n), XPUMemcpyKind::XPU_HOST_TO_DEVICE));
        }
        done += n;
    }

    if (stream) {
        // The staging buffer is shared and refilled by the next call; wait for
        // the stream-ordered copies to finish reading it before unlocking.
        CHECK_KUNLUNRT(xpu_wait(stream));
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace

infiniStatus_t memsetDevice(void *ptr, int value, size_t count) {
    return memsetViaCopy(ptr, value, count, nullptr);
}

infiniStatus_t memsetDeviceAsync(void *ptr, int value, size_t count, infinirtStream_t stream) {
    // XRE has no async memset primitive, so emulate it with copies; they must
    // still be issued on `stream` to preserve stream ordering.
    return memsetViaCopy(ptr, value, count, (kunlunStream_t)stream);
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_KUNLUNRT(xpu_host_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_HOST_TO_DEVICE));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2H:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_HOST));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2D:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_DEVICE));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_H2H:
        std::memcpy(dst, src, size);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        CHECK_KUNLUNRT(xpu_memcpy_async(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_HOST_TO_DEVICE, (kunlunStream_t)stream));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2H:
        CHECK_KUNLUNRT(xpu_memcpy_async(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_HOST, (kunlunStream_t)stream));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2D:
        CHECK_KUNLUNRT(xpu_memcpy_async(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_DEVICE, (kunlunStream_t)stream));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_H2H:
        std::memcpy(dst, src, size);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    // kunlun3 does not support async memory allocation
    // TODO: support async malloc
    CHECK_KUNLUNRT(xpu_malloc(p_ptr, static_cast<uint64_t>(size)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t getMemInfo(int device_id, size_t *free_bytes, size_t *total_bytes) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t getDeviceResourceSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

} // namespace infinirt::kunlun
