#include "infinirt_cpu.h"

#include <infini/rt.h>

namespace infinirt::cpu {
namespace {

infini::rt::MemcpyKind toRtMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2H:
        return infini::rt::MemcpyKind::kMemcpyHostToHost;
    case INFINIRT_MEMCPY_H2D:
        return infini::rt::MemcpyKind::kMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return infini::rt::MemcpyKind::kMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return infini::rt::MemcpyKind::kMemcpyDeviceToDevice;
    }
    return infini::rt::MemcpyKind::kMemcpyHostToHost;
}

infiniStatus_t validateCpuDeviceId(int device_id) {
    return device_id == 0 ? INFINI_STATUS_SUCCESS : INFINI_STATUS_DEVICE_NOT_FOUND;
}

infiniStatus_t runtimeStatus(infini::rt::Error status) {
    return status == infini::rt::kSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace

infiniStatus_t getDeviceCount(int *count) {
    if (count == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::GetDeviceCount(count));
}

infiniStatus_t setDevice(int device_id) {
    auto status = validateCpuDeviceId(device_id);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    return runtimeStatus(infini::rt::SetDevice(device_id));
}

infiniStatus_t getMemInfo(int device_id, size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = validateCpuDeviceId(device_id);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    status = runtimeStatus(infini::rt::MemGetInfo(free_bytes, total_bytes));
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    return *total_bytes == 0 ? INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED : INFINI_STATUS_SUCCESS;
}

infiniStatus_t getDeviceResourceSnapshot(int device_id, infinirtDeviceResourceSnapshot_t *snapshot) {
    if (snapshot == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = validateCpuDeviceId(device_id);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    *snapshot = infinirtDeviceResourceSnapshot_t{};
    snapshot->device_type = INFINI_DEVICE_CPU;
    snapshot->device_id = device_id;

    status = getMemInfo(device_id, &snapshot->free_bytes, &snapshot->total_bytes);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    if (snapshot->total_bytes >= snapshot->free_bytes) {
        snapshot->used_bytes = snapshot->total_bytes - snapshot->free_bytes;
    }
    snapshot->valid_fields = INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    return runtimeStatus(infini::rt::DeviceSynchronize());
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    if (stream_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::StreamCreate(stream_ptr));
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return runtimeStatus(infini::rt::StreamDestroy(stream));
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return runtimeStatus(infini::rt::StreamSynchronize(stream));
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return runtimeStatus(infini::rt::StreamWaitEvent(stream, event, 0));
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    if (event_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::EventCreate(event_ptr));
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    if (event_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::EventCreateWithFlags(event_ptr, flags));
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    if (event == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::EventRecord(event, stream));
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    if (event == nullptr || status_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto rt_status = infini::rt::EventQuery(event);
    *status_ptr = rt_status == infini::rt::kSuccess ? INFINIRT_EVENT_COMPLETE : INFINIRT_EVENT_NOT_READY;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    if (event == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::EventSynchronize(event));
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    if (event == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }
    return runtimeStatus(infini::rt::EventDestroy(event));
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    if (ms_ptr == nullptr || start == nullptr || end == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::EventElapsedTime(ms_ptr, start, end));
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    if (p_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = runtimeStatus(infini::rt::Malloc(p_ptr, size));
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    return size != 0 && *p_ptr == nullptr ? INFINI_STATUS_INTERNAL_ERROR : INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    if (p_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = runtimeStatus(infini::rt::MallocHost(p_ptr, size));
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    return size != 0 && *p_ptr == nullptr ? INFINI_STATUS_INTERNAL_ERROR : INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    return runtimeStatus(infini::rt::Free(ptr));
}

infiniStatus_t freeHost(void *ptr) {
    return runtimeStatus(infini::rt::FreeHost(ptr));
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    if ((dst == nullptr || src == nullptr) && size != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::Memcpy(dst, src, size, toRtMemcpyKind(kind)));
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    if ((dst == nullptr || src == nullptr) && size != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::MemcpyAsync(dst, src, size, toRtMemcpyKind(kind), stream));
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    if (p_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    auto status = runtimeStatus(infini::rt::MallocAsync(p_ptr, size, stream));
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    return size != 0 && *p_ptr == nullptr ? INFINI_STATUS_INTERNAL_ERROR : INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return runtimeStatus(infini::rt::FreeAsync(ptr, stream));
}

infiniStatus_t memsetDevice(void *ptr, int value, size_t count) {
    if (ptr == nullptr && count != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::Memset(ptr, value, count));
}

infiniStatus_t memsetDeviceAsync(void *ptr, int value, size_t count, infinirtStream_t stream) {
    if (ptr == nullptr && count != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }
    return runtimeStatus(infini::rt::MemsetAsync(ptr, value, count, stream));
}

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    (void)stream;
    (void)mode;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    (void)stream;
    (void)graph_ptr;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    (void)graph;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    (void)graph_exec_ptr;
    (void)graph;
    (void)node_ptr;
    (void)log_buffer;
    (void)buffer_size;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    (void)graph_exec;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    (void)graph_exec;
    (void)stream;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

} // namespace infinirt::cpu
