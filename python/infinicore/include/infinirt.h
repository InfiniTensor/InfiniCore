#ifndef __INFINIRT_API_H__
#define __INFINIRT_API_H__

#include "infinicore.h"
#include <stdint.h>

typedef void *infinirtStream_t;
typedef void *infinirtEvent_t;
typedef void *infinirtGraph_t;
typedef void *infinirtGraphNode_t;
typedef void *infinirtGraphExec_t;

// Bitmask describing which fields of an infinirtDeviceResourceSnapshot_t
// have been populated by the backend. Backends without a particular
// capability simply leave the flag clear.
typedef enum {
    INFINIRT_RESOURCE_FIELD_NONE = 0,
    INFINIRT_RESOURCE_FIELD_MEMORY_CAPACITY = 1ull << 0,
    INFINIRT_RESOURCE_FIELD_COMPUTE_UTILIZATION = 1ull << 1,
    INFINIRT_RESOURCE_FIELD_MEMORY_BANDWIDTH_UTILIZATION = 1ull << 2,
    INFINIRT_RESOURCE_FIELD_KERNEL_TIME_RATIO = 1ull << 3,
    INFINIRT_RESOURCE_FIELD_COMMUNICATION = 1ull << 4,
} infinirtResourceField_t;

// Vendor-neutral resource snapshot consumed by the optional mutual
// awareness analyzer. Backends populate the fields they can observe and
// set the matching `valid_fields` bits; consumers must check the bits
// before relying on a value.
typedef struct {
    infiniDevice_t device_type;
    int device_id;

    uint64_t valid_fields;
    uint64_t estimated_fields;

    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    size_t reserved_bytes;

    float compute_utilization;
    float memory_bandwidth_utilization;
    float kernel_time_ratio;

    float communication_time_ratio;
    uint64_t communication_bytes;
} infinirtDeviceResourceSnapshot_t;

__INFINI_C __export infiniStatus_t infinirtInit();

// Device
__INFINI_C __export infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
__INFINI_C __export infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
__INFINI_C __export infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
__INFINI_C __export infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
__INFINI_C __export infiniStatus_t infinirtGetMemInfo(infiniDevice_t device, int device_id, size_t *free_bytes, size_t *total_bytes);
__INFINI_C __export infiniStatus_t infinirtGetDeviceResourceSnapshot(
    infiniDevice_t device,
    int device_id,
    infinirtDeviceResourceSnapshot_t *snapshot);
__INFINI_C __export infiniStatus_t infinirtDeviceSynchronize();

// Stream
__INFINI_C __export infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
__INFINI_C __export infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Event
typedef enum {
    INFINIRT_EVENT_COMPLETE = 0,
    INFINIRT_EVENT_NOT_READY = 1,
} infinirtEventStatus_t;

// Event flags for precise timing
typedef enum {
    INFINIRT_EVENT_DEFAULT = 0x0,        // Default event creation flags
    INFINIRT_EVENT_DISABLE_TIMING = 0x1, // Event will not record timing data
    INFINIRT_EVENT_BLOCKING_SYNC = 0x2,  // Event uses blocking synchronization
} infinirtEventFlags_t;

__INFINI_C __export infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
__INFINI_C __export infiniStatus_t infinirtEventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags);
__INFINI_C __export infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
__INFINI_C __export infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__INFINI_C __export infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);
__INFINI_C __export infiniStatus_t infinirtEventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end);

// Memory
typedef enum {
    INFINIRT_MEMCPY_H2H = 0,
    INFINIRT_MEMCPY_H2D = 1,
    INFINIRT_MEMCPY_D2H = 2,
    INFINIRT_MEMCPY_D2D = 3,
} infinirtMemcpyKind_t;

__INFINI_C __export infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
__INFINI_C __export infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
__INFINI_C __export infiniStatus_t infinirtFree(void *ptr);
__INFINI_C __export infiniStatus_t infinirtFreeHost(void *ptr);

__INFINI_C __export infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
__INFINI_C __export infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);

// Stream-ordered memory
__INFINI_C __export infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);

// Memset
__INFINI_C __export infiniStatus_t infinirtMemset(void *ptr, int value, size_t count);
__INFINI_C __export infiniStatus_t infinirtMemsetAsync(void *ptr, int value, size_t count, infinirtStream_t stream);

// Graph
typedef enum {
    INFINIRT_STREAM_CAPTURE_MODE_GLOBAL = 0,
    INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    INFINIRT_STREAM_CAPTURE_MODE_RELAXED = 2,

} infinirtStreamCaptureMode_t;

__INFINI_C __export infiniStatus_t infinirtStreamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode);
__INFINI_C __export infiniStatus_t infinirtStreamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr);
__INFINI_C __export infiniStatus_t infinirtGraphDestroy(infinirtGraph_t graph);
__INFINI_C __export infiniStatus_t infinirtGraphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size);
__INFINI_C __export infiniStatus_t infinirtGraphExecDestroy(infinirtGraphExec_t graph_exec);
__INFINI_C __export infiniStatus_t infinirtGraphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream);

#endif // __INFINIRT_API_H__
