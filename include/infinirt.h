#ifndef __INFINIRT_API_H__
#define __INFINIRT_API_H__

#include "infinicore.h"

#include <cstddef>
#include <map>
#include <memory>
#include <variant>

typedef void *infinirtStream_t;
typedef void *infinirtEvent_t;

__C __export infiniStatus_t infinirtInit();

// Device
__C __export infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
__C __export infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
__C __export infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
__C __export infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
__C __export infiniStatus_t infinirtDeviceSynchronize();

// Stream
__C __export infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
__C __export infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__C __export infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
__C __export infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Event
typedef enum {
    INFINIRT_EVENT_COMPLETE = 0,
    INFINIRT_EVENT_NOT_READY = 1,
} infinirtEventStatus_t;

__C __export infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
__C __export infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__C __export infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
__C __export infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__C __export infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);

// Memory
typedef enum {
    INFINIRT_MEMCPY_H2H = 0,
    INFINIRT_MEMCPY_H2D = 1,
    INFINIRT_MEMCPY_D2H = 2,
    INFINIRT_MEMCPY_D2D = 3,
} infinirtMemcpyKind_t;

__C __export infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
__C __export infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
__C __export infiniStatus_t infinirtFree(void *ptr);
__C __export infiniStatus_t infinirtFreeHost(void *ptr);

__C __export infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
__C __export infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);

// Stream-ordered memory
__C __export infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
__C __export infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);

// Virtual memory & physical memory
typedef void *infinirtMemProp_t;
typedef void *infinirtDeviceptr_t;
typedef void *infinirtAllocationHandle_t;

// Represents a physical memory allocation, mirroring Rust's PhyMem.
struct infinirtPhyMem {
    infinirtAllocationHandle_t handle; // Opaque handle to physical memory
    size_t len;
    infinirtMemProp_t prop;
};

// Represents a vacant region, storing its length.
using infinirtVacantRegion = size_t;
// Represents a mapped region, holding a shared pointer to the physical memory object.
using infinirtMappedRegion = std::shared_ptr<infinirtPhyMem>;
// A region in virtual memory can be either mapped or vacant.
using infinirtPhyRegion = std::variant<infinirtMappedRegion, infinirtVacantRegion>;

struct infinirtVirtualMemManager {
    infinirtDeviceptr_t device_ptr;
    size_t len;
    // Maps offset to a physical region (mapped or vacant).
    std::map<size_t, infinirtPhyRegion> map;
};

__C __export infiniStatus_t infinirtGetMemProp(infinirtMemProp_t *prop, infiniDevice_t device, int device_id);
__C __export infiniStatus_t infinirtGetMemGranularityMinimum(size_t *granularity, infinirtMemProp_t prop);
__C __export infiniStatus_t infinirtCreatePhysicalMem(infinirtPhyMem *phy_mem, size_t len, infinirtMemProp_t prop);

__C __export infiniStatus_t infinirtCreateVirtualMemManager(infinirtVirtualMemManager *vm, infiniDevice_t device, size_t len, size_t min_addr);
__C __export infiniStatus_t infinirtMapVirtualMem(void **mapped_ptr, infinirtVirtualMemManager *vm, size_t offset, infinirtPhyMem *phy_mem);
__C __export infiniStatus_t infinirtUnmapVirtualMem(infinirtVirtualMemManager *vm, size_t offset);

#endif // __INFINIRT_API_H__
