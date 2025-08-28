#include "../../utils.h"
#include "infinirt_cuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#define CHECK_CUDART(RT_API) CHECK_INTERNAL(RT_API, cudaSuccess)

namespace infinirt::cuda {

infiniStatus_t getDeviceCount(int *count) {
    CHECK_CUDART(cudaGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_CUDART(cudaSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
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
    CHECK_CUDART(cudaStreamDestroy((cudaStream_t)stream));
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
    CHECK_CUDART(cudaEventDestroy((cudaEvent_t)event));
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
    CHECK_CUDART(cudaFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_CUDART(cudaFreeHost(ptr));
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
    CHECK_CUDART(cudaFreeAsync(ptr, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

CUmemAllocationProp *getMemProp() {
    int device_id;
    infinirtGetDevice(nullptr, &device_id);
    CUmemAllocationProp *cuda_prop = new CUmemAllocationProp();
    memset(cuda_prop, 0, sizeof(CUmemAllocationProp));
    cuda_prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    cuda_prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    cuda_prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    cuda_prop->location.id = device_id;
    return cuda_prop;
}

infiniStatus_t getMemGranularityMinimum(size_t *granularity) {
    CUmemAllocationProp *cuda_prop = getMemProp();
    CHECK_CUDART(cuMemGetAllocationGranularity(granularity, cuda_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t createPhysicalMem(infinirtPhysicalMemoryHandle_t *pm_handle, size_t len) {
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp *cuda_prop = getMemProp();
    CHECK_CUDART(cuMemCreate(&handle, len, cuda_prop, 0));

    *pm_handle = (infinirtPhysicalMemoryHandle_t)handle;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t releasePhysicalMem(infinirtPhysicalMemoryHandle_t pm_handle) {
    CHECK_CUDART(cuMemRelease((CUmemGenericAllocationHandle)pm_handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t createVirtualMem(void **vm, size_t len) {
    CUdeviceptr device_ptr;
    CHECK_CUDART(cuMemAddressReserve(&device_ptr, len, 0, (CUdeviceptr)0, 0));

    *vm = (void *)device_ptr;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t releaseVirtualMem(void *vm, size_t len) {
    CHECK_CUDART(cuMemAddressFree((CUdeviceptr)vm, len));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mapVirtualMem(void *vm, size_t len, size_t offset,
                             infinirtPhysicalMemoryHandle_t pm_handle) {

    CUdeviceptr ptr = (CUdeviceptr)vm + offset;
    CHECK_CUDART(cuMemMap(ptr, len, 0, (CUmemGenericAllocationHandle)pm_handle, 0));

    CUmemAllocationProp *cuda_prop = getMemProp();
    CUmemAccessDesc desc = {};
    desc.location = cuda_prop->location;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDART(cuMemSetAccess(ptr, len, &desc, 1));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t unmapVirtualMem(void *vm, size_t len) {
    CUdeviceptr ptr = (CUdeviceptr)vm;
    CHECK_CUDART(cuMemUnmap(ptr, len));

    return INFINI_STATUS_SUCCESS;
}

} // namespace infinirt::cuda
