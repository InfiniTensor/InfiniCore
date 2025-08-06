#include "../../utils.h"
#include "infinirt_cuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string.h>
#include <variant>

#define CHECK_CUDART(RT_API) CHECK_INTERNAL(RT_API, cudaSuccess)

namespace infinirt::cuda {
// Internal struct definitions for opaque pointers
struct PhyMemImpl {
    infinirtAllocationHandle_t handle;
    size_t len;
    CUmemAllocationProp *prop;
};

using VacantRegion = size_t;
using MappedRegion = std::shared_ptr<PhyMemImpl>;
using PhyRegion = std::variant<MappedRegion, VacantRegion>;

struct VirtualMemManagerImpl {
    infinirtDeviceptr_t device_ptr;
    size_t len;
    std::map<size_t, PhyRegion> map;
};

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

infiniStatus_t createPhysicalMem(infinirtPhyMem_t *phy_mem, size_t len) {
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp *cuda_prop = getMemProp();
    CHECK_CUDART(cuMemCreate(&handle, len, cuda_prop, 0));

    PhyMemImpl *impl = new PhyMemImpl;
    impl->handle = (infinirtAllocationHandle_t)handle;
    impl->len = len;
    impl->prop = cuda_prop;

    *phy_mem = (infinirtPhyMem_t)impl;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t releasePhysicalMem(infinirtPhyMem_t phy_mem) {
    PhyMemImpl *impl = (PhyMemImpl *)phy_mem;
    CHECK_CUDART(cuMemRelease((CUmemGenericAllocationHandle)impl->handle));
    delete impl->prop;
    delete impl;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t createVirtualMem(infinirtVirtualMem_t *vm, size_t len) {
    CUdeviceptr device_ptr;
    CHECK_CUDART(cuMemAddressReserve(&device_ptr, len, 0, (CUdeviceptr)0, 0));

    VirtualMemManagerImpl *impl = new VirtualMemManagerImpl;
    impl->device_ptr = (infinirtDeviceptr_t)device_ptr;
    impl->len = len;
    impl->map.clear();
    impl->map[0] = VacantRegion(len);

    *vm = (infinirtVirtualMem_t)impl;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t releaseVirtualMem(infinirtVirtualMem_t vm) {
    VirtualMemManagerImpl *impl = (VirtualMemManagerImpl *)vm;

    // First unmap all mapped regions
    for (auto &[offset, region] : impl->map) {
        if (auto *mapped = std::get_if<MappedRegion>(&region)) {
            CUdeviceptr ptr = (CUdeviceptr)impl->device_ptr + offset;
            CHECK_CUDART(cuMemUnmap(ptr, (*mapped)->len));
        }
    }

    // Then free the virtual address space
    CHECK_CUDART(cuMemAddressFree((CUdeviceptr)impl->device_ptr, impl->len));

    delete impl;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mapVirtualMem(void **mapped_ptr, infinirtVirtualMem_t vm, size_t offset,
                             infinirtPhyMem_t phy_mem) {
    VirtualMemManagerImpl *vm_impl = (VirtualMemManagerImpl *)vm;
    PhyMemImpl *phy_impl = (PhyMemImpl *)phy_mem;

    if (offset > vm_impl->len || offset + phy_impl->len > vm_impl->len) {
        std::cerr << "Offset is out of range"
                  << " offset: " << offset << " phy_mem->len: " << phy_impl->len << " vm->len: " << vm_impl->len << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }
    auto it = vm_impl->map.upper_bound(offset);
    --it;
    auto &[head, region] = *it;

    if (auto *vacant = std::get_if<VacantRegion>(&region)) {
        if (phy_impl->len > *vacant) {
            std::cerr << "Physical memory length is greater than the vacant region length" << std::endl;
            return INFINI_STATUS_BAD_PARAM;
        }

        CUdeviceptr ptr = (CUdeviceptr)vm_impl->device_ptr + offset;
        CHECK_CUDART(cuMemMap(ptr, phy_impl->len, 0, (CUmemGenericAllocationHandle)phy_impl->handle, 0));
        CUmemAccessDesc desc = {};
        desc.location = phy_impl->prop->location;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CHECK_CUDART(cuMemSetAccess(ptr, phy_impl->len, &desc, 1));

        vm_impl->map.erase(it);
        vm_impl->map[offset] = std::make_shared<PhyMemImpl>(*phy_impl);
        auto head_len = offset - head;
        auto tail_len = *vacant - head_len - phy_impl->len;
        if (head_len > 0) {
            vm_impl->map[head] = head_len;
        }
        if (tail_len > 0) {
            vm_impl->map[head + head_len + phy_impl->len] = tail_len;
        }

        *mapped_ptr = (void *)ptr;
        return INFINI_STATUS_SUCCESS;
    } else {
        std::cerr << "Virtual memory already mapped at offset: " << offset << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

infiniStatus_t unmapVirtualMem(infinirtVirtualMem_t vm, size_t offset) {
    VirtualMemManagerImpl *vm_impl = (VirtualMemManagerImpl *)vm;
    auto it = vm_impl->map.find(offset);
    if (it == vm_impl->map.end()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (auto *mapped = std::get_if<MappedRegion>(&it->second)) {
        auto phy_mem = *mapped;
        auto ptr = (CUdeviceptr)vm_impl->device_ptr + offset;
        CHECK_CUDART(cuMemUnmap(ptr, phy_mem->len));

        it->second = phy_mem->len;
        return INFINI_STATUS_SUCCESS;
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }
}

} // namespace infinirt::cuda
