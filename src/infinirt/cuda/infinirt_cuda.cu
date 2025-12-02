#include "../../utils.h"
#include "infinirt_cuda.cuh"
#include <cuda_runtime.h>

#define CHECK_CUDART(RT_API) CHECK_INTERNAL(RT_API, cudaSuccess)

// Shared implementation for all CUDA-compatible devices
namespace infinirt::cuda_impl {

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
    CHECK_CUDART(cudaEventDestroy((cudaEvent_t)event));
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
} // namespace infinirt::cuda_impl

#ifdef ENABLE_NVIDIA_API
// NVIDIA namespace - wraps shared implementation
namespace infinirt::cuda {
infiniStatus_t getDeviceCount(int *count) {
    return infinirt::cuda_impl::getDeviceCount(count);
}

infiniStatus_t setDevice(int device_id) {
    return infinirt::cuda_impl::setDevice(device_id);
}

infiniStatus_t deviceSynchronize() {
    return infinirt::cuda_impl::deviceSynchronize();
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    return infinirt::cuda_impl::streamCreate(stream_ptr);
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamDestroy(stream);
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamSynchronize(stream);
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return infinirt::cuda_impl::streamWaitEvent(stream, event);
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    return infinirt::cuda_impl::eventCreate(event_ptr);
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return infinirt::cuda_impl::eventCreateWithFlags(event_ptr, flags);
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    return infinirt::cuda_impl::eventRecord(event, stream);
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    return infinirt::cuda_impl::eventQuery(event, status_ptr);
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventSynchronize(event);
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventDestroy(event);
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return infinirt::cuda_impl::eventElapsedTime(ms_ptr, start, end);
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocDevice(p_ptr, size);
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocHost(p_ptr, size);
}

infiniStatus_t freeDevice(void *ptr) {
    return infinirt::cuda_impl::freeDevice(ptr);
}

infiniStatus_t freeHost(void *ptr) {
    return infinirt::cuda_impl::freeHost(ptr);
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    return infinirt::cuda_impl::memcpy(dst, src, size, kind);
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    return infinirt::cuda_impl::memcpyAsync(dst, src, size, kind, stream);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return infinirt::cuda_impl::mallocAsync(p_ptr, size, stream);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return infinirt::cuda_impl::freeAsync(ptr, stream);
}
} // namespace infinirt::cuda
#endif

#ifdef ENABLE_ILUVATAR_API
// ILUVATAR namespace - wraps shared implementation
namespace infinirt::iluvatar {
infiniStatus_t getDeviceCount(int *count) {
    return infinirt::cuda_impl::getDeviceCount(count);
}

infiniStatus_t setDevice(int device_id) {
    return infinirt::cuda_impl::setDevice(device_id);
}

infiniStatus_t deviceSynchronize() {
    return infinirt::cuda_impl::deviceSynchronize();
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    return infinirt::cuda_impl::streamCreate(stream_ptr);
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamDestroy(stream);
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamSynchronize(stream);
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    return infinirt::cuda_impl::eventCreate(event_ptr);
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return infinirt::cuda_impl::eventCreateWithFlags(event_ptr, flags);
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    return infinirt::cuda_impl::eventRecord(event, stream);
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    return infinirt::cuda_impl::eventQuery(event, status_ptr);
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventSynchronize(event);
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventDestroy(event);
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return infinirt::cuda_impl::eventElapsedTime(ms_ptr, start, end);
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocDevice(p_ptr, size);
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocHost(p_ptr, size);
}

infiniStatus_t freeDevice(void *ptr) {
    return infinirt::cuda_impl::freeDevice(ptr);
}

infiniStatus_t freeHost(void *ptr) {
    return infinirt::cuda_impl::freeHost(ptr);
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    return infinirt::cuda_impl::memcpy(dst, src, size, kind);
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    return infinirt::cuda_impl::memcpyAsync(dst, src, size, kind, stream);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return infinirt::cuda_impl::mallocAsync(p_ptr, size, stream);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return infinirt::cuda_impl::freeAsync(ptr, stream);
}
} // namespace infinirt::iluvatar
#endif

#ifdef ENABLE_QY_API
// QY namespace - wraps shared implementation
namespace infinirt::qy {
infiniStatus_t getDeviceCount(int *count) {
    return infinirt::cuda_impl::getDeviceCount(count);
}

infiniStatus_t setDevice(int device_id) {
    return infinirt::cuda_impl::setDevice(device_id);
}

infiniStatus_t deviceSynchronize() {
    return infinirt::cuda_impl::deviceSynchronize();
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    return infinirt::cuda_impl::streamCreate(stream_ptr);
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamDestroy(stream);
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamSynchronize(stream);
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return infinirt::cuda_impl::streamWaitEvent(stream, event);
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    return infinirt::cuda_impl::eventCreate(event_ptr);
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return infinirt::cuda_impl::eventCreateWithFlags(event_ptr, flags);
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    return infinirt::cuda_impl::eventRecord(event, stream);
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    return infinirt::cuda_impl::eventQuery(event, status_ptr);
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventSynchronize(event);
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventDestroy(event);
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return infinirt::cuda_impl::eventElapsedTime(ms_ptr, start, end);
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocDevice(p_ptr, size);
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocHost(p_ptr, size);
}

infiniStatus_t freeDevice(void *ptr) {
    return infinirt::cuda_impl::freeDevice(ptr);
}

infiniStatus_t freeHost(void *ptr) {
    return infinirt::cuda_impl::freeHost(ptr);
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    return infinirt::cuda_impl::memcpy(dst, src, size, kind);
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    return infinirt::cuda_impl::memcpyAsync(dst, src, size, kind, stream);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return infinirt::cuda_impl::mallocAsync(p_ptr, size, stream);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return infinirt::cuda_impl::freeAsync(ptr, stream);
}
} // namespace infinirt::qy
#endif

#ifdef ENABLE_HYGON_API
// HYGON namespace - wraps shared implementation
namespace infinirt::hygon {
infiniStatus_t getDeviceCount(int *count) {
    return infinirt::cuda_impl::getDeviceCount(count);
}

infiniStatus_t setDevice(int device_id) {
    return infinirt::cuda_impl::setDevice(device_id);
}

infiniStatus_t deviceSynchronize() {
    return infinirt::cuda_impl::deviceSynchronize();
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    return infinirt::cuda_impl::streamCreate(stream_ptr);
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamDestroy(stream);
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return infinirt::cuda_impl::streamSynchronize(stream);
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return infinirt::cuda_impl::streamWaitEvent(stream, event);
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    return infinirt::cuda_impl::eventCreate(event_ptr);
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return infinirt::cuda_impl::eventCreateWithFlags(event_ptr, flags);
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    return infinirt::cuda_impl::eventRecord(event, stream);
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    return infinirt::cuda_impl::eventQuery(event, status_ptr);
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventSynchronize(event);
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    return infinirt::cuda_impl::eventDestroy(event);
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return infinirt::cuda_impl::eventElapsedTime(ms_ptr, start, end);
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocDevice(p_ptr, size);
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    return infinirt::cuda_impl::mallocHost(p_ptr, size);
}

infiniStatus_t freeDevice(void *ptr) {
    return infinirt::cuda_impl::freeDevice(ptr);
}

infiniStatus_t freeHost(void *ptr) {
    return infinirt::cuda_impl::freeHost(ptr);
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    return infinirt::cuda_impl::memcpy(dst, src, size, kind);
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    return infinirt::cuda_impl::memcpyAsync(dst, src, size, kind, stream);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return infinirt::cuda_impl::mallocAsync(p_ptr, size, stream);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return infinirt::cuda_impl::freeAsync(ptr, stream);
}
} // namespace infinirt::hygon
#endif
