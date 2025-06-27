#include "../../handle.h"
#include "infiniop/ops/spmv.h"
#include "spmv.h"

#ifdef ENABLE_CPU_API
#include "cpu/spmv_cpu.h"
#endif

#ifdef ENABLE_CUDA_API
#include "cuda/spmv_cuda.cuh"
#endif

#ifdef ENABLE_CAMBRICON_API
#include "bang/spmv_bang.h"
#endif

__C infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle, infiniopSpMVDescriptor_t *desc_ptr,
    size_t num_cols, size_t num_rows, size_t nnz, infiniDtype_t dtype) {
#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::spmv::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::spmv::NAMESPACE::Descriptor **>(desc_ptr), \
            num_cols, num_rows, nnz, dtype)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopSpMV(infiniopSpMVDescriptor_t desc, void *y,
                                const void *x, const void *values,
                                const void *row_ptr, const void *col_indices,
                                void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc) \
            ->calculate(y, x, values, row_ptr, col_indices, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySpMVDescriptor(infiniopSpMVDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::spmv::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
