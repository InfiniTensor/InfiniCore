#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/softmax.h"

#ifdef ENABLE_CPU_API
#include "cpu/softmax_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/softmax_cuda.cuh"
#endif

__C __export infiniStatus_t infiniopCreateSoftmaxDescriptor(infiniopHandle_t handle,
                                                            infiniopSoftmaxDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y_desc,
                                                            infiniopTensorDescriptor_t x_desc,
                                                            int axis) {
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::softmax::NAMESPACE::Descriptor::create(                     \
            handle,                                                            \
            reinterpret_cast<op::softmax::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                            \
            x_desc,                                                            \                                                 
            axis)
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t
infiniopGetSoftmaxWorkspaceSize(
    infiniopSoftmaxDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                       \
        *size = reinterpret_cast<const op::softmax::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopSoftmax(
    infiniopSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                    \
        return reinterpret_cast<const op::softmax::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                \
                        y,                                                        \
                        x,                                                        \
                        stream)
    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroySoftmaxDescriptor(infiniopSoftmaxDescriptor_t desc) {
#define DELETE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        delete reinterpret_cast<const op::softmax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
