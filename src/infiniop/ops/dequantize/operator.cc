#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dequantize.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequantize_w42f16_nvidia.cuh"
#endif

__C infiniStatus_t infiniopCreateDequantizeDescriptor(
    infiniopHandle_t handle,
    infiniopDequantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t zeros_desc) {

#define CREATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                    \
        return op::dequantize::NAMESPACE::Descriptor::create(                     \
            handle,                                                               \
            reinterpret_cast<op::dequantize::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                             \
            qweight_desc,                                                         \
            scales_desc,                                                          \
            zeros_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetDequantizeWorkspaceSize(infiniopDequantizeDescriptor_t desc,
                                                      size_t *size) {
#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<const op::dequantize::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__C infiniStatus_t infiniopDequantize(
    infiniopDequantizeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *qweight,
    const void *scales,
    const void *zeros,
    size_t split_k_iters,
    size_t thx,
    size_t thy,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                   \
    case CASE:                                                                       \
        return reinterpret_cast<const op::dequantize::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, qweight, scales, zeros, split_k_iters, thx, thy, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyDequantizeDescriptor(infiniopDequantizeDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        delete reinterpret_cast<const op::dequantize::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}

// #endif