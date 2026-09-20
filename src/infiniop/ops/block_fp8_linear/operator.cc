#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/block_fp8_linear.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/block_fp8_linear_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateBlockFP8LinearDescriptor(
    infiniopHandle_t handle,
    infiniopBlockFP8LinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc) {
#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                            \
        return op::block_fp8_linear::NAMESPACE::Descriptor::create(                      \
            handle,                                                                       \
            reinterpret_cast<op::block_fp8_linear::NAMESPACE::Descriptor **>(desc_ptr),    \
            output_desc, input_desc, weight_desc, weight_scale_desc)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetBlockFP8LinearWorkspaceSize(
    infiniopBlockFP8LinearDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                               \
        *size = reinterpret_cast<const op::block_fp8_linear::NAMESPACE::Descriptor *>(desc)  \
                    ->workspaceSize();                                                       \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopBlockFP8Linear(
    infiniopBlockFP8LinearDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *weight_scale,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                                \
        return reinterpret_cast<const op::block_fp8_linear::NAMESPACE::Descriptor *>(desc)    \
            ->calculate(workspace, workspace_size, output, input, weight, weight_scale, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyBlockFP8LinearDescriptor(
    infiniopBlockFP8LinearDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                \
        delete reinterpret_cast<const op::block_fp8_linear::NAMESPACE::Descriptor *>(desc);   \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
