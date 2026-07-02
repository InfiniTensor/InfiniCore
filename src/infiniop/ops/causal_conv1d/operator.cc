#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/causal_conv1d.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/causal_conv1d_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateCausalConv1dDescriptor(
    infiniopHandle_t handle,
    infiniopCausalConv1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t conv_state_desc,
    infiniopTensorDescriptor_t final_conv_state_desc,
    infiniopTensorDescriptor_t qkv_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::causal_conv1d::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::causal_conv1d::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, conv_state_desc, final_conv_state_desc,                        \
            qkv_desc, weight_desc, bias_desc, cu_seqlens_desc,                       \
            initial_state_indices_desc, final_state_indices_desc);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetCausalConv1dWorkspaceSize(
    infiniopCausalConv1dDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                       \
        *size = reinterpret_cast<op::causal_conv1d::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopCausalConv1d(
    infiniopCausalConv1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    void *conv_state,
    void *final_conv_state,
    const void *qkv,
    const void *weight,
    const void *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                                \
        return reinterpret_cast<op::causal_conv1d::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, conv_state, final_conv_state,                     \
            qkv, weight, bias, cu_seqlens, initial_state_indices,                             \
            final_state_indices, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyCausalConv1dDescriptor(
    infiniopCausalConv1dDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                   \
    case CASE:                                                                     \
        delete reinterpret_cast<op::causal_conv1d::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
