#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_topk_sigmoid.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_topk_sigmoid_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeTopkSigmoidDescriptor(
    infiniopHandle_t handle,
    infiniopMoeTopkSigmoidDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t gating_output_desc,
    infiniopTensorDescriptor_t correction_bias_desc,
    bool renormalize) {
#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::moe_topk_sigmoid::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::moe_topk_sigmoid::NAMESPACE::Descriptor **>(desc_ptr), \
            topk_weights_desc, topk_indices_desc, gating_output_desc, correction_bias_desc, renormalize)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeTopkSigmoidWorkspaceSize(
    infiniopMoeTopkSigmoidDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<op::moe_topk_sigmoid::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopMoeTopkSigmoid(
    infiniopMoeTopkSigmoidDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                   \
        return reinterpret_cast<op::moe_topk_sigmoid::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, topk_weights, topk_indices, gating_output, correction_bias, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeTopkSigmoidDescriptor(
    infiniopMoeTopkSigmoidDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                      \
    case CASE:                                                                        \
        delete reinterpret_cast<op::moe_topk_sigmoid::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
