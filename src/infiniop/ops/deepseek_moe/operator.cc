#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/deepseek_moe.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/deepseek_moe_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekMoeDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekMoeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t hidden_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    size_t intermediate_size,
    size_t num_experts) {

#define CREATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                              \
        return op::deepseek_moe::NAMESPACE::Descriptor::create(                             \
            handle, reinterpret_cast<op::deepseek_moe::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, hidden_desc, topk_indices_desc, topk_weights_desc,                    \
            intermediate_size, num_experts)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetDeepseekMoeWorkspaceSize(
    infiniopDeepseekMoeDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::deepseek_moe::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDeepseekMoe(
    infiniopDeepseekMoeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *const *gate_weights,
    const void *const *up_weights,
    const void *const *down_weights,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<op::deepseek_moe::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, hidden, topk_indices, topk_weights,              \
            gate_weights, up_weights, down_weights, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDeepseekMoeWithDevicePtrs(
    infiniopDeepseekMoeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *hidden,
    const void *topk_indices,
    const void *topk_weights,
    const void *gate_weight_ptrs,
    const void *up_weight_ptrs,
    const void *down_weight_ptrs,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                             \
        return reinterpret_cast<op::deepseek_moe::NAMESPACE::Descriptor *>(desc)->calculateWithDevicePtrs( \
            workspace, workspace_size, out, hidden, topk_indices, topk_weights,                            \
            gate_weight_ptrs, up_weight_ptrs, down_weight_ptrs, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyDeepseekMoeDescriptor(
    infiniopDeepseekMoeDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        delete reinterpret_cast<op::deepseek_moe::NAMESPACE::Descriptor *>(desc); \
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
