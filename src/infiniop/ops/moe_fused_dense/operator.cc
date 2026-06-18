#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_fused_dense.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_fused_dense_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeFusedDenseDescriptor(
    infiniopHandle_t handle,
    infiniopMoeFusedDenseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t w13_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc) {
#define CREATE(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                    \
        return op::moe_fused_dense::NAMESPACE::Descriptor::create(                                \
            handle, reinterpret_cast<op::moe_fused_dense::NAMESPACE::Descriptor **>(desc_ptr),    \
            output_desc, hidden_states_desc, w13_desc, w2_desc, topk_weights_desc, topk_ids_desc, \
            sorted_token_ids_desc, expert_ids_desc, num_tokens_post_padded_desc)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeFusedDenseWorkspaceSize(
    infiniopMoeFusedDenseDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                           \
    case CASE:                                                                                         \
        *size = reinterpret_cast<op::moe_fused_dense::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopMoeFusedDense(
    infiniopMoeFusedDenseDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *hidden_states,
    const void *w13,
    const void *w2,
    const void *topk_weights,
    const void *topk_ids,
    const void *sorted_token_ids,
    const void *expert_ids,
    const void *num_tokens_post_padded,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                              \
    case CASE:                                                                                  \
        return reinterpret_cast<op::moe_fused_dense::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, output, hidden_states, w13, w2, topk_weights, topk_ids,  \
            sorted_token_ids, expert_ids, num_tokens_post_padded, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeFusedDenseDescriptor(
    infiniopMoeFusedDenseDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                     \
    case CASE:                                                                       \
        delete reinterpret_cast<op::moe_fused_dense::NAMESPACE::Descriptor *>(desc); \
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
