#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_align.h"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/moe_align_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeAlignDescriptor(
    infiniopHandle_t handle,
    infiniopMoeAlignDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts,
    size_t block_size) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::moe_align::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::moe_align::NAMESPACE::Descriptor **>(desc_ptr), \
            sorted_token_ids_desc,                                               \
            expert_ids_desc,                                                     \
            num_tokens_post_padded_desc,                                         \
            topk_ids_desc,                                                       \
            num_experts,                                                         \
            block_size)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeAlignWorkspaceSize(
    infiniopMoeAlignDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::moe_align::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopMoeAlign(
    infiniopMoeAlignDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    const void *topk_ids,
    const void *expert_map,
    int pad_sorted_token_ids,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                               \
        return reinterpret_cast<op::moe_align::NAMESPACE::Descriptor *>(desc)->calculate(    \
            workspace, workspace_size, sorted_token_ids, expert_ids, num_tokens_post_padded, \
            topk_ids, expert_map, static_cast<bool>(pad_sorted_token_ids), stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeAlignDescriptor(
    infiniopMoeAlignDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<op::moe_align::NAMESPACE::Descriptor *>(desc); \
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
