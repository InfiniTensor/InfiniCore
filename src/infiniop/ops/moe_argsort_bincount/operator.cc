#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_argsort_bincount.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/moe_argsort_bincount_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateMoeArgsortBincountDescriptor(
    infiniopHandle_t handle,
    infiniopMoeArgsortBincountDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_indices_desc,
    infiniopTensorDescriptor_t inv_pos_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts) {
#define CREATE(CASE, NAMESPACE)                                         \
    case CASE:                                                          \
        return op::moe_argsort_bincount::NAMESPACE::Descriptor::create( \
            handle,                                                     \
            reinterpret_cast<                                           \
                op::moe_argsort_bincount::NAMESPACE::Descriptor **>(    \
                desc_ptr),                                              \
            tokens_per_experts_desc, sorted_indices_desc, inv_pos_desc, \
            topk_ids_desc, num_experts)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetMoeArgsortBincountWorkspaceSize(
    infiniopMoeArgsortBincountDescriptor_t desc,
    size_t *size) {
    *size = reinterpret_cast<
                op::moe_argsort_bincount::nvidia::Descriptor *>(desc)
                ->workspaceSize();
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infiniopMoeArgsortBincount(
    infiniopMoeArgsortBincountDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *tokens_per_experts,
    void *sorted_indices,
    void *inv_pos,
    const void *topk_ids,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                           \
    case CASE:                                                               \
        return reinterpret_cast<                                             \
                   const op::moe_argsort_bincount::NAMESPACE::Descriptor *>( \
                   desc)                                                     \
            ->calculate(workspace, workspace_size, tokens_per_experts,       \
                        sorted_indices, inv_pos, topk_ids, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyMoeArgsortBincountDescriptor(
    infiniopMoeArgsortBincountDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                            \
    case CASE:                                                              \
        delete reinterpret_cast<                                            \
            const op::moe_argsort_bincount::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
