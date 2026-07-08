#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/deepseek_v4_router.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_router_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4TopkRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4TopkRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool renormalize) {
#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::deepseek_v4_router::NAMESPACE::TopkRouterDescriptor::create(                 \
            handle, reinterpret_cast<op::deepseek_v4_router::NAMESPACE::TopkRouterDescriptor **>(desc_ptr), \
            topk_weights_desc, topk_indices_desc, logits_desc, bias_desc, renormalize)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4TopkRouterWorkspaceSize(
    infiniopDeepseekV4TopkRouterDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                          \
    case CASE:                                                                                                        \
        *size = reinterpret_cast<op::deepseek_v4_router::NAMESPACE::TopkRouterDescriptor *>(desc)->workspaceSize();   \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopDeepseekV4TopkRouter(
    infiniopDeepseekV4TopkRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *bias,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                                   \
    case CASE:                                                                                                       \
        return reinterpret_cast<op::deepseek_v4_router::NAMESPACE::TopkRouterDescriptor *>(desc)->calculate(         \
            workspace, workspace_size, topk_weights, topk_indices, logits, bias, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4TopkRouterDescriptor(
    infiniopDeepseekV4TopkRouterDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                       \
        delete reinterpret_cast<op::deepseek_v4_router::NAMESPACE::TopkRouterDescriptor *>(desc);                    \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4HashRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4HashRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize) {
#define CREATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        return op::deepseek_v4_router::NAMESPACE::HashRouterDescriptor::create(                 \
            handle, reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashRouterDescriptor **>(desc_ptr), \
            topk_weights_desc, topk_indices_desc, logits_desc, input_ids_desc, tid2eid_desc, renormalize)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4HashRouterWorkspaceSize(
    infiniopDeepseekV4HashRouterDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                         \
    case CASE:                                                                                                       \
        *size = reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashRouterDescriptor *>(desc)->workspaceSize();  \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopDeepseekV4HashRouter(
    infiniopDeepseekV4HashRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *input_ids,
    const void *tid2eid,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                                  \
    case CASE:                                                                                                      \
        return reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashRouterDescriptor *>(desc)->calculate(        \
            workspace, workspace_size, topk_weights, topk_indices, logits, input_ids, tid2eid, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4HashRouterDescriptor(
    infiniopDeepseekV4HashRouterDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                                    \
    case CASE:                                                                                                      \
        delete reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashRouterDescriptor *>(desc);                   \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
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


__INFINI_C infiniStatus_t infiniopCreateDeepseekV4HashTopkRouterDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4HashTopkRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize) {
#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::deepseek_v4_router::NAMESPACE::HashTopkRouterDescriptor::create(                  \
            handle, reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashTopkRouterDescriptor **>(desc_ptr), \
            topk_weights_desc, topk_indices_desc, hidden_states_desc, weight_desc, input_ids_desc, tid2eid_desc, renormalize)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4HashTopkRouterWorkspaceSize(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                    \
    case CASE:                                                                  \
        *size = reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashTopkRouterDescriptor *>(desc)->workspaceSize();  \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopDeepseekV4HashTopkRouter(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *hidden_states,
    const void *weight,
    const void *input_ids,
    const void *tid2eid,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashTopkRouterDescriptor *>(desc)->calculate(        \
            workspace, workspace_size, topk_weights, topk_indices, hidden_states, weight, input_ids, tid2eid, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4HashTopkRouterDescriptor(
    infiniopDeepseekV4HashTopkRouterDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::deepseek_v4_router::NAMESPACE::HashTopkRouterDescriptor *>(desc);                   \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
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
