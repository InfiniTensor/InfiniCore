#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_sglang_hash_topk.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_sglang_hash_topk_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDsv4SglangHashTopkDescriptor(infiniopHandle_t handle, infiniopDsv4SglangHashTopkDescriptor_t *desc_ptr, infiniopTensorDescriptor_t router_logits_desc, infiniopTensorDescriptor_t input_ids_desc, infiniopTensorDescriptor_t tid2eid_desc, infiniopTensorDescriptor_t topk_weights_desc, infiniopTensorDescriptor_t topk_ids_desc, float routed_scaling_factor) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_sglang_hash_topk::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_sglang_hash_topk::NAMESPACE::Descriptor **>(desc_ptr), router_logits_desc, input_ids_desc, tid2eid_desc, topk_weights_desc, topk_ids_desc, routed_scaling_factor)
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

__INFINI_C infiniStatus_t infiniopGetDsv4SglangHashTopkWorkspaceSize(infiniopDsv4SglangHashTopkDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                 \
    case CASE:                                                                                               \
        *size = reinterpret_cast<op::dsv4_sglang_hash_topk::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
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

__INFINI_C infiniStatus_t infiniopDsv4SglangHashTopk(infiniopDsv4SglangHashTopkDescriptor_t desc, void *workspace, size_t workspace_size, const void *router_logits, const void *input_ids, const void *tid2eid, void *topk_weights, void *topk_ids, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_sglang_hash_topk::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, router_logits, input_ids, tid2eid, topk_weights, topk_ids, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDsv4SglangHashTopkDescriptor(infiniopDsv4SglangHashTopkDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                           \
    case CASE:                                                                             \
        delete reinterpret_cast<op::dsv4_sglang_hash_topk::NAMESPACE::Descriptor *>(desc); \
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
