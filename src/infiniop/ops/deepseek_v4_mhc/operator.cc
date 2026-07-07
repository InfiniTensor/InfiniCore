#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/deepseek_v4_mhc.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_mhc_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4MHCParamsDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCParamsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t pre_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    size_t sinkhorn_iters,
    float epsilon) {
#define CREATE(CASE, NAMESPACE)                                                                                 \
    case CASE:                                                                                                   \
        return op::deepseek_v4_mhc::NAMESPACE::ParamsDescriptor::create(                                         \
            handle, reinterpret_cast<op::deepseek_v4_mhc::NAMESPACE::ParamsDescriptor **>(desc_ptr),            \
            pre_desc, post_desc, comb_desc, mixes_desc, base_desc, scale_desc, sinkhorn_iters, epsilon)
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4MHCParamsWorkspaceSize(
    infiniopDeepseekV4MHCParamsDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                   \
        *size = reinterpret_cast<op::deepseek_v4_mhc::NAMESPACE::ParamsDescriptor *>(desc)->workspaceSize();     \
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

__INFINI_C infiniStatus_t infiniopDeepseekV4MHCParams(
    infiniopDeepseekV4MHCParamsDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *pre,
    void *post,
    void *comb,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                                   \
        return reinterpret_cast<op::deepseek_v4_mhc::NAMESPACE::ParamsDescriptor *>(desc)->calculate(            \
            workspace, workspace_size, pre, post, comb, mixes, base, scale, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4MHCParamsDescriptor(
    infiniopDeepseekV4MHCParamsDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                                 \
    case CASE:                                                                                                   \
        delete reinterpret_cast<op::deepseek_v4_mhc::NAMESPACE::ParamsDescriptor *>(desc);                       \
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
