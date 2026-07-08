#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/deepseek_v4_mhc_head.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_mhc_head_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4MHCHeadCollapseDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t mixes_desc,
    infiniopTensorDescriptor_t base_desc,
    infiniopTensorDescriptor_t scale_desc,
    float epsilon) {
#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::deepseek_v4_mhc_head::NAMESPACE::HeadCollapseDescriptor::create( \
            handle, reinterpret_cast<op::deepseek_v4_mhc_head::NAMESPACE::HeadCollapseDescriptor **>(desc_ptr), \
            y_desc, x_desc, mixes_desc, base_desc, scale_desc, epsilon)
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4MHCHeadCollapseWorkspaceSize(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                    \
    case CASE:                                                                  \
        *size = reinterpret_cast<op::deepseek_v4_mhc_head::NAMESPACE::HeadCollapseDescriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDeepseekV4MHCHeadCollapse(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *mixes,
    const void *base,
    const void *scale,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<op::deepseek_v4_mhc_head::NAMESPACE::HeadCollapseDescriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, mixes, base, scale, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4MHCHeadCollapseDescriptor(
    infiniopDeepseekV4MHCHeadCollapseDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::deepseek_v4_mhc_head::NAMESPACE::HeadCollapseDescriptor *>(desc); \
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
