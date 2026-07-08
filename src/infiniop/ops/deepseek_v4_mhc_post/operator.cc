#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/deepseek_v4_mhc_post.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_mhc_post_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4MHCPostDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4MHCPostDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t new_x_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t post_desc,
    infiniopTensorDescriptor_t comb_desc) {
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::deepseek_v4_mhc_post::NAMESPACE::PostDescriptor::create(     \
            handle, reinterpret_cast<op::deepseek_v4_mhc_post::NAMESPACE::PostDescriptor **>(desc_ptr), \
            y_desc, new_x_desc, residual_desc, post_desc, comb_desc)
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4MHCPostWorkspaceSize(
    infiniopDeepseekV4MHCPostDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                   \
    case CASE:                                                                 \
        *size = reinterpret_cast<op::deepseek_v4_mhc_post::NAMESPACE::PostDescriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDeepseekV4MHCPost(
    infiniopDeepseekV4MHCPostDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *new_x,
    const void *residual,
    const void *post,
    const void *comb,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<op::deepseek_v4_mhc_post::NAMESPACE::PostDescriptor *>(desc)->calculate( \
            workspace, workspace_size, y, new_x, residual, post, comb, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4MHCPostDescriptor(
    infiniopDeepseekV4MHCPostDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                               \
    case CASE:                                                                 \
        delete reinterpret_cast<op::deepseek_v4_mhc_post::NAMESPACE::PostDescriptor *>(desc); \
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
