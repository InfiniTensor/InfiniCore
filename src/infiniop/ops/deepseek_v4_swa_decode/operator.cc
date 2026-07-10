#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/deepseek_v4_swa_decode.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_swa_decode_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4SwaDecodeDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4SwaDecodeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t attn_sink_desc,
    float softmax_scale) {
#define CREATE(CASE, NAMESPACE) \
    case CASE: \
        return op::deepseek_v4_swa_decode::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::deepseek_v4_swa_decode::NAMESPACE::Descriptor **>(desc_ptr), y_desc, q_desc, k_desc, attn_sink_desc, softmax_scale)
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4SwaDecodeWorkspaceSize(
    infiniopDeepseekV4SwaDecodeDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE) \
    case CASE: \
        *size = reinterpret_cast<op::deepseek_v4_swa_decode::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDeepseekV4SwaDecode(
    infiniopDeepseekV4SwaDecodeDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *q,
    const void *k,
    const void *attn_sink,
    void *stream) {
#define CALC(CASE, NAMESPACE) \
    case CASE: \
        return reinterpret_cast<op::deepseek_v4_swa_decode::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, y, q, k, attn_sink, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALC(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4SwaDecodeDescriptor(
    infiniopDeepseekV4SwaDecodeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE) \
    case CASE: \
        delete reinterpret_cast<op::deepseek_v4_swa_decode::NAMESPACE::Descriptor *>(desc); \
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
