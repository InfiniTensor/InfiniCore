#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/deepseek_v4_compressor.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/deepseek_v4_compressor_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDeepseekV4CompressorDescriptor(
    infiniopHandle_t handle,
    infiniopDeepseekV4CompressorDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t kv_desc,
    infiniopTensorDescriptor_t score_desc,
    infiniopTensorDescriptor_t ape_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    size_t compress_ratio,
    float epsilon) {
#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::deepseek_v4_compressor::NAMESPACE::Descriptor::create(                \
            handle, reinterpret_cast<op::deepseek_v4_compressor::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, kv_desc, score_desc, ape_desc, norm_weight_desc, compress_ratio, epsilon)
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

__INFINI_C infiniStatus_t infiniopGetDeepseekV4CompressorWorkspaceSize(
    infiniopDeepseekV4CompressorDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                           \
    case CASE:                                                                         \
        *size = reinterpret_cast<op::deepseek_v4_compressor::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDeepseekV4Compressor(
    infiniopDeepseekV4CompressorDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *kv,
    const void *score,
    const void *ape,
    const void *norm_weight,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<op::deepseek_v4_compressor::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, kv, score, ape, norm_weight, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDeepseekV4CompressorDescriptor(
    infiniopDeepseekV4CompressorDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                       \
    case CASE:                                                                         \
        delete reinterpret_cast<op::deepseek_v4_compressor::NAMESPACE::Descriptor *>(desc); \
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
