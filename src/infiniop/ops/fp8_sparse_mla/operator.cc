#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/fp8_sparse_mla.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)
#include "nvidia/fp8_sparse_mla_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateFp8SparseMlaDescriptor(
    infiniopHandle_t handle,
    infiniopFp8SparseMlaDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t topk_lens_desc,
    float scale) {
#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::fp8_sparse_mla::NAMESPACE::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::fp8_sparse_mla::NAMESPACE::Descriptor **>(desc_ptr), \
            output_desc, query_desc, kv_cache_desc, indices_desc,                     \
            topk_lens_desc, scale)
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
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetFp8SparseMlaWorkspaceSize(
    infiniopFp8SparseMlaDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                              \
    case CASE:                                                                            \
        *size = reinterpret_cast<const op::fp8_sparse_mla::NAMESPACE::Descriptor *>(desc) \
                    ->workspaceSize();                                                    \
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
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopFp8SparseMla(
    infiniopFp8SparseMlaDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *query,
    const void *kv_cache,
    const void *indices,
    const void *topk_lens,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<const op::fp8_sparse_mla::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, query,                        \
                        kv_cache, indices, topk_lens, stream)
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
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyFp8SparseMlaDescriptor(
    infiniopFp8SparseMlaDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                          \
    case CASE:                                                                            \
        delete reinterpret_cast<const op::fp8_sparse_mla::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
