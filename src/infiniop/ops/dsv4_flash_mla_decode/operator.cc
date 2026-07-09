#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_flash_mla_decode.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_flash_mla_decode_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateDsv4FlashMlaDecodeDescriptor(infiniopHandle_t handle, infiniopDsv4FlashMlaDecodeDescriptor_t *desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t lse_desc, infiniopTensorDescriptor_t q_nope_desc, infiniopTensorDescriptor_t q_pe_desc, infiniopTensorDescriptor_t k_cache_desc, infiniopTensorDescriptor_t block_table_desc, infiniopTensorDescriptor_t cache_seqlens_desc, infiniopTensorDescriptor_t tile_scheduler_metadata_desc, infiniopTensorDescriptor_t num_splits_desc, int head_dim_v, float softmax_scale, bool causal) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_flash_mla_decode::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_flash_mla_decode::NAMESPACE::Descriptor **>(desc_ptr), out_desc, lse_desc, q_nope_desc, q_pe_desc, k_cache_desc, block_table_desc, cache_seqlens_desc, tile_scheduler_metadata_desc, num_splits_desc, head_dim_v, softmax_scale, causal)
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

__INFINI_C infiniStatus_t infiniopGetDsv4FlashMlaDecodeWorkspaceSize(infiniopDsv4FlashMlaDecodeDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                 \
    case CASE:                                                                                               \
        *size = reinterpret_cast<op::dsv4_flash_mla_decode::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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

__INFINI_C infiniStatus_t infiniopDsv4FlashMlaDecode(infiniopDsv4FlashMlaDecodeDescriptor_t desc, void *workspace, size_t workspace_size, void *out, void *lse, const void *q_nope, const void *q_pe, const void *k_cache, const void *block_table, const void *cache_seqlens, const void *tile_scheduler_metadata, const void *num_splits, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_flash_mla_decode::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, out, lse, q_nope, q_pe, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits, stream)
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

__INFINI_C infiniStatus_t infiniopDestroyDsv4FlashMlaDecodeDescriptor(infiniopDsv4FlashMlaDecodeDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                           \
    case CASE:                                                                             \
        delete reinterpret_cast<op::dsv4_flash_mla_decode::NAMESPACE::Descriptor *>(desc); \
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
