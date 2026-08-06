#ifndef __INFINIOP_SPARSE_FLASH_MLA_API_H__
#define __INFINIOP_SPARSE_FLASH_MLA_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopSparseFlashMla(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t query_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t topk_lens_desc,
    void *output,
    const void *query,
    const void *kv_cache,
    const void *indices,
    const void *topk_lens,
    float scale,
    void *stream);

#endif
