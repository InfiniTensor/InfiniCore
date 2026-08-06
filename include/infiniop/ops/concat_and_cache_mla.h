#ifndef __INFINIOP_CONCAT_AND_CACHE_MLA_API_H__
#define __INFINIOP_CONCAT_AND_CACHE_MLA_API_H__

#include "../operator_descriptor.h"

__INFINI_C __export infiniStatus_t infiniopConcatAndCacheMla(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t kv_c_desc,
    infiniopTensorDescriptor_t k_pe_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    const void *kv_c,
    const void *k_pe,
    void *kv_cache,
    const void *slot_mapping,
    void *stream);

#endif
