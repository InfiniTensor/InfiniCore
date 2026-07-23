#ifndef __INFINIOP_FP8_MLA_RMSNORM_CACHE_API_H__
#define __INFINIOP_FP8_MLA_RMSNORM_CACHE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFp8MlaRmsnormCacheDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFp8MlaRmsnormCacheDescriptor(
    infiniopHandle_t handle,
    infiniopFp8MlaRmsnormCacheDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t cache_desc,
    infiniopTensorDescriptor_t vendor_cache_desc,
    infiniopTensorDescriptor_t compressed_kv_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t rope_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    double eps);

__INFINI_C __export infiniStatus_t infiniopFp8MlaRmsnormCache(
    infiniopFp8MlaRmsnormCacheDescriptor_t desc,
    void *cache,
    void *vendor_cache,
    const void *compressed_kv,
    const void *norm_weight,
    const void *rope,
    const void *slot_mapping,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFp8MlaRmsnormCacheDescriptor(
    infiniopFp8MlaRmsnormCacheDescriptor_t desc);

#endif
